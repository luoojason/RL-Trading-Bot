import sys
from pathlib import Path
import argparse
import os
import yfinance as yf

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import pandas as pd

def download_real_data(ticker="EURUSD=X", period="1mo", interval="1h", save_path="data/real_market_data.csv"):
    print(f"Downloading real data for {ticker} ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        raise ValueError("No data downloaded. Check ticker or internet connection.")
    
    # Reset index to get Date/Time as a column
    df.reset_index(inplace=True)
    
    # Flatten multi-level columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Rename time column to 'Gmt time' to match indicators.py expectation
    # yfinance uses 'Datetime' or 'Date' depending on interval
    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    if time_col in df.columns:
        df.rename(columns={time_col: 'Gmt time'}, inplace=True)
    else:
        # Fallback: check validation
        print("Warning: Could not identify time column. Columns found:", df.columns)
        
    # Ensure columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            # yfinance might generate 'Adj Close' but we need 'Close'. 
            # Usually 'Close' is present.
            print(f"Warning: Missing column {col}")

    # Handle flat zero volume (common in Forex YF data) to preventing indicator crashes (VWAP)
    if (df['Volume'] == 0).all():
        print("Warning: Volume is all 0. Setting to 1 to allow indicator caalculation.")
        df['Volume'] = 1

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✓ Data saved to {save_path}")
    return save_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="online", choices=["online", "local"], help="Data source: 'online' (yfinance) or 'local' (file)")
    parser.add_argument("--ticker", type=str, default="EURUSD=X", help="Ticker to download")
    parser.add_argument("--period", type=str, default="1mo", help="Data period (e.g. 1mo, 1y)")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (e.g. 1h, 1d)")
    parser.add_argument("--file", type=str, default="data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv", help="Path to local file")
    
    args = parser.parse_args()

    print("Starting SAC evaluation...")

    # -----------------------
    # 1. PREPARE TEST DATA
    # -----------------------
    data_path = args.file
    
    if args.source == "online":
        data_path = download_real_data(args.ticker, args.period, args.interval)

    print(f"Loading test data from {data_path}...")
    try:
        test_df = load_and_preprocess_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"✓ Loaded {len(test_df)} bars of test data")

    # -----------------------
    # 2. LOAD TRAINED MODEL
    # -----------------------
    print("\nLoading trained SAC model...")
    # Make robust path
    model_dir = "models"
    stats_path = os.path.join(model_dir, "model_sac_eurusd_vecnormalize.pkl")
    model_path = os.path.join(model_dir, "model_sac_eurusd")
    
    # Create env wrapper exactly as training
    vec_env_wrapper = VecNormalize.load(stats_path, 
                                         DummyVecEnv([lambda: ForexTradingEnv(
                                             df=test_df,
                                             window_size=30,
                                             leverage=5.0
                                         )]))
    vec_env_wrapper.training = False
    vec_env_wrapper.norm_reward = False

    model = SAC.load(model_path, env=vec_env_wrapper)
    print("✓ Model loaded successfully")

    # -----------------------
    # 3. RUN EVALUATION
    # -----------------------
    print("\nRunning backtest...")
    obs = vec_env_wrapper.reset()
    equity_curve = []
    actions_taken = []
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env_wrapper.step(action)
        equity_curve.append(info[0]['equity'])
        actions_taken.append(action[0][0])
        
        # Stop if we hit end of data (env might loop or specific done handling needed)
        # DummyVecEnv usually handles auto-reset. 
        # But we want one pass. 
        # ForexTradingEnv resets when index >= len(df) - 1.
        # We can check info['idx'] if available or just check done.
        # done[0] is True on reset. But we want to preserve equity_curve.
        
        if done[0]:
            break

    # -----------------------
    # 4. CALCULATE METRICS
    # -----------------------
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    import pandas as pd
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if returns.std() > 0 else 0
    
    initial_equity = 10000 # default
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_equity) / initial_equity

    cummax = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - cummax) / cummax
    max_dd = drawdown.min()

    print(f"Sharpe Ratio:     {sharpe:.4f}")
    print(f"Total Return:     {total_return*100:.2f}%")
    print(f"Max Drawdown:     {max_dd*100:.2f}%")
    print(f"Final Equity:     ${final_equity:,.2f}")
    print(f"Starting Equity:  ${initial_equity:,.2f}")
    print(f"Profit/Loss:      ${final_equity - initial_equity:,.2f}")
    print(f"Avg Position:     {np.mean(np.abs(actions_taken)):.2f}")
    print(f"Bars Tested:      {len(equity_curve)}")
    print("="*60)

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        'bar': range(len(equity_curve)),
        'equity': equity_curve,
        'action': actions_taken
    })
    results_path = 'results/evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")

if __name__ == "__main__":
    main()