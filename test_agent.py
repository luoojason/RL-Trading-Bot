import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def calculate_sharpe(equity_curve):
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0: return 0.0
    return np.sqrt(24 * 252) * returns.mean() / returns.std()

def calculate_max_drawdown(equity_curve):
    equity = pd.Series(equity_curve)
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    return drawdown.min()

def run_sma_strategy(df, short_window=50, long_window=200):
    """
    Simulates a simple SMA Crossover strategy on the dataframe.
    """
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    
    # Use pre-calculated if available or calc fresh
    sma_short = df['Close'].rolling(window=short_window).mean()
    sma_long = df['Close'].rolling(window=long_window).mean()
    
    # Use .loc to avoid ChainedAssignmentError
    # Align shapes carefully
    mask = (sma_short[short_window:] > sma_long[short_window:])
    signals.loc[mask.index, 'signal'] = np.where(mask, 1.0, -1.0)
    # Shift signal to next bar (action at Close of T affects T+1 return)
    # Actually action at T Close captures return from T Close to T+1 Close.
    # So we align signal at T with return T->T+1.
    
    # Calculate returns
    market_returns = df['Close'].pct_change()
    strategy_returns = signals['signal'].shift(1) * market_returns
    
    equity_curve = [10000]
    for r in strategy_returns.dropna():
        equity_curve.append(equity_curve[-1] * (1 + r))
        
    return equity_curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3", "a2c"])
    parser.add_argument("--data", type=str, default="generated_test_data.csv")
    args = parser.parse_args()

    print(f"Loading test data from {args.data}...")
    df = load_and_preprocess_data(args.data)
    
    # v3.0 Env
    test_env = ForexTradingEnv(
        df=df,
        window_size=30,
        continuous_actions=True,
        spread=0.0,
        transaction_cost=0.0
    )
    
    # Load Normalization Stats
    stats_path = f"model_{args.algo}_eurusd_vecnormalize.pkl"
    vec_test_env = DummyVecEnv([lambda: test_env])
    try:
        vec_test_env = VecNormalize.load(stats_path, vec_test_env)
        vec_test_env.training = False 
        vec_test_env.norm_reward = False
        print(f"Loaded VecNormalize stats from {stats_path}")
    except Exception as e:
        print(f"Warning: Could not load VecNormalize stats: {e}. Testing without normalization.")

    model_name = f"model_{args.algo}_eurusd"
    print(f"Loading model {model_name}...")
    
    if args.algo == "ppo":
        model = PPO.load(model_name, env=vec_test_env)
    elif args.algo == "sac":
        model = SAC.load(model_name, env=vec_test_env)
    # Add others as needed
        
    obs = vec_test_env.reset()
    done = False
    equity_curve = []
    
    print("Running Agent Backtest...")
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_test_env.step(action)
        
        env_info = info[0]
        equity_curve.append(env_info['equity'])
        
        if done[0]:
            break
            
    # Benchmarks
    print("Running SMA Benchmark...")
    sma_curve = run_sma_strategy(df)
    # Align lengths roughly (SMA loses initial bars, env loses window_size)
    # Simple fix: Slice to common length or just plot overlay
    
    initial_price = df['Close'].iloc[0]
    bnh_curve = 10000 * (df['Close'] / initial_price).values
    bnh_curve = bnh_curve[test_env.window_size:] # Align with agent start
    
    # Metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - 10000) / 10000
    sharpe = calculate_sharpe(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    
    print(f"--- Results ({args.algo.upper()}) ---")
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Return: {total_return*100:.2f}% | Sharpe: {sharpe:.4f} | MaxDD: {max_dd*100:.2f}%")
    
    sma_final = sma_curve[-1]
    print(f"--- Benchmark (SMA) ---")
    print(f"Final Equity: {sma_final:.2f}")
    print(f"Return: {(sma_final-10000)/10000*100:.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label=f'Agent ({args.algo.upper()})', linewidth=2)
    plt.plot(bnh_curve, label='Buy & Hold', alpha=0.5, linestyle='--')
    
    # Adjust SMA plot to match length
    plt.plot(sma_curve[-len(equity_curve):], label='SMA Strategy', alpha=0.5, linestyle=':')
    
    plt.title(f"Backtest: Agent vs SMA vs B&H")
    plt.xlabel("Hours")
    plt.ylabel("Equity")
    plt.legend()
    plt.savefig("backtest_v3.png")
    print("Results plot saved to backtest_v3.png")

if __name__ == "__main__":
    main()