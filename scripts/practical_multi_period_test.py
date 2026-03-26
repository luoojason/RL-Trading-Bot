"""
Practical Multi-Period Tester
Uses available test data and splits it into periods for validation
"""

import pandas as pd
import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def split_test_data_into_periods(data_file='data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv'):
    """Split existing test data into multiple periods for robustness testing"""
    
    print("Loading test data...")
    df = pd.read_csv(data_file)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'])
    
    total_hours = len(df)
    print(f"Total hours available: {total_hours}")
    print(f"Date range: {df['Gmt time'].min()} to {df['Gmt time'].max()}")
    
    # Split into 4 quarters (Q1-Q4)
    quarter_size = total_hours // 4
    
    # Actual date range is 2020-2021 based on the data
    period_labels = [
        "Q3_2020",  # Jul-Aug 2020
        "Q4_2020",  # Sep-Oct 2020
        "Q1_2021",  # Nov-Dec 2020
        "Q4_2021"   # Jan 2021
    ]
    
    periods = []
    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else total_hours
        
        period_df = df.iloc[start_idx:end_idx].copy()
        
        period_name = period_labels[i]
        filename = f"data/test_period_{period_name}.csv"
        
        # Create data directory if needed
        os.makedirs('data', exist_ok=True)
        period_df.to_csv(filename, index=False)
        
        periods.append({
            'name': period_name,
            'file': filename,
            'start': period_df['Gmt time'].min(),
            'end': period_df['Gmt time'].max(),
            'hours': len(period_df)
        })
        
        print(f"  {period_name}: {len(period_df)} hours ({period_df['Gmt time'].min().date()} to {period_df['Gmt time'].max().date()})")
    
    return periods


def test_model_on_period(model_path, normalize_path, data_file, period_name):
    """Test model on a specific period"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {period_name}")
    print(f"{'='*60}")
    
    # Load data
    df = load_and_preprocess_data(data_file)
    print(f"Loaded {len(df)} hours")
    
    # Create environment
    test_env = ForexTradingEnv(
        df=df,
        window_size=30,
        leverage=5.0,
        spread=0.0,
        transaction_cost=0.0
    )
    
    vec_env = DummyVecEnv([lambda: test_env])
    
    # Load normalization
    try:
        vec_env = VecNormalize.load(normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    except:
        print("Warning: Could not load normalization stats")
    
    # Load model
    model = SAC.load(model_path, env=vec_env)
    
    # Run backtest
    obs = vec_env.reset()
    equity_curve = []
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        equity_curve.append(info[0]['equity'])
        
        if done[0]:
            break
    
    # Calculate metrics
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if returns.std() > 0 else 0
    total_return = (equity_curve[-1] - 10000) / 10000
    
    cummax = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - cummax) / cummax
    max_dd = drawdown.min()
    
    print(f"\nResults:")
    print(f"  Sharpe:  {sharpe:.4f}")
    print(f"  Return:  {total_return*100:.2f}%")
    print(f"  MaxDD:   {max_dd*100:.2f}%")
    print(f"  Final:   ${equity_curve[-1]:.2f}")
    
    return {
        'period': period_name,
        'sharpe': sharpe,
        'return': total_return,
        'max_dd': max_dd,
        'final_equity': equity_curve[-1],
        'hours': len(equity_curve)
    }


def main():
    print("="*60)
    print("PRACTICAL MULTI-PERIOD VALIDATION")
    print("Testing SAC model on different time periods")
    print("="*60)
    
    # Split test data into periods
    periods = split_test_data_into_periods()
    
    # Test on each period
    results = []
    
    for period in periods:
        result = test_model_on_period(
            model_path='model_sac_eurusd',
            normalize_path='model_sac_eurusd_vecnormalize.pkl',
            data_file=period['file'],
            period_name=period['name']
        )
        results.append(result)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('multi_period_validation_results.csv', index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(df_results[['period', 'sharpe', 'return', 'max_dd']].to_string(index=False))
    print(f"{'='*60}")
    
    print(f"\nAggregate Stats:")
    print(f"  Average Sharpe: {df_results['sharpe'].mean():.4f}")
    print(f"  Sharpe Std Dev: {df_results['sharpe'].std():.4f}")
    print(f"  Min Sharpe:     {df_results['sharpe'].min():.4f}")
    print(f"  Max Sharpe:     {df_results['sharpe'].max():.4f}")
    
    # Check consistency
    if df_results['sharpe'].min() > 2.0:
        print(f"\n  ✅ EXCELLENT: All periods Sharpe > 2.0 (very stable)")
    elif df_results['sharpe'].min() > 1.0:
        print(f"\n  ✓ GOOD: All periods Sharpe > 1.0 (stable)")
    elif df_results['sharpe'].min() > 0:
        print(f"\n  ⚠️  MODERATE: All periods profitable but inconsistent")
    else:
        print(f"\n  ✗ UNSTABLE: Some periods unprofitable")
    
    print(f"\n✓ Results saved to multi_period_validation_results.csv")


if __name__ == "__main__":
    main()
