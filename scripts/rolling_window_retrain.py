"""
Rolling Window Retraining Pipeline
Implements continual learning with rolling window strategy for production
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


class RollingWindowRetrainer:
    """Rolling window retrain

ing for continual learning"""
    
    def __init__(self, 
                 data_dir='data/historical',
                 window_months=24,
                 output_dir='models/rolling_window'):
        self.data_dir = data_dir
        self.window_months = window_months
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.retraining_log = []
        
    def load_all_data(self):
        """Load all available historical data"""
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        all_data = []
        for filename in all_files:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            all_data.append(df)
        
        # Combine and sort
        combined = pd.concat(all_data, ignore_index=True)
        combined['Gmt time'] = pd.to_datetime(combined['Gmt time'])
        combined = combined.sort_values('Gmt time').drop_duplicates(subset=['Gmt time']).reset_index(drop=True)
        
        print(f"✓ Loaded {len(combined)} total hours from {len(all_files)} files")
        print(f"  Date range: {combined['Gmt time'].min()} to {combined['Gmt time'].max()}")
        
        return combined
    
    def get_rolling_windows(self, full_data, test_months=1):
        """Generate rolling windows for walk-forward validation"""
        # Convert to datetime if needed
        full_data['Gmt time'] = pd.to_datetime(full_data['Gmt time'])
        
        # Start date: earliest + training window
        start_date = full_data['Gmt time'].min() + timedelta(days=self.window_months * 30)
        end_date = full_data['Gmt time'].max()
        
        windows = []
        current_test_start = start_date
        
        while current_test_start + timedelta(days=test_months * 30) <= end_date:
            # Define window bounds
            test_end = current_test_start + timedelta(days=test_months * 30)
            train_end = current_test_start
            train_start = train_end - timedelta(days=self.window_months * 30)
            
            # Extract data
            train_data = full_data[
                (full_data['Gmt time'] >= train_start) &
                (full_data['Gmt time'] < train_end)
            ].copy()
            
            test_data = full_data[
                (full_data['Gmt time'] >= current_test_start) &
                (full_data['Gmt time'] < test_end)
            ].copy()
            
            if len(train_data) > 1000 and len(test_data) > 100:  # Minimum data requirements
                windows.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': current_test_start,
                    'test_end': test_end,
                    'train_data': train_data,
                    'test_data': test_data
                })
            
            # Move forward by test_months
            current_test_start += timedelta(days=test_months * 30)
        
        print(f"✓ Created {len(windows)} rolling windows")
        return windows
    
    def train_on_window(self, train_data, window_id, timesteps=200000):
        """Train model on a specific window"""
        print(f"\nTraining model on window {window_id}...")
        print(f"  Training data: {len(train_data)} hours")
        
        # Preprocess
        train_data_processed = load_and_preprocess_data(
            train_data,  # Pass DataFrame directly
            is_dataframe=True
        )
        
        # Create environment
        env = Monitor(ForexTradingEnv(
            df=train_data_processed,
            window_size=30,
            leverage=5.0,
            spread=0.0,
            transaction_cost=0.0
        ))
        
        # Wrap and normalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0
        )
        
        # Create SAC model
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=0.0003,
            buffer_size=100000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            verbose=0
        )
        
        # Train
        print(f"  Training for {timesteps} steps...")
        model.learn(total_timesteps=timesteps)
        
        # Save
        model_path = f"{self.output_dir}/model_window_{window_id}"
        model.save(model_path)
        vec_env.save(f"{model_path}_vecnormalize.pkl")
        
        print(f"  ✓ Model saved to {model_path}")
        
        return model, vec_env
    
    def test_on_window(self, model, vec_env_stats, test_data, window_id):
        """Test model on test window"""
        print(f"  Testing on window {window_id}...")
        
        # Preprocess test data
        test_data_processed = load_and_preprocess_data(
            test_data,
            is_dataframe=True
        )
        
        # Create test environment
        test_env = ForexTradingEnv(
            df=test_data_processed,
            window_size=30,
            leverage=5.0,
            spread=0.0,
            transaction_cost=0.0
        )
        
        vec_test_env = DummyVecEnv([lambda: test_env])
        
        # Load normalization stats from training
        vec_test_env = VecNormalize.load(
            f"{self.output_dir}/model_window_{window_id}_vecnormalize.pkl",
            vec_test_env
        )
        vec_test_env.training = False
        vec_test_env.norm_reward = False
        
        # Run backtest
        obs = vec_test_env.reset()
        equity_curve = []
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_test_env.step(action)
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
        
        print(f"  Sharpe: {sharpe:.2f} | Return: {total_return*100:.1f}% | MaxDD: {max_dd*100:.1f}%")
        
        return {
            'sharpe': sharpe,
            'return': total_return,
            'max_dd': max_dd,
            'final_equity': equity_curve[-1]
        }
    
    def run_walk_forward_validation(self, timesteps_per_window=200000):
        """Run complete walk-forward validation"""
        print("="*60)
        print("ROLLING WINDOW WALK-FORWARD VALIDATION")
        print("="*60)
        
        # Load all data
        full_data = self.load_all_data()
        
        # Generate windows
        windows = self.get_rolling_windows(full_data, test_months=1)
        
        if len(windows) == 0:
            print("✗ No windows created. Need more historical data.")
            return
        
        # Train and test on each window
        all_results = []
        
        for idx, window in enumerate(windows):
            print(f"\n{'─'*60}")
            print(f"Window {idx+1}/{len(windows)}")
            print(f"  Train: {window['train_start'].date()} to {window['train_end'].date()}")
            print(f"  Test:  {window['test_start'].date()} to {window['test_end'].date()}")
            print(f"{'─'*60}")
            
            # Train
            model, vec_env = self.train_on_window(
                window['train_data'],
                window_id=idx,
                timesteps=timesteps_per_window
            )
            
            # Test
            result = self.test_on_window(
                model,
                vec_env,
                window['test_data'],
                window_id=idx
            )
            
            result['window_id'] = idx
            result['test_start'] = window['test_start']
            result['test_end'] = window['test_end']
            
            all_results.append(result)
            
            # Log
            self.retraining_log.append({
                'timestamp': datetime.now(),
                'window_id': idx,
                **result
            })
        
        # Save results
        self.save_walk_forward_results(all_results)
        
        return all_results
    
    def save_walk_forward_results(self, results):
        """Save and analyze walk-forward results"""
        df = pd.DataFrame(results)
        df.to_csv('walk_forward_results.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION RESULTS")
        print(f"{'='*60}")
        print(df[['window_id', 'sharpe', 'return', 'max_dd']].to_string(index=False))
        print(f"{'='*60}")
        
        # Aggregate statistics
        print(f"\nAggregate Performance:")
        print(f"  Average Sharpe:  {df['sharpe'].mean():.4f}")
        print(f"  Sharpe Std Dev:  {df['sharpe'].std():.4f}")
        print(f"  Min Sharpe:      {df['sharpe'].min():.4f}")
        print(f"  Max Sharpe:      {df['sharpe'].max():.4f}")
        print(f"  Windows > 2.0:   {sum(df['sharpe'] > 2.0)}/{len(df)}")
        print(f"  Windows < 0:     {sum(df['sharpe'] < 0)}/{len(df)}")
        
        # Production readiness
        avg_sharpe = df['sharpe'].mean()
        consistency = df['sharpe'].std()
        negative_windows_pct = sum(df['sharpe'] < 0) / len(df)
        
        print(f"\n{'='*60}")
        print(f"PRODUCTION READINESS:")
        print(f"{'='*60}")
        
        ready = True
        
        if avg_sharpe > 2.0:
            print(f"  ✓ Average Sharpe > 2.0: {avg_sharpe:.2f}")
        else:
            print(f"  ✗ Average Sharpe < 2.0: {avg_sharpe:.2f}")
            ready = False
        
        if consistency < 1.5:
            print(f"  ✓ Consistency (std < 1.5): {consistency:.2f}")
        else:
            print(f"  ✗ High variance: {consistency:.2f}")
            ready = False
        
        if negative_windows_pct < 0.2:
            print(f"  ✓ Negative windows < 20%: {negative_windows_pct*100:.0f}%")
        else:
            print(f"  ✗ Too many negative windows: {negative_windows_pct*100:.0f}%")
            ready = False
        
        if ready:
            print(f"\n  ✅ READY FOR PRODUCTION DEPLOYMENT")
        else:
            print(f"\n  ⚠️  NEEDS FURTHER OPTIMIZATION")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Initialize retrainer
    retrainer = RollingWindowRetrainer(
        data_dir='data/historical',
        window_months=24,  # 2-year training window
        output_dir='models/rolling_window'
    )
    
    # Run walk-forward validation
    results = retrainer.run_walk_forward_validation(timesteps_per_window=200000)
    
    if results:
        print(f"\n✅ Walk-forward validation complete!")
        print(f"   Tested {len(results)} windows")
        print(f"   Results saved to walk_forward_results.csv")
    else:
        print(f"\n⚠️  Validation failed - check data availability")
