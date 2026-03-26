import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class ForexTradingEnv(gym.Env):
    """
    Forex Environment v5.2: Zero-Cost Absolute Reward with Leverage.
    Objective: Prove Alpha by maximizing PnL in frictionless environment with 5x Leverage.
    """
    
    def __init__(self, df, window_size=30, spread=0.0, transaction_cost=0.0,
                 continuous_actions=True, leverage=5.0):
        super(ForexTradingEnv, self).__init__()

        # Data validation
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(df) < 30:
            raise ValueError(f"DataFrame too short ({len(df)} rows). Need at least 30 rows.")

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.spread = spread
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        
        # Action Space: Target Position Ratio [-1, 1]
        self.continuous_actions = continuous_actions
        if self.continuous_actions:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3) 
            
        # Observation Space
        self.num_base_features = self.df.shape[1]
        self.num_features = self.num_base_features + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.num_features), 
            dtype=np.float32
        )
        
        # Date management
        if not np.issubdtype(self.df.index.dtype, np.datetime64):
             if 'Gmt time' in self.df.columns:
                 self.dates = pd.to_datetime(self.df['Gmt time'])
             else:
                 self.dates = pd.Series([pd.Timestamp('2000-01-01')] * len(self.df))
        else:
            self.dates = self.df.index.to_series()
            
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = 10000.0
        self.equity = 10000.0
        self.position = 0.0 
        self.entry_price = 0.0
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        end = self.current_step
        start = max(end - self.window_size, 0)
        
        valid_cols = self.df.select_dtypes(include=[np.number]).columns
        if 'Gmt time' in valid_cols: valid_cols = valid_cols.drop('Gmt time')
        
        base_obs = self.df.loc[start:end-1, valid_cols].values
        
        if len(base_obs) < self.window_size:
            padding = np.tile(base_obs[0], (self.window_size - len(base_obs), 1))
            base_obs = np.concatenate([padding, base_obs], axis=0)
            
        # Add State
        current_price = self.df.iloc[end-1]['Close']
        if self.entry_price > 0 and self.position != 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price * np.sign(self.position)
        else:
            pnl_pct = 0.0
            
        pos_channel = np.full((self.window_size, 1), self.position, dtype=np.float32)
        pnl_channel = np.full((self.window_size, 1), pnl_pct, dtype=np.float32)
        
        obs = np.hstack([base_obs, pos_channel, pnl_channel])
        return obs.astype(np.float32)
        
    def step(self, action):
        prev_equity = self.equity
        current_idx = self.current_step

        # Array bounds check
        if self.current_step >= len(self.df):
            # Episode is done
            terminated = True
            obs = self._get_observation()
            info = {'equity': self.equity, 'position': self.position}
            return obs, 0.0, terminated, False, info

        current_price = self.df.iloc[current_idx]['Close']
        
        # --- 1. Action & Costs ---
        if self.continuous_actions:
            target_pos = float(action[0])
        else:
            mapping = {0: -1.0, 1: 0.0, 2: 1.0}
            target_pos = mapping[action]
        
        # v6.2 Simple Leverage (5x)
        target_pos = np.clip(target_pos, -1.0, 1.0) * self.leverage
        
        # Turnover & Cost
        turnover = abs(target_pos - self.position)
        
        # Apply Cost (Only if spread/cost > 0)
        if self.transaction_cost > 0 or self.spread > 0:
            cost_val = self.equity * turnover * (self.transaction_cost + self.spread/2)
            self.equity -= cost_val
        
        # --- 2. Step Physics ---
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if not terminated:
            next_price = self.df.iloc[self.current_step]['Close']

            # Agent Return
            step_return = (next_price - current_price) / max(current_price, 1e-10)
            self.equity += self.equity * target_pos * step_return
            
        # --- 3. Reward Calculation (Absolute Log Return) ---
        safe_equity = max(self.equity, 1e-6)
        safe_prev_equity = max(prev_equity, 1e-6)
        
        # Primary Signal: Absolute Log Return (Growth)
        reward = np.log(safe_equity / safe_prev_equity)
        
        # Update State
        self.position = target_pos
        if turnover > 0: self.entry_price = current_price
        
        obs = self._get_observation()
        info = {
            'equity': self.equity,
            'position': self.position
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Eq: {self.equity:.2f}, Pos: {self.position:.2f}")