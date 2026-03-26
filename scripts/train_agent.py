import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import sys
import os

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Agent for Forex (v3.0)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3", "a2c"])
    parser.add_argument("--timesteps", type=int, default=300000, help="Total training timesteps (increased for stability)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate") # Will be used as base
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (increased for stability)")
    parser.add_argument("--data", type=str, default="data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data from {args.data}...")
    df = load_and_preprocess_data(args.data)
    
    # All algos now support continuous actions in v3.0 env
    # PPO/A2C can work with Box action space naturally.
    # Env action space is usually Box(-1, 1).
    continuous_actions = True 
    
    print(f"Initializing Environment (v4.0 - Benchmark Relative)...")
    env = Monitor(ForexTradingEnv(
        df=df,
        window_size=30,
        continuous_actions=continuous_actions,
        spread=0.0001,
        transaction_cost=0.00005
    ))
    
    # Wrapper for Normalization (Critical for PPO/SAC)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(
        vec_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.,
        gamma=0.99
    )
    
    print(f"Initializing {args.algo.upper()} Model with Robust Hyperparams...")
    
    # Shared Args
    model_kwargs = {
        "verbose": 1,
        "learning_rate": args.lr,
        "seed": 42
    }
    
    if args.algo == "ppo":
        model_kwargs.update({
            "batch_size": args.batch_size,
            "ent_coef": 0.02, # Healthy exploration
            "clip_range": 0.2, # Stable updates
            "n_steps": 2048,
            "gae_lambda": 0.95,
            "gamma": 0.99,
        })
        model = PPO("MlpPolicy", vec_env, **model_kwargs)
        
    elif args.algo == "sac":
        model_kwargs.update({
            "batch_size": args.batch_size,
            "ent_coef": "auto", # Auto-tune entropy
            "tau": 0.005,
        })
        model = SAC("MlpPolicy", vec_env, **model_kwargs)
        
    elif args.algo == "a2c":
        model_kwargs.update({
             "ent_coef": 0.01 
        })
        model = A2C("MlpPolicy", vec_env, **model_kwargs)
        
    elif args.algo == "td3":
        model_kwargs["batch_size"] = args.batch_size
        model = TD3("MlpPolicy", vec_env, **model_kwargs)
    
    print(f"Starting Training for {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps)
    
    model_name = f"model_{args.algo}_eurusd"
    model.save(model_name)
    vec_env.save(f"{model_name}_vecnormalize.pkl")
    print(f"Model saved to {model_name}")
    
    # No inline eval in train script, use test_agent.py for proper benchmarking
    print("Training Complete. Run 'test_agent.py' to evaluate.")

if __name__ == "__main__":
    main()