import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_eurusd(
    start_date='2023-11-01',
    end_date='2025-11-30',
    timeframe_hours=1,
    initial_price=1.0800,
    volatility=0.0012,  # Increased to match ~9% annual vol of real data
    trend=0.0,          # Neutral trend by default
    mean_reversion=0.15, # Corresponding to -0.15 autocorrelation
    output_file='generated_test_data.csv'
):
    """
    Generate synthetic EUR/USD data using an AR(1) process to mimic real market properties.
    """
    
    # Generate datetime range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start, end=end, freq=f'{timeframe_hours}H')
    
    num_candles = len(date_range)
    print(f"Generating {num_candles} candles from {start_date} to {end_date} ({timeframe_hours}H timeframe)...")
    
    # Initialize arrays
    opens = np.zeros(num_candles)
    highs = np.zeros(num_candles)
    lows = np.zeros(num_candles)
    closes = np.zeros(num_candles)
    volumes = np.zeros(num_candles)
    
    # Generate log returns using AR(1) process
    # r_t = -k * r_{t-1} + sigma * epsilon
    # This creates the negative autocorrelation seen in real FX markets
    
    log_returns = np.zeros(num_candles)
    epsilons = np.random.randn(num_candles)
    
    # Burn-in period to stabilize process
    prev_return = 0
    for i in range(num_candles):
        # r_t = alpha * r_{t-1} + noise + drift
        # If we see -0.15 autocorrelation, alpha should be approx -0.15
        current_return = -mean_reversion * prev_return + volatility * epsilons[i] + trend
        log_returns[i] = current_return
        prev_return = current_return

    # Generate prices
    current_price = initial_price
    
    for i in range(num_candles):
        # Open price (mostly close of prev bar, with slight noise)
        opens[i] = current_price
        
        # Calculate Close
        ret = log_returns[i]
        close_price = current_price * np.exp(ret)
        
        # Simulate High/Low based on volatility within the period
        # Standard high/low estimation relative to Open/Close
        candle_vol = abs(ret) + volatility * np.random.rand()
        
        mx = max(current_price, close_price)
        mn = min(current_price, close_price)
        
        highs[i] = mx * (1 + candle_vol * 0.5)
        lows[i] = mn * (1 - candle_vol * 0.5)
        closes[i] = close_price
        
        # Random volume with some seasonality
        hour_factor = 1.0 + 0.5 * np.sin(2 * np.pi * date_range[i].hour / 24)
        volumes[i] = np.random.randint(1000, 50000) * hour_factor
        
        current_price = close_price
    
    # Create DataFrame
    df = pd.DataFrame({
        'Gmt time': date_range,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes.astype(int)
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Generated {num_candles} candles")
    print(f"✓ Price range: {df['Low'].min():.4f} - {df['High'].max():.4f}")
    print(f"✓ Saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_eurusd(
        start_date='2023-11-01',
        end_date='2025-11-30',
        timeframe_hours=1,
        initial_price=1.0800,
        volatility=0.0012,    # Tuned to match ~9% annualized vol
        trend=0.0,            # Flat trend
        mean_reversion=0.15,  # Matches ~-0.15 autocorrelation
        output_file='generated_test_data.csv'
    )
