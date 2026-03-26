
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_data(filepath):
    # Try loading with standard pandas read_csv (expects specific format from indicators.py usually, but we just want stats)
    # The files index column might vary (Gmt time vs Date)
    try:
        df = pd.read_csv(filepath)
        # Identify time column
        time_col = 'Gmt time' if 'Gmt time' in df.columns else 'Date'
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calc_stats(df, name):
    print(f"--- Stats for {name} ---")
    if df is None:
        print("Data not found.")
        return
    
    close = df['Close']
    returns = close.pct_change().dropna()
    
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    volatility = returns.std()
    annualized_vol = volatility * np.sqrt(24 * 252) # Assuming 1H data
    skew = returns.skew()
    kurt = returns.kurtosis()
    autocorr = returns.autocorr(lag=1)
    
    print(f"Start Price: {close.iloc[0]:.5f}")
    print(f"End Price:   {close.iloc[-1]:.5f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Volatility (std): {volatility*100:.4f}%")
    print(f"Ann. Volatility:  {annualized_vol*100:.2f}%")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurt:.4f}")
    print(f"Autocorrelation (Lag-1): {autocorr:.4f}")
    print("-" * 30)

def main():
    real_data_path = os.path.join(PROJECT_ROOT, "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv")
    gen_data_path = os.path.join(PROJECT_ROOT, "generated_test_data.csv")
    
    df_real = load_data(real_data_path)
    df_gen = load_data(gen_data_path)
    
    calc_stats(df_real, "REAL DATA (Test Set)")
    calc_stats(df_gen, "GENERATED DATA")

if __name__ == "__main__":
    main()
