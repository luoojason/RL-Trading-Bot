"""
Unified Data Ingestion Module
Standardized data downloading and preprocessing for forex trading models
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pandas_ta as ta


class ForexDataManager:
    """
    Centralized data management for forex trading
    Handles downloading, caching, and preprocessing of historical data
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_data(self, symbol='EUR/USD', start_date='2020-01-01', 
                     end_date=None, interval='1h', force_refresh=False):
        """
        Download historical forex data from Yahoo Finance
        
        Parameters:
        -----------
        symbol : str
            Currency pair (e.g., 'EUR/USD', 'GBP/USD')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (default: today)
        interval : str
            Timeframe: '1h', '1d', '1wk', '1mo'
        force_refresh : bool
            Force re-download even if cached
        
        Returns:
        --------
        pd.DataFrame with columns: [Gmt time, Open, High, Low, Close, Volume]
        """
        
        # Convert symbol to yfinance ticker
        ticker = symbol.replace('/', '') + '=X'
        
        # Set end date if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache
        cache_file = self.raw_dir / f"{symbol.replace('/', '')}_{interval}_{start_date}_{end_date}.csv"
        
        if cache_file.exists() and not force_refresh:
            print(f"✓ Loading cached data: {cache_file.name}")
            return pd.read_csv(cache_file)
        
        # Download from Y Finance
        print(f"Downloading {symbol} data ({interval} from {start_date} to {end_date})...")
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, 
                           interval=interval, progress=False)
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Format data
            df = df.reset_index()
            
            # Flatten multi-level columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Rename columns
            time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df = df.rename(columns={time_col: 'Gmt time'})
            
            # Select required columns
            df = df[['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Save cache
            df.to_csv(cache_file, index=False)
            print(f"✓ Downloaded {len(df)} {interval} bars")
            print(f"  Range: {df['Gmt time'].min()} to {df['Gmt time'].max()}")
            print(f"  Cached to: {cache_file}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error downloading {symbol}: {e}")
            return None
    
    def add_indicators(self, df):
        """
        Add technical indicators to OHLCV dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLCV data
        
        Returns:
        --------
        pd.DataFrame with technical indicators added
        """
        
        df = df.copy()
        
        # MACD
        macd = ta.macd(df['Close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        # RSI
        df['rsi_14'] = ta.rsi(df['Close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (volatility)
        df['atr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['volatility_ratio'] = df['atr_14'] / df['Close']
        
        # EMAs
        df['ema_50'] = ta.ema(df['Close'], length=50)
        df['ema_200'] = ta.ema(df['Close'], length=200)
        
        # ADX (trend strength)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX_14'] = adx['ADX_14']
        
        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # Volume
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Custom features
        df['trend_ema200'] = np.where(df['Close'] > df['ema_200'], 1, -1)
        df['rsi_status'] = np.where(df['rsi_14'] > 70, 1, 
                                   np.where(df['rsi_14'] < 30, -1, 0))
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def prepare_for_training(self, symbol='EUR/USD', start_date='2020-01-01',
                            end_date=None, interval='1h', save=True):
        """
        Download and preprocess data ready for model training
        
        Returns:
        --------
        pd.DataFrame ready for RL environment
        """
        
        # Download raw data
        raw_data = self.download_data(symbol, start_date, end_date, interval)
        
        if raw_data is None:
            return None
        
        # Add indicators
        processed_data = self.add_indicators(raw_data)
        
        # Save processed data
        if save:
            filename = f"{symbol.replace('/', '')}_{interval}_{start_date}_{end_date}_processed.csv"
            filepath = self.processed_dir / filename
            processed_data.to_csv(filepath, index=False)
            print(f"✓ Saved processed data: {filepath}")
        
        return processed_data


# Convenience functions for backward compatibility
def load_and_preprocess_data(filepath_or_df, is_dataframe=False):
    """Load and preprocess data (legacy function)"""
    if is_dataframe:
        df = filepath_or_df.copy()
    else:
        df = pd.read_csv(filepath_or_df)
    
    manager = ForexDataManager()
    return manager.add_indicators(df)


if __name__ == "__main__":
    # Example usage
    manager = ForexDataManager()
    
    # Download and process EUR/USD hourly data
    df = manager.prepare_for_training(
        symbol='EUR/USD',
        start_date='2020-01-01',
        end_date='2023-12-31',
        interval='1h'
    )
    
    if df is not None:
        print(f"\n✅ Data ready for training:")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['Gmt time'].min()} to {df['Gmt time'].max()}")
