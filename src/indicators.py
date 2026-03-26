import pandas as pd
import pandas_ta as ta
import numpy as np

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Loads EURUSD data from CSV and preprocesses it by adding technical indicators.
    Expects columns: [Gmt time, Open, High, Low, Close, Volume].
    """
    df = pd.read_csv(csv_path, parse_dates=True, index_col='Gmt time')
    
    # Sort by date just in case
    df.sort_index(inplace=True)
    
    # --- Trend Indicators ---
    # MACD (Line, Signal, Hist)
    macd = ta.macd(df['Close'])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    
    # ADX and DMI
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    # EMA Crossovers
    df['ema_50'] = ta.ema(df['Close'], length=50)
    df['ema_200'] = ta.ema(df['Close'], length=200)
    # Check for NaN before subtraction if EMAs didn't calculate
    if 'ema_50' in df and 'ema_200' in df:
        df['ema_cross'] = df['ema_50'] - df['ema_200'] # Positive = Golden Cross territory

    # --- Volatility Indicators ---
    # Bollinger Bands (Width, %B included in default pandas_ta bbands)
    bb = ta.bbands(df['Close'], length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
    
    # ATR (Average True Range)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # --- Momentum Indicators ---
    # RSI
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
    
    # CCI (Commodity Channel Index)
    df['cci'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)

    # --- Volume Indicators ---
    # OBV (On-Balance Volume)
    df['obv'] = ta.obv(df['Close'], df['Volume'])
    
    # VWAP (Volume Weighted Average Price)
    # Note: VWAP usually resets daily. Pandas TA vwap might need 'anchor'. 
    # For simplified usage we can use rolling VWAP or just use default if compatible.
    # If error occurs (often due to date index issues), we wrap in try-except or check docs.
    try:
        df['vwap'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    except Exception:
        pass # VWAP can be finicky with index types

    # Volume Spikes (Volume > 2 * Moving Average of Volume)
    vol_ma = df['Volume'].rolling(window=20).mean()
    df['vol_spike'] = (df['Volume'] > 2 * vol_ma).astype(int)

    # --- Derived Interactions / Patterns ---
    # Pattern: Engulfing (Simplified)
    # Bullish Engulfing: Prev red, Curr green, Curr Open < Prev Close, Curr Close > Prev Open
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    curr_open = df['Open']
    curr_close = df['Close']
    
    is_prev_bear = prev_close < prev_open
    is_curr_bull = curr_close > curr_open
    
    engulfing_bull = (is_prev_bear & is_curr_bull & (curr_open < prev_close) & (curr_close > prev_open)).astype(int)
    df['pattern_engulfing_bull'] = engulfing_bull
    
    # Consecutive Candles
    # 3 consecutive bullish candles
    three_white_soldiers = ((curr_close > curr_open) & 
                            (df['Close'].shift(1) > df['Open'].shift(1)) & 
                            (df['Close'].shift(2) > df['Open'].shift(2))).astype(int)
    df['pattern_3_bulls'] = three_white_soldiers

    # --- Rolling Statistics ---
    # Rolling Returns (momentum proxy)
    df['roll_return_24h'] = df['Close'].pct_change(24)
    
    # Rolling Volatility (std dev of returns)
    df['roll_vol_24h'] = df['Close'].pct_change(1).rolling(24).std()

    # --- Derived Features ---
    # Momentum (Price change)
    df['mom_1h'] = df['Close'].pct_change(periods=1)
    df['mom_4h'] = df['Close'].pct_change(periods=4)
    
    # Slope of EMAs (normalized by price to keep scale reasonable)
    if 'ema_50' in df:
        df['ema_50_slope'] = df['ema_50'].diff() / df['Close']
    
    # Volatility Regime (Ratio of ATR to Price)
    if 'atr' in df:
        df['volatility_ratio'] = df['atr'] / df['Close']

    # --- v6.2 Feature Engineering (Sharpe 1.5) ---
    # Macro Trend (Price vs EMA200) - 1.0 if Bullish, -1.0 if Bearish
    if 'ema_200' in df:
        df['trend_ema200'] = np.where(df['Close'] > df['ema_200'], 1.0, -1.0)
    
    # RSI Regime - Overbought (1), Oversold (-1), Neutral (0)
    if 'rsi_14' in df:
        conditions = [
            (df['rsi_14'] > 70),
            (df['rsi_14'] < 30)
        ]
        choices = [1.0, -1.0]
        df['rsi_status'] = np.select(conditions, choices, default=0.0)
    
    # Drop any rows with NaN created by indicators (e.g. 200 period EMA)
    df.dropna(inplace=True)

    # Empty DataFrame guard
    if len(df) == 0:
        raise ValueError("DataFrame is empty after dropping NaN values. Ensure input data has enough rows (at least 200 for EMA-200).")

    return df