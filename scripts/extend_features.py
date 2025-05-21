import numpy as np
import pandas as pd

def extract_extended_features(price_series: pd.Series, volume_series: pd.Series):
    """
    price_series: pd.Series of float, 30-day price history
    volume_series: pd.Series of float, 30-day volume history
    returns a 1D numpy array of extracted features
    """

    # Ensure length
    if len(price_series) < 30:
        price_series = price_series.reindex(range(30), fill_value=price_series.iloc[-1])
        volume_series = volume_series.reindex(range(30), fill_value=volume_series.iloc[-1])

    # Basic returns
    returns = price_series.pct_change().fillna(0).values

    # Log returns
    log_returns = np.log(price_series / price_series.shift(1)).replace([np.inf, -np.inf], 0).fillna(0).values

    # Moving Averages
    ma5 = price_series.rolling(5).mean().iloc[-1]
    ma10 = price_series.rolling(10).mean().iloc[-1]
    ma20 = price_series.rolling(20).mean().iloc[-1]

    # Volatility (7-day rolling std of returns)
    volatility = pd.Series(returns).rolling(window=7).std().iloc[-1]
    momentum = price_series.diff().fillna(0).values
    log_volume = np.log1p(volume_series).fillna(0).values

    # Bollinger Bands width
    rolling_mean = price_series.rolling(20).mean().iloc[-1]
    rolling_std = price_series.rolling(20).std().iloc[-1]
    bb_width = 2 * rolling_std / rolling_mean if rolling_mean != 0 else 0

    # RSI (Relative Strength Index)
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(14).mean().iloc[-1]
    avg_loss = down.rolling(14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

    # MACD
    ema12 = price_series.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = price_series.ewm(span=26, adjust=False).mean().iloc[-1]
    macd = ema12 - ema26
    signal = price_series.ewm(span=9, adjust=False).mean().iloc[-1]
    macd_signal = signal

    # ADX (simplified version)
    high = price_series.max()
    low = price_series.min()
    adx = (high - low) / price_series.mean() if price_series.mean() != 0 else 0

    # Volume EMA
    vol_ema = volume_series.ewm(span=10, adjust=False).mean().iloc[-1]

    return np.concatenate([
        [returns[-1],
         log_returns[-1],
         ma5, ma10, ma20,
         volatility,
         bb_width,
         rsi,
         macd,
         macd_signal,
         adx,
         vol_ema],
        momentum[-30:],         # 길이 30
        log_volume[-30:]        # 길이 30
    ])

