# -*- coding: utf-8 -*-
"""
Enhanced feature engineering with technical analysis indicators.
Provides more predictive features for stock price direction.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Original 10 features for backward compatibility
FEATURES_10 = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20",
    "dd_20", "range_pct"
]

# Enhanced feature set with technical indicators
ENHANCED_FEATURES = [
    # Returns
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    # Volatility
    "vol_5", "vol_10", "vol_20",
    # Drawdown
    "dd_20",
    # Range
    "range_pct",
    # Technical indicators
    "rsi_14",
    "macd_signal",
    "bb_position",
    "momentum_10",
    "obv_change",
    "atr_14",
    "ema_ratio",
    "volume_sma_ratio",
]


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Neutral RSI for NaN


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Compute MACD signal line crossover indicator."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # Return normalized signal (positive = bullish, negative = bearish)
    return (macd_line - signal_line) / prices


def compute_bollinger_position(prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
    """Compute price position within Bollinger Bands (-1 to 1)."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Position from -1 (at lower band) to 1 (at upper band)
    bb_width = upper_band - lower_band
    position = (2 * (prices - lower_band) / bb_width.replace(0, np.nan)) - 1
    return position.clip(-1, 1).fillna(0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range normalized by close price."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr / close  # Normalize by price


def compute_obv_change(close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
    """Compute On-Balance Volume change rate."""
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()
    obv_change = obv.pct_change(period)
    return obv_change.replace([np.inf, -np.inf], 0).fillna(0)


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build enhanced feature set from OHLCV data.
    
    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        
    Returns:
        DataFrame with all features and date column
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure we have required columns
    required = ["close", "high", "low", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    
    out = pd.DataFrame()
    if "date" in df.columns:
        out["date"] = df["date"]
    
    # Returns
    for n in [1, 3, 5, 10, 20]:
        out[f"ret_{n}"] = close.pct_change(n)
    
    # Volatility
    r1 = close.pct_change()
    for n in [5, 10, 20]:
        out[f"vol_{n}"] = r1.rolling(n).std()
    
    # Drawdown
    roll_max = close.rolling(20).max()
    out["dd_20"] = (close / roll_max) - 1.0
    
    # Range
    out["range_pct"] = (high - low) / close
    
    # RSI
    out["rsi_14"] = (compute_rsi(close, 14) - 50) / 50  # Normalize to -1 to 1
    
    # MACD
    out["macd_signal"] = compute_macd(close)
    
    # Bollinger Bands position
    out["bb_position"] = compute_bollinger_position(close)
    
    # Momentum
    out["momentum_10"] = close.pct_change(10)
    
    # OBV change
    out["obv_change"] = compute_obv_change(close, volume)
    
    # ATR
    out["atr_14"] = compute_atr(high, low, close, 14)
    
    # EMA ratio (short vs long term trend)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["ema_ratio"] = (ema_12 / ema_26) - 1
    
    # Volume trend
    vol_sma = volume.rolling(20).mean()
    out["volume_sma_ratio"] = (volume / vol_sma) - 1
    
    # Clean up
    out = out.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Built enhanced features. Shape: {out.shape}")
    return out


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Create binary target: 1 if price goes up in next `horizon` days, 0 otherwise.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    close = df["close"].astype(float)
    future_return = close.shift(-horizon) / close - 1
    target = (future_return > 0).astype(int)
    return target


def prepare_train_data(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    horizon: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data with features and target.
    
    Args:
        df: Raw OHLCV DataFrame
        feature_cols: List of feature columns to use (default: ENHANCED_FEATURES)
        horizon: Prediction horizon in days
        
    Returns:
        X: Feature DataFrame
        y: Target Series
    """
    if feature_cols is None:
        feature_cols = ENHANCED_FEATURES
    
    features_df = build_enhanced_features(df)
    target = create_target(df, horizon)
    
    # Align and drop NaN
    combined = features_df.copy()
    combined["target"] = target
    combined = combined.dropna()
    
    X = combined[feature_cols]
    y = combined["target"]
    
    logger.info(f"Prepared training data. X: {X.shape}, y: {y.shape}")
    return X, y
