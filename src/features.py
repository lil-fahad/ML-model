import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

CORE_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20",
    "px_sma10", "px_sma20", "px_sma50",
    "slope_sma20", "slope_sma50",
    "dd_20", "dd_50",
    "range_pct"
]

LSTM_FEATURES = ["ret_1", "vol_5", "dd_20"]
LOOKBACK = 20


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build core features from OHLCV data.
    
    Args:
        df: DataFrame with at minimum a 'close' column, optionally 'high' and 'low'
        
    Returns:
        DataFrame with engineered features
        
    Raises:
        ValueError: If required columns are missing
    """
    df = df.copy()
    
    # Normalize column names to lowercase
    col_map = {c.lower(): c for c in df.columns}
    if "close" in col_map:
        df = df.rename(columns={col_map["close"]: "Close"})
    if "high" in col_map:
        df = df.rename(columns={col_map["high"]: "High"})
    if "low" in col_map:
        df = df.rename(columns={col_map["low"]: "Low"})
    
    if "Close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    px = df["Close"].astype(float)
    
    # Returns over different periods
    df["ret_1"] = px.pct_change(1)
    df["ret_3"] = px.pct_change(3)
    df["ret_5"] = px.pct_change(5)
    df["ret_10"] = px.pct_change(10)
    df["ret_20"] = px.pct_change(20)
    
    # Volatility measures
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    
    # Moving averages
    df["sma10"] = px.rolling(10).mean()
    df["sma20"] = px.rolling(20).mean()
    df["sma50"] = px.rolling(50).mean()
    
    # Price relative to moving averages
    df["px_sma10"] = px / df["sma10"] - 1
    df["px_sma20"] = px / df["sma20"] - 1
    df["px_sma50"] = px / df["sma50"] - 1
    
    # Moving average slopes
    df["slope_sma20"] = df["sma20"].diff(5)
    df["slope_sma50"] = df["sma50"].diff(5)
    
    # Drawdown (negative distance from rolling max)
    df["dd_20"] = px / px.rolling(20).max() - 1
    df["dd_50"] = px / px.rolling(50).max() - 1
    
    # Range percentage
    if "High" in df.columns and "Low" in df.columns:
        df["range_pct"] = (df["High"] - df["Low"]) / px
    else:
        # Fallback: use 2x volatility as proxy for range
        df["range_pct"] = df["vol_5"] * 2
    
    logger.info(f"Built features. Shape: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features by building and cleaning them.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Tuple of (X: feature DataFrame, df_clean: cleaned full DataFrame)
    """
    df = build_features(df)
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        logger.warning("No valid samples after feature engineering")
        return pd.DataFrame(), df_clean
    
    X = df_clean[CORE_FEATURES]
    logger.info(f"Prepared features. X shape: {X.shape}")
    return X, df_clean


def get_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the latest row of features after building and cleaning.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Single-row DataFrame with latest features, or empty DataFrame if none available
    """
    df = build_features(df)
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        logger.warning("No valid samples after feature engineering")
        return pd.DataFrame()
    
    return df_clean[CORE_FEATURES].iloc[[-1]]
