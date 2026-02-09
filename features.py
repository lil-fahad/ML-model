import numpy as np
import pandas as pd
from typing import Tuple
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
    df = df.copy()
    
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
    
    df["ret_1"] = px.pct_change(1)
    df["ret_3"] = px.pct_change(3)
    df["ret_5"] = px.pct_change(5)
    df["ret_10"] = px.pct_change(10)
    df["ret_20"] = px.pct_change(20)
    
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    
    df["sma10"] = px.rolling(10).mean()
    df["sma20"] = px.rolling(20).mean()
    df["sma50"] = px.rolling(50).mean()
    
    df["px_sma10"] = px / df["sma10"] - 1
    df["px_sma20"] = px / df["sma20"] - 1
    df["px_sma50"] = px / df["sma50"] - 1
    
    df["slope_sma20"] = df["sma20"].diff(5)
    df["slope_sma50"] = df["sma50"].diff(5)
    
    df["dd_20"] = px / px.rolling(20).max() - 1
    df["dd_50"] = px / px.rolling(50).max() - 1
    
    if "High" in df.columns and "Low" in df.columns:
        df["range_pct"] = (df["High"] - df["Low"]) / px
    else:
        df["range_pct"] = df["vol_5"] * 2
    
    logger.info(f"Built features. Shape: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = build_features(df)
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        return pd.DataFrame(), df_clean
    
    X = df_clean[CORE_FEATURES]
    logger.info(f"Prepared features. X shape: {X.shape}")
    return X, df_clean


def get_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(df)
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        return pd.DataFrame()
    
    return df_clean[CORE_FEATURES].iloc[[-1]]
