# -*- coding: utf-8 -*-
"""
Feature builder for the packaged Hybrid model.

The model expects these 10 features in this order:
  ret_1, ret_3, ret_5, ret_10, ret_20,
  vol_5, vol_10, vol_20,
  dd_20,
  range_pct
"""
import numpy as np
import pandas as pd

FEATURES_10 = ["ret_1","ret_3","ret_5","ret_10","ret_20","vol_5","vol_10","vol_20","dd_20","range_pct"]

def build_features_from_candles(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low  = df["low"].astype(float)

    out = pd.DataFrame({"date": df["date"].astype(str)})

    for n in [1,3,5,10,20]:
        out[f"ret_{n}"] = close.pct_change(n)

    r1 = close.pct_change()
    for n in [5,10,20]:
        out[f"vol_{n}"] = r1.rolling(n).std()

    roll_max = close.rolling(20).max()
    out["dd_20"] = (close / roll_max) - 1.0

    out["range_pct"] = (high - low) / close

    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out
