#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download historical stock data for multiple tickers using yfinance.

Usage:
    python scripts/fetch_data.py                  # Download all default tickers
    python scripts/fetch_data.py AAPL MSFT GOOG   # Download specific tickers
    python scripts/fetch_data.py --period max      # Use a specific period (default: max)
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required. Install with: pip install yfinance")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# A broad list of popular, liquid US-listed tickers spanning different sectors
DEFAULT_TICKERS: List[str] = [
    # --- Technology ---
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ADBE", "NFLX", "ORCL", "CSCO", "AVGO", "QCOM", "TXN", "IBM", "NOW", "SHOP",
    # --- Finance ---
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
    # --- Healthcare ---
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
    # --- Energy ---
    "XOM", "CVX", "COP", "SLB", "EOG",
    # --- Consumer ---
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "HD", "LOW",
    # --- Industrial ---
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "DE", "LMT", "MMM",
    # --- ETFs (broad market) ---
    "SPY", "QQQ", "DIA", "IWM", "VTI",
]


def download_ticker(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a single ticker and return a cleaned DataFrame."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for {ticker}")

    df = df.reset_index()

    # Normalise column names to lowercase
    rename = {}
    for col in df.columns:
        lower = col.lower() if isinstance(col, str) else str(col).lower()
        if lower == "date" or lower == "datetime":
            rename[col] = "date"
        elif lower == "open":
            rename[col] = "open"
        elif lower == "high":
            rename[col] = "high"
        elif lower == "low":
            rename[col] = "low"
        elif lower == "close":
            rename[col] = "close"
        elif lower == "volume":
            rename[col] = "volume"

    df = df.rename(columns=rename)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns after rename for {ticker}: {missing}")

    df = df[required].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_all(
    tickers: List[str],
    data_dir: Path = DATA_DIR,
    period: str = "max",
    interval: str = "1d",
) -> None:
    """Download data for every ticker and save as CSV."""
    data_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed: List[str] = []

    for ticker in tickers:
        ticker = ticker.upper()
        dest = data_dir / f"{ticker}.csv"
        try:
            df = download_ticker(ticker, period=period, interval=interval)
            df.to_csv(dest, index=False)
            logger.info(f"{ticker}: saved {len(df)} rows -> {dest}")
            success += 1
        except Exception as exc:
            logger.warning(f"{ticker}: FAILED – {exc}")
            failed.append(ticker)

    logger.info(f"\nDone. {success} succeeded, {len(failed)} failed.")
    if failed:
        logger.info(f"Failed tickers: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical stock data via yfinance")
    parser.add_argument(
        "tickers",
        nargs="*",
        default=None,
        help="Ticker symbols to download (default: built-in list of ~70 tickers)",
    )
    parser.add_argument(
        "--period",
        default="max",
        help="yfinance period string, e.g. max, 10y, 5y, 2y (default: max)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval, e.g. 1d, 1wk (default: 1d)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"Output directory for CSVs (default: {DATA_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = args.tickers if args.tickers else DEFAULT_TICKERS
    data_dir = Path(args.data_dir)

    logger.info(f"Fetching {len(tickers)} tickers (period={args.period}, interval={args.interval})")
    fetch_all(tickers, data_dir=data_dir, period=args.period, interval=args.interval)


if __name__ == "__main__":
    main()
