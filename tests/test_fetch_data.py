# -*- coding: utf-8 -*-
"""
Tests for the fetch_data script.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_data import download_ticker, fetch_all, DEFAULT_TICKERS


class TestDefaultTickers:
    """Verify the default ticker list is well-formed."""

    def test_default_tickers_not_empty(self):
        assert len(DEFAULT_TICKERS) > 0

    def test_default_tickers_are_uppercase(self):
        for t in DEFAULT_TICKERS:
            assert t == t.upper(), f"Ticker {t} should be uppercase"

    def test_default_tickers_unique(self):
        assert len(DEFAULT_TICKERS) == len(set(DEFAULT_TICKERS)), "Duplicate tickers found"


class TestDownloadTicker:
    """Test download_ticker with a mocked yfinance response."""

    @staticmethod
    def _fake_yf_dataframe(rows: int = 100) -> pd.DataFrame:
        """Return a DataFrame that mimics yfinance output."""
        dates = pd.date_range("2020-01-01", periods=rows, freq="D")
        close = 100 + np.cumsum(np.random.randn(rows))
        return pd.DataFrame({
            "Date": dates,
            "Open": close + np.random.randn(rows) * 0.5,
            "High": close + np.abs(np.random.randn(rows)),
            "Low": close - np.abs(np.random.randn(rows)),
            "Close": close,
            "Adj Close": close,
            "Volume": np.abs(np.random.randn(rows) * 1e6).astype(int),
        }).set_index("Date")  # yfinance returns DatetimeIndex

    @patch("fetch_data.yf")
    def test_download_returns_expected_columns(self, mock_yf):
        mock_yf.download.return_value = self._fake_yf_dataframe()
        df = download_ticker("TEST")
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]

    @patch("fetch_data.yf")
    def test_download_sorted_by_date(self, mock_yf):
        mock_yf.download.return_value = self._fake_yf_dataframe()
        df = download_ticker("TEST")
        assert df["date"].is_monotonic_increasing

    @patch("fetch_data.yf")
    def test_download_raises_on_empty(self, mock_yf):
        mock_yf.download.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="No data returned"):
            download_ticker("INVALID")


class TestFetchAll:
    """Test the fetch_all orchestrator."""

    @patch("fetch_data.download_ticker")
    def test_creates_csv_files(self, mock_dl, tmp_path):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        mock_dl.return_value = pd.DataFrame({
            "date": dates,
            "open": [1] * 5,
            "high": [2] * 5,
            "low": [0.5] * 5,
            "close": [1.5] * 5,
            "volume": [1000] * 5,
        })
        fetch_all(["AA", "BB"], data_dir=tmp_path)
        assert (tmp_path / "AA.csv").exists()
        assert (tmp_path / "BB.csv").exists()

    @patch("fetch_data.download_ticker", side_effect=RuntimeError("network error"))
    def test_handles_failures_gracefully(self, mock_dl, tmp_path):
        # Should not raise even when all tickers fail
        fetch_all(["FAIL1", "FAIL2"], data_dir=tmp_path)
        assert not (tmp_path / "FAIL1.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
