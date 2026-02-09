# -*- coding: utf-8 -*-
"""
Tests for utility functions.
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import (
    validate_dataframe,
    safe_divide,
    clean_numeric_data,
    ensure_datetime_index,
    DataValidationError,
)


class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_validate_dataframe_success(self):
        """Valid DataFrame should pass validation."""
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "volume": [1000, 1100, 1200]
        })
        
        # Should not raise
        validate_dataframe(df, ["close", "high", "low", "volume"], min_rows=3)
    
    def test_validate_dataframe_missing_columns(self):
        """Should raise error for missing columns."""
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "high": [101, 102, 103]
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df, ["close", "high", "low", "volume"])
    
    def test_validate_dataframe_insufficient_rows(self):
        """Should raise error for insufficient rows."""
        df = pd.DataFrame({
            "close": [100, 101]
        })
        
        with pytest.raises(ValueError, match="minimum 10 required"):
            validate_dataframe(df, ["close"], min_rows=10)
    
    def test_validate_dataframe_empty(self):
        """Should raise error for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty or None"):
            validate_dataframe(df, ["close"])
    
    def test_validate_dataframe_case_insensitive(self):
        """Column validation should be case-insensitive."""
        df = pd.DataFrame({
            "Close": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Volume": [1000, 1100, 1200]
        })
        
        # Should not raise (lowercase required, uppercase actual)
        validate_dataframe(df, ["close", "high", "low", "volume"], min_rows=3)


class TestSafeDivide:
    """Tests for safe division function."""
    
    def test_safe_divide_normal(self):
        """Normal division should work correctly."""
        result = safe_divide(10, 2)
        assert result == 5
    
    def test_safe_divide_by_zero_scalar(self):
        """Division by zero should return fill_value."""
        result = safe_divide(10, 0, fill_value=0.0)
        assert result == 0.0
    
    def test_safe_divide_series(self):
        """Should handle pandas Series correctly."""
        num = pd.Series([10, 20, 30])
        denom = pd.Series([2, 4, 0])
        result = safe_divide(num, denom, fill_value=0.0)
        
        expected = pd.Series([5.0, 5.0, 0.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_safe_divide_array(self):
        """Should handle numpy arrays correctly."""
        num = np.array([10, 20, 30])
        denom = np.array([2, 4, 0])
        result = safe_divide(num, denom, fill_value=0.0)
        
        expected = np.array([5.0, 5.0, 0.0])
        np.testing.assert_array_equal(result, expected)


class TestCleanNumericData:
    """Tests for numeric data cleaning."""
    
    def test_clean_numeric_data_inf(self):
        """Should replace inf values with NaN."""
        series = pd.Series([1, 2, np.inf, -np.inf, 3])
        result = clean_numeric_data(series, replace_inf=True)
        
        assert result.iloc[0] == 1
        assert result.iloc[1] == 2
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 3
    
    def test_clean_numeric_data_fill_na(self):
        """Should fill NaN values with specified value."""
        series = pd.Series([1, 2, np.nan, 4])
        result = clean_numeric_data(series, replace_inf=True, fill_na=0.0)
        
        expected = pd.Series([1.0, 2.0, 0.0, 4.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_clean_numeric_data_no_fill(self):
        """Should not fill NaN if fill_na is None."""
        series = pd.Series([1, 2, np.nan, 4])
        result = clean_numeric_data(series, replace_inf=True, fill_na=None)
        
        assert pd.isna(result.iloc[2])


class TestEnsureDatetimeIndex:
    """Tests for datetime index handling."""
    
    def test_ensure_datetime_index(self):
        """Should convert date column to datetime index."""
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value": [1, 2, 3]
        })
        
        result = ensure_datetime_index(df, date_column="date")
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3
    
    def test_ensure_datetime_index_already_datetime(self):
        """Should handle already datetime columns."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "value": [1, 2, 3]
        })
        
        result = ensure_datetime_index(df, date_column="date")
        
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_ensure_datetime_index_sorts(self):
        """Should sort by date."""
        df = pd.DataFrame({
            "date": ["2023-01-03", "2023-01-01", "2023-01-02"],
            "value": [3, 1, 2]
        })
        
        result = ensure_datetime_index(df, date_column="date")
        
        assert result["value"].iloc[0] == 1
        assert result["value"].iloc[1] == 2
        assert result["value"].iloc[2] == 3
    
    def test_ensure_datetime_index_case_insensitive(self):
        """Should handle different date column cases."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "value": [1, 2]
        })
        
        result = ensure_datetime_index(df, date_column="date")
        
        assert isinstance(result.index, pd.DatetimeIndex)
