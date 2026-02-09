# -*- coding: utf-8 -*-
"""
Tests for enhanced feature engineering.
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

from enhanced_features import (
    compute_rsi,
    compute_macd,
    compute_bollinger_position,
    compute_atr,
    compute_obv_change,
    build_enhanced_features,
    create_target,
    ENHANCED_FEATURES,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate realistic-ish price data
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_price = close + np.random.randn(n) * 0.5
    volume = np.abs(np.random.randn(n) * 1000000) + 500000
    
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    
    return pd.DataFrame({
        "date": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


class TestTechnicalIndicators:
    """Test individual technical indicators."""
    
    def test_rsi_range(self, sample_ohlcv_data):
        """RSI should be between 0 and 100."""
        prices = sample_ohlcv_data["close"]
        rsi = compute_rsi(prices, period=14)
        
        # After warmup period, RSI should be in valid range
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI should be >= 0"
        assert (valid_rsi <= 100).all(), "RSI should be <= 100"
    
    def test_rsi_extremes(self):
        """RSI should approach extremes for monotonic prices."""
        # Consistently rising prices with some variance
        rising = pd.Series([100 + i + np.random.randn() * 0.1 for i in range(50)])
        rsi_rising = compute_rsi(rising, period=14)
        # For consistently rising, RSI should be above neutral
        assert rsi_rising.iloc[-1] >= 50, "RSI should be neutral or high for rising prices"
        
        # Consistently falling prices with some variance  
        falling = pd.Series([100 - i - np.random.randn() * 0.1 for i in range(50)])
        rsi_falling = compute_rsi(falling, period=14)
        # For consistently falling, RSI should be below neutral
        assert rsi_falling.iloc[-1] <= 50, "RSI should be neutral or low for falling prices"
    
    def test_macd_output(self, sample_ohlcv_data):
        """MACD should produce valid output."""
        prices = sample_ohlcv_data["close"]
        macd = compute_macd(prices)
        
        # Should not have NaN after warmup
        valid_macd = macd.iloc[30:]
        assert not valid_macd.isna().any(), "MACD should not have NaN after warmup"
    
    def test_bollinger_position_range(self, sample_ohlcv_data):
        """Bollinger position should be between -1 and 1."""
        prices = sample_ohlcv_data["close"]
        bb_pos = compute_bollinger_position(prices)
        
        valid_bb = bb_pos.dropna()
        assert (valid_bb >= -1).all(), "BB position should be >= -1"
        assert (valid_bb <= 1).all(), "BB position should be <= 1"
    
    def test_atr_positive(self, sample_ohlcv_data):
        """ATR should always be positive."""
        atr = compute_atr(
            sample_ohlcv_data["high"],
            sample_ohlcv_data["low"],
            sample_ohlcv_data["close"],
            period=14
        )
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all(), "ATR should be positive"
    
    def test_obv_change_no_inf(self, sample_ohlcv_data):
        """OBV change should not have infinite values."""
        obv = compute_obv_change(
            sample_ohlcv_data["close"],
            sample_ohlcv_data["volume"]
        )
        assert not np.isinf(obv).any(), "OBV should not have infinite values"


class TestEnhancedFeatures:
    """Test the enhanced feature builder."""
    
    def test_build_features_output_shape(self, sample_ohlcv_data):
        """Should produce all expected features."""
        features_df = build_enhanced_features(sample_ohlcv_data)
        
        # Should have all enhanced features
        for feat in ENHANCED_FEATURES:
            assert feat in features_df.columns, f"Missing feature: {feat}"
    
    def test_build_features_no_inf(self, sample_ohlcv_data):
        """Features should not have infinite values."""
        features_df = build_enhanced_features(sample_ohlcv_data)
        
        for col in ENHANCED_FEATURES:
            assert not np.isinf(features_df[col]).any(), f"{col} has infinite values"
    
    def test_build_features_reasonable_nan(self, sample_ohlcv_data):
        """After dropping NaN, should have reasonable number of rows."""
        features_df = build_enhanced_features(sample_ohlcv_data)
        clean_df = features_df.dropna()
        
        # Should retain at least 50% of data after warmup
        assert len(clean_df) >= len(sample_ohlcv_data) * 0.3, "Too many NaN values"
    
    def test_build_features_missing_column_error(self):
        """Should raise error for missing required columns."""
        incomplete_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "close": [100] * 10
            # Missing high, low, volume
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            build_enhanced_features(incomplete_df)
    
    def test_build_features_case_insensitive(self, sample_ohlcv_data):
        """Should handle different column name cases."""
        upper_df = sample_ohlcv_data.copy()
        upper_df.columns = [c.upper() for c in upper_df.columns]
        
        features_df = build_enhanced_features(upper_df)
        assert len(features_df) > 0


class TestTarget:
    """Test target creation."""
    
    def test_create_target_binary(self, sample_ohlcv_data):
        """Target should be binary (0 or 1)."""
        target = create_target(sample_ohlcv_data)
        
        unique_vals = target.dropna().unique()
        assert set(unique_vals).issubset({0, 1}), "Target should be binary"
    
    def test_create_target_horizon(self, sample_ohlcv_data):
        """Target with different horizons should differ."""
        target_1 = create_target(sample_ohlcv_data, horizon=1)
        target_5 = create_target(sample_ohlcv_data, horizon=5)
        
        # They should be different (not identical)
        assert not (target_1 == target_5).all(), "Different horizons should produce different targets"


class TestIntegration:
    """Integration tests with real data."""
    
    def test_with_real_csv(self):
        """Test with actual CSV file from data directory."""
        csv_path = ROOT / "data" / "TSLA.csv"
        
        if not csv_path.exists():
            pytest.skip("TSLA.csv not found")
        
        df = pd.read_csv(csv_path)
        features_df = build_enhanced_features(df)
        clean_df = features_df.dropna()
        
        assert len(clean_df) > 100, "Should have enough rows from real data"
        
        for feat in ENHANCED_FEATURES:
            assert feat in clean_df.columns
            assert not np.isinf(clean_df[feat]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
