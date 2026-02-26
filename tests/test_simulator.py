# -*- coding: utf-8 -*-
"""
Tests for gym-anytrading simulator integration.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    num_days = 200
    
    close = 100 + np.cumsum(np.random.randn(num_days) * 2)
    high = close + np.abs(np.random.randn(num_days)) * 2
    low = close - np.abs(np.random.randn(num_days)) * 2
    open_price = close + np.random.randn(num_days) * 0.5
    volume = np.abs(np.random.randn(num_days) * 1000000) + 500000
    
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")
    
    return pd.DataFrame({
        "date": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


class TestMLTradingEnv:
    """Test the ML Trading Environment."""
    
    def test_env_initialization(self, sample_ohlcv_data):
        """Environment should initialize without errors."""
        from run_simulator import MLTradingEnv
        
        window_size = 30
        frame_bound = (window_size, len(sample_ohlcv_data))
        
        env = MLTradingEnv(
            sample_ohlcv_data,
            window_size=window_size,
            frame_bound=frame_bound
        )
        
        assert env is not None
        env.close()
    
    def test_env_reset(self, sample_ohlcv_data):
        """Environment reset should return valid observation."""
        from run_simulator import MLTradingEnv
        
        window_size = 30
        frame_bound = (window_size, len(sample_ohlcv_data))
        
        env = MLTradingEnv(
            sample_ohlcv_data,
            window_size=window_size,
            frame_bound=frame_bound
        )
        
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(info, dict)
        env.close()
    
    def test_env_step(self, sample_ohlcv_data):
        """Environment step should work correctly."""
        from run_simulator import MLTradingEnv
        
        window_size = 30
        frame_bound = (window_size, len(sample_ohlcv_data))
        
        env = MLTradingEnv(
            sample_ohlcv_data,
            window_size=window_size,
            frame_bound=frame_bound
        )
        
        env.reset()
        
        # Take a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            assert obs is not None
            assert np.isreal(reward), "Reward should be a real number"
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)
            
            if done or truncated:
                break
        
        env.close()


class TestSimulatorFunctions:
    """Test simulator utility functions."""
    
    def test_load_model(self):
        """Should load models correctly."""
        from run_simulator import load_model
        
        # Test with hybrid model
        model, meta = load_model("hybrid_model")
        
        assert model is not None
        assert isinstance(meta, dict)
        assert "features" in meta
    
    def test_get_sample_data(self):
        """Should generate sample data."""
        from run_simulator import get_sample_data
        
        df = get_sample_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns
    
    def test_run_random_baseline(self):
        """Random baseline should complete without errors."""
        from run_simulator import run_random_baseline
        
        results = run_random_baseline(num_episodes=2, verbose=False)
        
        assert isinstance(results, dict)
        assert "episodes" in results
        assert "avg_profit" in results
        assert len(results["episodes"]) == 2


class TestModelComparison:
    """Test that ML models can run in the simulator."""
    
    def test_hybrid_model_runs(self):
        """Hybrid model should run in simulator."""
        from run_simulator import run_ml_trading_simulation
        
        results = run_ml_trading_simulation(
            model_name="hybrid_model",
            num_episodes=1,
            verbose=False
        )
        
        assert isinstance(results, dict)
        assert results["model"] == "hybrid_model"
        assert "avg_profit" in results
    
    @pytest.mark.skipif(
        not (ROOT / "models" / "enhanced_model.pkl").exists(),
        reason="Enhanced model not available"
    )
    def test_enhanced_model_runs(self):
        """Enhanced model should run in simulator."""
        from run_simulator import run_ml_trading_simulation
        
        results = run_ml_trading_simulation(
            model_name="enhanced_model",
            num_episodes=1,
            verbose=False
        )
        
        assert isinstance(results, dict)
        assert results["model"] == "enhanced_model"
        assert "avg_profit" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
