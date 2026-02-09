"""
Configuration module for ML Stock Predictor.
Centralized configuration for paths, parameters, and model settings.
"""
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field


# Project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SCRIPTS_DIR = ROOT_DIR / "scripts"
SRC_DIR = ROOT_DIR / "src"
TESTS_DIR = ROOT_DIR / "tests"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Original 10 features
    features_10: List[str] = field(default_factory=lambda: [
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
        "vol_5", "vol_10", "vol_20",
        "dd_20", "range_pct"
    ])
    
    # Enhanced 18 features
    enhanced_features: List[str] = field(default_factory=lambda: [
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
        "vol_5", "vol_10", "vol_20",
        "dd_20", "range_pct",
        "rsi_14", "macd_signal", "bb_position", "momentum_10",
        "obv_change", "atr_14", "ema_ratio", "volume_sma_ratio",
        "stoch_k", "adx_14"
    ])
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    atr_period: int = 14
    stoch_window: int = 14
    stoch_smooth: int = 3
    adx_window: int = 14


@dataclass
class ModelConfig:
    """Model training configuration."""
    default_model_path: str = "models/hybrid_model.pkl"
    enhanced_model_path: str = "models/enhanced_model.pkl"
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    n_cv_splits: int = 5
    
    # Model hyperparameters
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    })
    
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    })


@dataclass
class AppConfig:
    """Streamlit app configuration."""
    app_title: str = "Hybrid Stock Predictor (Professional)"
    default_ticker: str = "TSLA"
    default_period: str = "2y"
    default_interval: str = "1d"
    
    # Data validation
    min_rows_required: int = 50
    required_columns: List[str] = field(default_factory=lambda: [
        "close", "high", "low", "volume"
    ])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "ml_predictor.log"


# Global config instances
feature_config = FeatureConfig()
model_config = ModelConfig()
app_config = AppConfig()
logging_config = LoggingConfig()
