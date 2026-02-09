"""
Utility functions for ML Stock Predictor.
Provides common functions for data validation, logging setup, and error handling.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, List, Union
import pandas as pd
import numpy as np


def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1,
    name: str = "DataFrame"
) -> None:
    """
    Validate that a DataFrame meets requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
        name: Name of the DataFrame for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError(f"{name} is empty or None")
    
    if len(df) < min_rows:
        raise ValueError(f"{name} has {len(df)} rows, minimum {min_rows} required")
    
    # Normalize column names to lowercase for comparison
    df_columns_lower = [col.lower() for col in df.columns]
    required_lower = [col.lower() for col in required_columns]
    
    missing_columns = [col for col in required_lower if col not in df_columns_lower]
    if missing_columns:
        raise ValueError(
            f"{name} missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def safe_divide(
    numerator: Union[pd.Series, np.ndarray, float],
    denominator: Union[pd.Series, np.ndarray, float],
    fill_value: float = 0.0
) -> Union[pd.Series, np.ndarray, float]:
    """
    Safely divide two values, handling division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result of division with safe handling of zero denominators
    """
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(fill_value)
    elif isinstance(denominator, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result = np.where(np.isfinite(result), result, fill_value)
        return result
    else:
        return numerator / denominator if denominator != 0 else fill_value


def clean_numeric_data(
    series: pd.Series,
    replace_inf: bool = True,
    fill_na: Optional[float] = None
) -> pd.Series:
    """
    Clean numeric data by handling inf and NaN values.
    
    Args:
        series: Pandas Series to clean
        replace_inf: Whether to replace inf values with NaN
        fill_na: Optional value to fill NaN values with
        
    Returns:
        Cleaned Series
    """
    if replace_inf:
        series = series.replace([np.inf, -np.inf], np.nan)
    
    if fill_na is not None:
        series = series.fillna(fill_na)
    
    return series


def ensure_datetime_index(
    df: pd.DataFrame,
    date_column: str = "date"
) -> pd.DataFrame:
    """
    Ensure DataFrame has a proper datetime index.
    
    Args:
        df: DataFrame to process
        date_column: Name of the date column
        
    Returns:
        DataFrame with datetime index
    """
    df = df.copy()
    
    # Normalize column names
    col_lower = {col.lower(): col for col in df.columns}
    date_col = col_lower.get(date_column.lower())
    
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(date_col)
    
    return df


def validate_model_features(
    model,
    dataframe: pd.DataFrame,
    feature_names: Optional[List[str]] = None
) -> List[str]:
    """
    Validate that DataFrame has all features required by the model.
    
    Args:
        model: Trained model (scikit-learn compatible)
        dataframe: DataFrame to validate
        feature_names: Optional list of expected feature names
        
    Returns:
        List of feature names in correct order
        
    Raises:
        ValueError: If required features are missing
    """
    # Try to get feature names from model
    if feature_names is None:
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None and hasattr(model, "named_steps"):
            # Try pipeline
            est = list(model.named_steps.values())[-1]
            feature_names = getattr(est, "feature_names_in_", None)
    
    if feature_names is None:
        raise ValueError("Could not determine required features from model")
    
    # Normalize column names
    df_cols_lower = {col.lower(): col for col in dataframe.columns}
    missing = []
    
    for feat in feature_names:
        if feat.lower() not in df_cols_lower:
            missing.append(feat)
    
    if missing:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"Available: {list(dataframe.columns)}"
        )
    
    return list(feature_names)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors."""
    pass
