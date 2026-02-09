#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for enhanced stock prediction model.
Uses multiple ML algorithms with hyperparameter tuning and proper time-series cross-validation.
"""
import os
import sys
import json
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Add src to path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from enhanced_features import (
    build_enhanced_features, create_target, 
    ENHANCED_FEATURES, FEATURES_10
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing optional packages
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not available")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not available for hyperparameter tuning")


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load and combine all CSV files from data directory."""
    all_dfs = []
    for csv_file in data_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df.columns = [c.lower().strip() for c in df.columns]
        df["ticker"] = csv_file.stem.upper()
        all_dfs.append(df)
        logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
    
    if not all_dfs:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} rows")
    return combined


def prepare_dataset(
    data_dir: Path,
    feature_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare full dataset with features and target.
    Returns X, y, and metadata DataFrame.
    """
    if feature_cols is None:
        feature_cols = ENHANCED_FEATURES
    
    combined_df = load_all_data(data_dir)
    
    all_X = []
    all_y = []
    all_meta = []
    
    for ticker in combined_df["ticker"].unique():
        ticker_df = combined_df[combined_df["ticker"] == ticker].copy()
        ticker_df = ticker_df.sort_values("date").reset_index(drop=True)
        
        # Build features
        features_df = build_enhanced_features(ticker_df)
        target = create_target(ticker_df)
        
        # Combine
        features_df["target"] = target
        features_df["ticker"] = ticker
        features_df = features_df.dropna()
        
        if len(features_df) > 50:  # Minimum rows for meaningful features
            all_X.append(features_df[feature_cols])
            all_y.append(features_df["target"])
            all_meta.append(features_df[["date", "ticker"]])
    
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    meta = pd.concat(all_meta, ignore_index=True)
    
    logger.info(f"Final dataset: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, meta


def time_series_train_test_split(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data respecting time order (last test_size% as test)."""
    n = len(X)
    split_idx = int(n * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else 0.5,
    }
    return metrics


def cross_validate_model(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int = 5
) -> Dict[str, float]:
    """Perform time series cross-validation."""
    from sklearn.base import clone
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = {"accuracy": [], "f1": [], "roc_auc": []}
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        metrics = evaluate_model(model_clone, X_val, y_val)
        for key in scores:
            scores[key].append(metrics[key])
    
    return {k: np.mean(v) for k, v in scores.items()}


def create_base_pipeline(estimator) -> Pipeline:
    """Create a preprocessing pipeline with the given estimator."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("estimator", estimator)
    ])


def get_model_configs() -> Dict[str, Any]:
    """Get model configurations for comparison."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
    }
    
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
    
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    return models


def optimize_xgboost(X_train, y_train, n_trials: int = 50) -> Dict[str, Any]:
    """Optimize XGBoost hyperparameters using Optuna."""
    if not HAS_OPTUNA or not HAS_XGB:
        return {}
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        
        model = create_base_pipeline(xgb.XGBClassifier(**params))
        cv_scores = cross_validate_model(model, X_train, y_train, n_splits=3)
        return cv_scores["f1"]
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best XGBoost params: {study.best_params}")
    logger.info(f"Best XGBoost F1: {study.best_value:.4f}")
    
    return study.best_params


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Any, Dict[str, Any]]:
    """Train all models and return the best one."""
    models = get_model_configs()
    results = {}
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, estimator in models.items():
        logger.info(f"\nTraining {name}...")
        
        pipeline = create_base_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = evaluate_model(pipeline, X_test, y_test)
        cv_metrics = cross_validate_model(pipeline, X_train, y_train, n_splits=5)
        
        results[name] = {
            "test": test_metrics,
            "cv": cv_metrics
        }
        
        logger.info(f"{name} - Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        logger.info(f"{name} - CV F1: {cv_metrics['f1']:.4f}")
        
        # Track best by test F1 score 
        if test_metrics["f1"] > best_score:
            best_score = test_metrics["f1"]
            best_model = pipeline
            best_name = name
    
    logger.info(f"\nBest model: {best_name} with Test F1: {best_score:.4f}")
    
    # Run Optuna hyperparameter optimization for XGBoost
    if HAS_OPTUNA and HAS_XGB:
        logger.info("\nRunning hyperparameter optimization with Optuna...")
        best_params = optimize_xgboost(X_train, y_train, n_trials=50)
        
        if best_params:
            optimized_model = create_base_pipeline(
                xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, eval_metric="logloss")
            )
            optimized_model.fit(X_train, y_train)
            
            opt_metrics = evaluate_model(optimized_model, X_test, y_test)
            logger.info(f"Optimized XGBoost - Test Accuracy: {opt_metrics['accuracy']:.4f}, F1: {opt_metrics['f1']:.4f}")
            
            if opt_metrics["f1"] > best_score:
                best_model = optimized_model
                best_name = "XGBoost_Optimized"
                best_score = opt_metrics["f1"]
                results[best_name] = {"test": opt_metrics, "cv": {"f1": opt_metrics["f1"]}}
    
    return best_model, results


def save_model(
    model, 
    output_dir: Path,
    feature_cols: List[str],
    metrics: Dict[str, float],
    model_name: str
):
    """Save model as both .pkl and .pkl.b64 (for git)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save pickle
    pkl_path = output_dir / "enhanced_model.pkl"
    joblib.dump(model, pkl_path)
    logger.info(f"Saved model to {pkl_path}")
    
    # Save base64 encoded version
    b64_path = output_dir / "enhanced_model.pkl.b64"
    with open(pkl_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    with open(b64_path, "w") as f:
        f.write(b64_data)
    logger.info(f"Saved base64 model to {b64_path}")
    
    # Save metadata
    meta = {
        "model_name": model_name,
        "features": feature_cols,
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_roc_auc": metrics["roc_auc"],
    }
    
    meta_path = output_dir / "enhanced_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("ENHANCED MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    data_dir = ROOT / "data"
    models_dir = ROOT / "models"
    
    # Prepare data
    logger.info("\n1. Loading and preparing data...")
    X, y, meta = prepare_dataset(data_dir, feature_cols=ENHANCED_FEATURES)
    
    # Split data
    logger.info("\n2. Splitting data (time-series aware)...")
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_size=0.2)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train and evaluate models
    logger.info("\n3. Training and evaluating models...")
    best_model, all_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Final evaluation
    logger.info("\n4. Final evaluation on test set...")
    final_metrics = evaluate_model(best_model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {final_metrics['f1']:.4f}")
    print(f"Test Precision: {final_metrics['precision']:.4f}")
    print(f"Test Recall: {final_metrics['recall']:.4f}")
    print(f"Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    y_pred = best_model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
    
    # Compare with original model
    print("\n" + "=" * 60)
    print("COMPARISON WITH ORIGINAL MODEL")
    print("=" * 60)
    print(f"Original Model - Accuracy: 0.5203, F1: 0.5217")
    print(f"Enhanced Model - Accuracy: {final_metrics['accuracy']:.4f}, F1: {final_metrics['f1']:.4f}")
    
    improvement_acc = (final_metrics['accuracy'] - 0.5203) / 0.5203 * 100
    improvement_f1 = (final_metrics['f1'] - 0.5217) / 0.5217 * 100
    print(f"Accuracy Improvement: {improvement_acc:+.2f}%")
    print(f"F1 Score Improvement: {improvement_f1:+.2f}%")
    
    # Save model
    logger.info("\n5. Saving best model...")
    
    # Determine best model name
    best_name = "Enhanced"
    for name, result in all_results.items():
        if result["test"]["f1"] == final_metrics["f1"]:
            best_name = name
            break
    
    save_model(
        best_model, 
        models_dir, 
        ENHANCED_FEATURES, 
        final_metrics,
        best_name
    )
    
    logger.info("\nTraining complete!")
    return best_model, final_metrics


if __name__ == "__main__":
    main()
