#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for LSTM-based stock prediction model.
Uses sequence data from enhanced features for deep learning classification.
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add src to path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from enhanced_features import build_enhanced_features, create_target, ENHANCED_FEATURES
from deep_learning import (
    LSTM_FEATURES,
    DEFAULT_LOOKBACK,
    LSTMClassifier,
    train_lstm,
    predict_lstm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load and combine all CSV files from data directory."""
    all_dfs = []
    for csv_file in sorted(data_dir.glob("*.csv")):
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


def prepare_dataset(data_dir: Path):
    """Prepare feature/target arrays per ticker, then concatenate."""
    combined_df = load_all_data(data_dir)
    all_X, all_y = [], []

    for ticker in sorted(combined_df["ticker"].unique()):
        ticker_df = combined_df[combined_df["ticker"] == ticker].copy()
        ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

        features_df = build_enhanced_features(ticker_df)
        target = create_target(ticker_df)

        features_df["target"] = target
        features_df = features_df.dropna()

        if len(features_df) > 50:
            all_X.append(features_df[LSTM_FEATURES])
            all_y.append(features_df["target"])

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    logger.info(f"Final dataset: X={X.shape}, y distribution={y.value_counts().to_dict()}")
    return X, y


def evaluate(model: LSTMClassifier, X: pd.DataFrame, y: pd.Series, lookback: int) -> Dict[str, float]:
    """Evaluate LSTM model and return metric dict."""
    preds, probs = predict_lstm(model, X, lookback=lookback)
    # predictions are offset by lookback
    y_true = y.values[lookback:]
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def main():
    logger.info("=" * 60)
    logger.info("LSTM DEEP LEARNING TRAINING PIPELINE")
    logger.info("=" * 60)

    data_dir = ROOT / "data"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data
    logger.info("\n1. Loading data...")
    X, y = prepare_dataset(data_dir)

    # 2. Time-series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Train LSTM
    logger.info("\n2. Training LSTM model...")
    lookback = DEFAULT_LOOKBACK
    model, history = train_lstm(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        lr=1e-3,
        epochs=30,
        batch_size=64,
        lookback=lookback,
    )

    # 4. Evaluate
    logger.info("\n3. Evaluating...")
    metrics = evaluate(model, X_test, y_test, lookback)

    print("\n" + "=" * 60)
    print("LSTM RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5. Save
    model_path = models_dir / "lstm_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": X_train.shape[1],
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "lookback": lookback,
            "features": LSTM_FEATURES,
        },
        model_path,
    )
    logger.info(f"Saved LSTM model to {model_path}")

    # Save metadata
    meta = {
        "model_name": "LSTM",
        "features": LSTM_FEATURES,
        "lookback": lookback,
        "trained_at": datetime.now().isoformat(),
        **metrics,
    }
    meta_path = models_dir / "lstm_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    logger.info("\nTraining complete!")
    return model, metrics


if __name__ == "__main__":
    main()
