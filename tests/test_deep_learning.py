# -*- coding: utf-8 -*-
"""
Tests for deep learning module (LSTM stock prediction).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deep_learning import (
    LSTMClassifier,
    StockSequenceDataset,
    prepare_sequences,
    train_lstm,
    predict_lstm,
    LSTM_FEATURES,
    DEFAULT_LOOKBACK,
)


@pytest.fixture
def sample_feature_data():
    """Create sample feature data matching LSTM_FEATURES."""
    np.random.seed(42)
    n = 120
    data = {feat: np.random.randn(n) * 0.01 for feat in LSTM_FEATURES}
    X = pd.DataFrame(data)
    y = pd.Series((np.random.rand(n) > 0.5).astype(float))
    return X, y


class TestStockSequenceDataset:
    """Tests for sequence dataset creation."""

    def test_dataset_length(self):
        """Dataset length should equal rows - lookback."""
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.float32)
        ds = StockSequenceDataset(X, y, lookback=10)
        assert len(ds) == 40

    def test_dataset_shapes(self):
        """Each sample should have correct sequence shape."""
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.float32)
        ds = StockSequenceDataset(X, y, lookback=10)
        x_sample, y_sample = ds[0]
        assert x_sample.shape == (10, 5)
        assert y_sample.shape == ()

    def test_dataset_empty_for_short_input(self):
        """Dataset should be empty when input is shorter than lookback."""
        X = np.random.randn(5, 3).astype(np.float32)
        y = np.random.randint(0, 2, size=5).astype(np.float32)
        ds = StockSequenceDataset(X, y, lookback=10)
        assert len(ds) == 0


class TestLSTMClassifier:
    """Tests for the LSTM model architecture."""

    def test_forward_output_shape(self):
        """Forward pass should produce one logit per sample."""
        model = LSTMClassifier(input_size=5, hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(4, 10, 5)  # batch=4, seq=10, features=5
        out = model(x)
        assert out.shape == (4,)

    def test_output_is_finite(self):
        """Model output should not contain nan/inf."""
        model = LSTMClassifier(input_size=5, hidden_size=16, num_layers=2, dropout=0.1)
        model.eval()
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestPrepareSequences:
    """Tests for the prepare_sequences helper."""

    def test_returns_dataset(self, sample_feature_data):
        """Should return a StockSequenceDataset."""
        X, y = sample_feature_data
        ds = prepare_sequences(X, y, lookback=DEFAULT_LOOKBACK)
        assert isinstance(ds, StockSequenceDataset)
        assert len(ds) == len(X) - DEFAULT_LOOKBACK

    def test_normalized_values(self, sample_feature_data):
        """Sequences should contain normalized (finite) values."""
        X, y = sample_feature_data
        ds = prepare_sequences(X, y, lookback=DEFAULT_LOOKBACK)
        x_sample, _ = ds[0]
        assert torch.isfinite(x_sample).all()


class TestTrainAndPredict:
    """End-to-end training and prediction tests."""

    def test_train_runs(self, sample_feature_data):
        """Training should complete without errors."""
        X, y = sample_feature_data
        model, history = train_lstm(
            X, y, epochs=2, batch_size=16, lookback=10, hidden_size=8, num_layers=1
        )
        assert isinstance(model, LSTMClassifier)
        assert len(history["train_loss"]) == 2

    def test_predict_shapes(self, sample_feature_data):
        """Predictions should have correct length."""
        X, y = sample_feature_data
        lookback = 10
        model, _ = train_lstm(
            X, y, epochs=1, batch_size=16, lookback=lookback, hidden_size=8, num_layers=1
        )
        preds, probs = predict_lstm(model, X, lookback=lookback)
        assert len(preds) == len(X) - lookback
        assert len(probs) == len(X) - lookback

    def test_predict_values_range(self, sample_feature_data):
        """Predictions should be 0/1 and probabilities should be in [0, 1]."""
        X, y = sample_feature_data
        lookback = 10
        model, _ = train_lstm(
            X, y, epochs=2, batch_size=16, lookback=lookback, hidden_size=8, num_layers=1
        )
        preds, probs = predict_lstm(model, X, lookback=lookback)
        assert set(preds).issubset({0, 1})
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_train_with_validation(self, sample_feature_data):
        """Training with validation split should record val metrics."""
        X, y = sample_feature_data
        split = 80
        model, history = train_lstm(
            X.iloc[:split],
            y.iloc[:split],
            X_val=X.iloc[split:],
            y_val=y.iloc[split:],
            epochs=2,
            batch_size=16,
            lookback=10,
            hidden_size=8,
            num_layers=1,
        )
        assert history["val_loss"][0] is not None
        assert history["val_acc"][0] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
