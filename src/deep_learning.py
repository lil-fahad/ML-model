# -*- coding: utf-8 -*-
"""
Deep learning module for stock prediction using LSTM.
Provides sequence-based prediction as a complement to tree-based models.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Default features for LSTM input
LSTM_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20",
    "dd_20", "range_pct",
    "rsi_14", "macd_signal", "bb_position",
    "momentum_10", "obv_change", "atr_14",
    "ema_ratio", "volume_sma_ratio",
]

DEFAULT_LOOKBACK = 20


class StockSequenceDataset(Dataset):
    """PyTorch dataset that creates sliding-window sequences from feature data."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int = DEFAULT_LOOKBACK):
        self.lookback = lookback
        self.X_seq = []
        self.y_seq = []
        for i in range(lookback, len(X)):
            self.X_seq.append(X[i - lookback : i])
            self.y_seq.append(y[i])
        self.X_seq = np.array(self.X_seq, dtype=np.float32)
        self.y_seq = np.array(self.y_seq, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y_seq)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X_seq[idx], dtype=torch.float32),
            torch.tensor(self.y_seq[idx], dtype=torch.float32),
        )


class LSTMClassifier(nn.Module):
    """LSTM-based binary classifier for stock direction prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits


def prepare_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int = DEFAULT_LOOKBACK,
) -> StockSequenceDataset:
    """Convert feature DataFrame and target Series into a sequence dataset."""
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    # Normalize features per-column (z-score)
    mean = np.nanmean(X_arr, axis=0)
    std = np.nanstd(X_arr, axis=0)
    std[std == 0] = 1.0
    X_arr = (X_arr - mean) / std

    # Replace any remaining nan/inf with 0
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    return StockSequenceDataset(X_arr, y_arr, lookback=lookback)


def train_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 64,
    lookback: int = DEFAULT_LOOKBACK,
    device: Optional[str] = None,
) -> Tuple[LSTMClassifier, dict]:
    """
    Train an LSTM model for stock direction classification.

    Returns:
        Trained model and a dict of training metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = X_train.shape[1]
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    train_ds = prepare_sequences(X_train, y_train, lookback)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # preserve temporal order

    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = prepare_sequences(X_val, y_val, lookback)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        epoch_loss = running_loss / max(len(train_ds), 1)
        history["train_loss"].append(epoch_loss)

        # Validation
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * len(y_batch)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == y_batch).sum().item()
                    val_total += len(y_batch)
            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch + 1}/{epochs} - train_loss: {epoch_loss:.4f}"
            if val_loader is not None:
                msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            logger.info(msg)

    return model, history


def predict_lstm(
    model: LSTMClassifier,
    X: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from a trained LSTM model.

    Returns:
        (predictions, probabilities) – both as numpy arrays.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_arr = X.values.astype(np.float32)
    mean = np.nanmean(X_arr, axis=0)
    std = np.nanstd(X_arr, axis=0)
    std[std == 0] = 1.0
    X_arr = (X_arr - mean) / std
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Build sequences
    sequences = []
    for i in range(lookback, len(X_arr)):
        sequences.append(X_arr[i - lookback : i])
    if not sequences:
        return np.array([]), np.array([])

    X_seq = torch.tensor(np.array(sequences, dtype=np.float32), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_seq)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    return preds, probs
