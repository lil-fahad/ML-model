# Hybrid Stock Predictor (Streamlit)

Local Streamlit UI to run ML models on **engineered technical features** for stock price direction prediction.

## Project Structure

```
ML-model/
├── app/                        # Streamlit application
│   └── streamlit_app.py
├── data/                       # Stock data CSV files
│   ├── AAPL.csv
│   └── TSLA.csv
├── models/                     # ML model files
│   ├── hybrid_model.pkl.b64       # Original model (10 features)
│   ├── enhanced_model.pkl.b64     # Enhanced model (20 features)
│   └── *_meta.json                # Model metadata
├── scripts/                    # Utility scripts
│   ├── restore_models.py       # Restores models from base64
│   ├── train_model.py          # Train enhanced model
│   └── inspect_model.py        # Inspects model structure
├── src/                        # Source modules
│   ├── features.py             # Basic feature engineering
│   ├── hybrid_features.py      # Hybrid model features
│   └── enhanced_features.py    # Enhanced features with technical indicators
├── tests/                      # Unit tests
│   └── test_enhanced_features.py
├── requirements.txt            # Python dependencies
└── README.md
```

## Quick Start

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/restore_models.py
streamlit run app/streamlit_app.py
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\restore_models.py
streamlit run app\streamlit_app.py
```

## Training a New Model

```bash
python scripts/train_model.py
```

This will:
1. Load all CSV files from `data/`
2. Build enhanced features (RSI, MACD, Bollinger Bands, etc.)
3. Train multiple models (LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM)
4. Optimize hyperparameters with Optuna
5. Save the best model to `models/enhanced_model.pkl.b64`

## Features

### Original Features (10)
- `ret_1`, `ret_3`, `ret_5`, `ret_10`, `ret_20` - Returns over different periods
- `vol_5`, `vol_10`, `vol_20` - Volatility measures
- `dd_20` - 20-day drawdown
- `range_pct` - Daily range as percentage

### Enhanced Features (20)
All original features plus:
- `rsi_14` - Relative Strength Index (normalized)
- `macd_signal` - MACD signal line crossover
- `bb_position` - Bollinger Bands position (-1 to 1)
- `momentum_10` - 10-day momentum
- `obv_change` - On-Balance Volume change
- `atr_14` - Average True Range (normalized)
- `ema_ratio` - EMA 12/26 ratio (trend indicator)
- `volume_sma_ratio` - Volume relative to 20-day SMA
- `stoch_k` - Normalized Stochastic %K oscillator (0-1)
- `adx_14` - Normalized Average Directional Index (trend strength)

## Model Performance

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| Original (RandomForest) | 52.03% | 52.17% | 10 features |
| Enhanced (XGBoost Optimized) | 47.17% | 53.48% | 20 features, Optuna tuned |

**Note:** Stock prediction is inherently difficult. Models are optimized for F1 score to balance precision and recall.

## Data

- Put your CSVs in `data/` as `TICKER.csv` (columns: Date, Open, High, Low, Close, Volume).
- Or use the built-in `yfinance` option in the UI (if network allows).

## Dependencies

- **Core:** streamlit, pandas, numpy, scikit-learn, joblib, yfinance
- **Enhanced ML:** xgboost, lightgbm, optuna
- **Testing:** pytest

## Notes

- Models are stored as base64-encoded text (`.pkl.b64`) for version control
- `scripts/restore_models.py` recreates binary `.pkl` files locally
- Run `pytest tests/` to verify feature engineering works correctly
