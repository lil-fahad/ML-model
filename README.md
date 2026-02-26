# Hybrid Stock Predictor (Streamlit)

> 🇸🇦 [اقرأ هذا بالعربية](README.ar.md)

Local Streamlit UI to run ML models on **engineered technical features** for stock price direction prediction.

## Project Structure

```
ML-model/
├── app/                        # Streamlit application
│   └── streamlit_app.py
├── data/                       # Stock data CSV files (populated by download_data.py)
├── models/                     # ML model files
│   ├── hybrid_model.pkl.b64       # Original model (10 features)
│   ├── enhanced_model.pkl.b64     # Enhanced model (18 features)
│   └── *_meta.json                # Model metadata
├── scripts/                    # Utility scripts
│   ├── download_data.py        # Download Kaggle stock market dataset
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

## Getting Data

### Option A – Kaggle MCP Server (recommended)

The project ships a `.mcp.json` file that points any MCP-enabled client
(Claude Desktop, Cursor, VS Code with the MCP extension, etc.) at Kaggle's
remote MCP server:

```
https://www.kaggle.com/mcp
```

#### 1. Generate your auth token

Your Kaggle API key lives at [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account) under **API → Create New Token**.
Encode it as a Basic-auth token:

```bash
# Linux / macOS
export KAGGLE_BASIC_AUTH_TOKEN=$(echo -n "YOUR_KAGGLE_USERNAME:YOUR_KAGGLE_KEY" | base64)
```

```powershell
# Windows PowerShell
$env:KAGGLE_BASIC_AUTH_TOKEN = [Convert]::ToBase64String(
    [Text.Encoding]::ASCII.GetBytes("YOUR_KAGGLE_USERNAME:YOUR_KAGGLE_KEY"))
```

The `.mcp.json` file already references `${KAGGLE_BASIC_AUTH_TOKEN}`, so MCP
clients will pick it up automatically once the env var is set.

#### 2. Download the dataset via the MCP client

Connect your MCP client to the server and run:

```
Download paultimothymooney/stock-market-data
```

The Kaggle MCP server will stream the dataset files into your session.

### Option B – Python download script

```bash
# Requires KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json)
python scripts/download_data.py
```

This fetches the same dataset via `kagglehub` and copies the CSV files into
`data/`.

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

### Enhanced Features (18)
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
| Enhanced (XGBoost Optimized) | 49.19% | 53.60% | 18 features, Optuna tuned |

**Note:** Stock prediction is inherently difficult. Models are optimized for F1 score to balance precision and recall.

## Data

- Use the **Kaggle MCP server** (see "Getting Data" above) or run
  `python scripts/download_data.py` to populate `data/` with the
  `paultimothymooney/stock-market-data` dataset.
- You can also add your own CSVs to `data/` as `TICKER.csv`
  (columns: `date`, `open`, `high`, `low`, `close`, `volume`).
- The `yfinance` option in the Streamlit UI also works when a network
  connection is available.


## Dependencies

- **Core:** streamlit, pandas, numpy, scikit-learn, joblib, yfinance
- **Enhanced ML:** xgboost, lightgbm, optuna
- **Testing:** pytest

## Notes

- Models are stored as base64-encoded text (`.pkl.b64`) for version control
- `scripts/restore_models.py` recreates binary `.pkl` files locally
- Run `pytest tests/` to verify feature engineering works correctly
