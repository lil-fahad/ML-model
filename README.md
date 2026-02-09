# Hybrid Stock Predictor (Professional Edition)

Professional-grade local Streamlit UI to run ML models on **engineered technical features** for stock price direction prediction.

## ✨ Key Features

- **Professional Code Quality**: Comprehensive error handling, logging, and validation
- **Type Safety**: Full type hints and data validation
- **Configuration Management**: Centralized configuration for easy customization
- **Robust Error Handling**: Custom exceptions and detailed error messages
- **Enhanced UX**: Improved Streamlit interface with better feedback
- **Modular Design**: Well-organized codebase with utility functions
- **Comprehensive Testing**: Full test coverage with pytest
- **Documentation**: Detailed docstrings for all functions

## Project Structure

```
ML-model/
├── app/                        # Streamlit application
│   └── streamlit_app.py       # Professional UI with enhanced error handling
├── data/                       # Stock data CSV files
│   ├── AAPL.csv
│   └── TSLA.csv
├── models/                     # ML model files
│   ├── hybrid_model.pkl.b64       # Original model (10 features)
│   ├── enhanced_model.pkl.b64     # Enhanced model (18 features)
│   └── *_meta.json                # Model metadata
├── scripts/                    # Utility scripts
│   ├── restore_models.py       # Restores models from base64
│   ├── train_model.py          # Train enhanced model
│   └── inspect_model.py        # Inspects model structure
├── src/                        # Source modules
│   ├── features.py             # Basic feature engineering
│   ├── hybrid_features.py      # Hybrid model features
│   ├── enhanced_features.py    # Enhanced features with technical indicators
│   └── utils.py                # Utility functions (validation, error handling)
├── tests/                      # Unit tests
│   └── test_enhanced_features.py
├── config.py                   # Configuration management
├── logging.conf                # Logging configuration
├── Makefile                    # Common development tasks
├── .editorconfig              # Editor configuration
├── requirements.txt            # Python dependencies
└── README.md
```

## Quick Start

### Using Make (Recommended)

```bash
# Install dependencies
make install

# Restore models
make restore-models

# Run tests
make test

# Start the app
make run
```

### Manual Setup

#### Linux / macOS

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

- Put your CSVs in `data/` as `TICKER.csv` (columns: Date, Open, High, Low, Close, Volume).
- Or use the built-in `yfinance` option in the UI (if network allows).

## Dependencies

- **Core:** streamlit, pandas, numpy, scikit-learn, joblib, yfinance
- **Enhanced ML:** xgboost, lightgbm, optuna
- **Technical Analysis:** ta (python technical analysis library)
- **Testing:** pytest

## Professional Code Features

### Error Handling
- Custom exception classes: `DataValidationError`, `ModelLoadError`, `FeatureEngineeringError`
- Comprehensive try-catch blocks with detailed error messages
- Graceful degradation and user-friendly error reporting

### Configuration Management
- Centralized configuration in `config.py`
- Feature parameters, model settings, and app configuration
- Easy customization without code changes

### Utilities
- Data validation functions
- Safe mathematical operations (division by zero handling)
- Consistent logging throughout the application
- Model feature validation

### Code Quality
- Full type hints for better IDE support
- Comprehensive docstrings following Google style
- Modular design with separation of concerns
- EditorConfig for consistent formatting

## Development

### Running Tests
```bash
# Run all tests
make test
# or
python -m pytest tests/ -v

# Run with coverage (if installed)
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Lint code
make lint

# Format code (if black/isort installed)
make format

# Clean temporary files
make clean
```

## Notes

- Models are stored as base64-encoded text (`.pkl.b64`) for version control
- `scripts/restore_models.py` recreates binary `.pkl` files locally
- Run `pytest tests/` to verify feature engineering works correctly
