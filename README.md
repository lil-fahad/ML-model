# Hybrid Stock Predictor (Streamlit)

Local Streamlit UI to run `hybrid_model` on **10 engineered features**.

## Project Structure

```
ML-model/
├── app/                    # Streamlit application
│   └── streamlit_app.py
├── data/                   # Stock data CSV files
│   ├── AAPL.csv
│   └── TSLA.csv
├── models/                 # ML model files
│   ├── hybrid_model.pkl.b64   # Base64-encoded model (stored in repo)
│   ├── hybrid_model.pkl       # Restored model (generated locally)
│   └── hybrid_model_meta.json # Model metadata
├── scripts/                # Utility scripts
│   ├── restore_models.py   # Restores model from base64
│   └── inspect_model.py    # Inspects model structure
├── src/                    # Source modules
│   ├── features.py         # Feature engineering
│   └── hybrid_features.py  # Hybrid model features
├── requirements.txt        # Python dependencies
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

## Data

- Put your CSVs in `data/` as `TICKER.csv` (columns: Date, Open, High, Low, Close, Volume).
- Or use the built-in `yfinance` option in the UI (if network allows).

## Notes

- `models/hybrid_model.pkl.b64` is stored in repo as text.
- `scripts/restore_models.py` recreates `models/hybrid_model.pkl` locally.
