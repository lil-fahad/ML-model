# Hybrid Stock Predictor (Streamlit)

Local Streamlit UI to run `hybrid_model` on **10 engineered features**.

## Quick start (Windows PowerShell)

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
