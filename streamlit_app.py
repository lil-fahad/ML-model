import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

from features import build_features

APP_TITLE = "Hybrid Stock Predictor (Local)"

def load_candles_from_csv(data_dir: str, ticker: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{ticker.upper()}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def load_candles_from_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed or failed to import.")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for ticker={ticker}. Try CSV mode or check network.")
    df = df.reset_index()
    # yfinance returns columns: Date, Open, High, Low, Close, Adj Close, Volume
    # rename to match features.py expectations
    rename = {}
    if "Date" in df.columns: rename["Date"] = "date"
    if "Open" in df.columns: rename["Open"] = "open"
    if "High" in df.columns: rename["High"] = "high"
    if "Low" in df.columns: rename["Low"] = "low"
    if "Close" in df.columns: rename["Close"] = "close"
    if "Volume" in df.columns: rename["Volume"] = "volume"
    df = df.rename(columns=rename)
    return df

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Runs locally. Uses hybrid_model.pkl (RandomForestClassifier pipeline) and builds the required 10 features.")

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="TSLA").strip().upper()
        source = st.selectbox("Data Source", ["data (local CSV)", "yfinance (live)"], index=0)
        model_path = st.text_input("Model path", value="models/hybrid_model.pkl")
        data_dir = st.text_input("CSV folder (for local)", value="data")
        period = st.text_input("yfinance period", value="2y")
        interval = st.text_input("yfinance interval", value="1d")
        run_btn = st.button("Run Prediction", type="primary")

    if not run_btn:
        st.info("Set options in the sidebar then click **Run Prediction**.")
        return

    # 1) Load candles
    try:
        if source.startswith("data"):
            candles = load_candles_from_csv(data_dir, ticker)
        else:
            candles = load_candles_from_yfinance(ticker, period=period, interval=interval)
    except Exception as e:
        st.error(f"Failed to load candles: {e}")
        return

    # 2) Build features
    try:
        feats_df = build_features(candles)
        feats_df = feats_df.dropna().copy()
        if len(feats_df) < 5:
            raise RuntimeError("Not enough rows after feature engineering. Use more history.")
    except Exception as e:
        st.error(f"Failed to build features: {e}")
        return

    # 3) Load model
    try:
        model = load_model(model_path)
        # Get the exact feature order expected by the model (Pipeline's estimator)
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None and hasattr(model, "named_steps"):
            est = list(model.named_steps.values())[-1]
            feature_names = getattr(est, "feature_names_in_", None)
        if feature_names is None:
            # fallback to the known 10 features
            feature_names = np.array(['ret_1','ret_3','ret_5','ret_10','ret_20','vol_5','vol_10','vol_20','dd_20','range_pct'])
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # 4) Latest row prediction
    latest = feats_df.iloc[[-1]].copy()
    # normalize feature names (features.py uses lowercase already)
    for c in list(latest.columns):
        latest.rename(columns={c: c.lower()}, inplace=True)

    missing = [f for f in feature_names if f not in latest.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.write("Available columns:", list(latest.columns))
        return

    X = latest[list(feature_names)].astype(float)

    # 5) Predict
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            pred_class = int(model.predict(X)[0])
            # assume binary: class 1 = UP
            p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # fallback
            pred = float(model.predict(X)[0])
            pred_class = 1 if pred > 0 else 0
            p_up = float(min(1.0, max(0.0, abs(pred))))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    direction = "UP" if pred_class == 1 else "DOWN"

    # Layout
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Result")
        st.metric("Ticker", ticker)
        st.metric("Direction", direction)
        st.metric("Prob(UP)", f"{p_up*100:.2f}%")
        st.write("Model expects features:", list(feature_names))

    with col2:
        st.subheader("Price chart (Close)")
        # Try to plot close if present
        c_map = {c.lower(): c for c in candles.columns}
        close_col = c_map.get("close") or c_map.get("Close".lower())
        if close_col is None:
            # yfinance path
            if "close" in candles.columns:
                close_col = "close"
            elif "Close" in candles.columns:
                close_col = "Close"
        if close_col and close_col in candles.columns:
            chart_df = candles.copy()
            if "date" in chart_df.columns:
                chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
                chart_df = chart_df.sort_values("date")
                chart_df = chart_df.set_index("date")
            elif "Date" in chart_df.columns:
                chart_df["Date"] = pd.to_datetime(chart_df["Date"], errors="coerce")
                chart_df = chart_df.sort_values("Date")
                chart_df = chart_df.set_index("Date")
            st.line_chart(chart_df[close_col].dropna())
        else:
            st.warning("Close column not found to plot.")

    st.subheader("Latest engineered features")
    st.dataframe(X, use_container_width=True)

    # Download signal snapshot
    out = X.copy()
    out["ticker"] = ticker
    out["direction"] = direction
    out["prob_up"] = p_up
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download prediction CSV", data=csv_bytes, file_name=f"{ticker}_prediction.csv", mime="text/csv")

if __name__ == "__main__":
    main()
