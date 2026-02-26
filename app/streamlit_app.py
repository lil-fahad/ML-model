import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Add project and src directories to path for imports
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _p in [ROOT_DIR, SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import yfinance as yf
except Exception:
    yf = None

from features import build_features
from src.seven_system import SevenGatesEngine
from src.seven_config import DEFAULT_SEVEN

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
        feats_df.columns = [c.lower() for c in feats_df.columns]
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
    missing = [f for f in feature_names if f not in feats_df.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.write("Available columns:", list(feats_df.columns))
        return
    X_all = feats_df[list(feature_names)].astype(float)
    latest_full = feats_df.iloc[[-1]].copy()
    X = X_all.iloc[[-1]].astype(float)

    # 5) Predict
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # fallback
            pred = float(model.predict(X)[0])
            p_up = float(min(1.0, max(0.0, abs(pred))))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    proba_series = None
    if hasattr(model, "predict_proba"):
        try:
            proba_series = pd.Series(model.predict_proba(X_all)[:, 1])
        except Exception:
            proba_series = None

    seven_engine = SevenGatesEngine(DEFAULT_SEVEN)
    g1_ok, g1_reason = seven_engine.gate1_universe(feats_df)
    g2_ok, g2_reason = seven_engine.gate2_regime(feats_df)
    g3_ok, g3_reason = seven_engine.gate3_event_risk(feats_df)

    if not g1_ok:
        gate4 = {"action": "BLOCKED", "reason": f"gate1:{g1_reason}", "proba": p_up}
    elif not g2_ok:
        gate4 = {"action": "BLOCKED", "reason": f"gate2:{g2_reason}", "proba": p_up}
    elif not g3_ok:
        gate4 = {"action": "BLOCKED", "reason": f"gate3:{g3_reason}", "proba": p_up}
    else:
        gate4 = seven_engine.gate4_decision(p_up, proba_series)

    action = gate4.get("action", "NO_TRADE")

    # Layout
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Result")
        st.metric("Ticker", ticker)
        st.metric("Seven Action", action)
        st.metric("Prob(UP)", f"{p_up*100:.2f}%")
        st.write("Model expects features:", list(feature_names))
        st.write("Reason:", gate4.get("reason", ""))
        if "q_hi" in gate4 or "q_lo" in gate4:
            st.write("Percentiles:", {"q_hi": gate4.get("q_hi"), "q_lo": gate4.get("q_lo")})
        close_lookup = {c.lower(): c for c in candles.columns}
        close_col = close_lookup.get("close") or close_lookup.get("close".lower()) or close_lookup.get("Close".lower())
        last_close_price = None
        if close_col and close_col in candles.columns:
            last_close_price = float(candles[close_col].iloc[-1])

        if action == "LONG":
            atr_norm = float(latest_full.get("atr_14", pd.Series([np.nan])).iloc[0]) if "atr_14" in latest_full.columns else float(max(latest_full.get("vol_20", pd.Series([0])).iloc[0], 0.01))
            if np.isnan(atr_norm) or atr_norm <= 0:
                atr_norm = float(max(latest_full.get("vol_20", pd.Series([0])).iloc[0], 0.01))
            equity = st.number_input("Equity (capital)", min_value=0.0, value=10000.0, step=100.0, format="%.2f")
            if last_close_price is not None:
                sizing = seven_engine.gate6_size(equity, last_close_price, atr_norm)
                exits = seven_engine.gate7_exits(entry_price=last_close_price, stop_price=sizing["stop_price"])

                st.write("Sizing:", sizing)
                st.write("Exits:", exits)
            else:
                st.warning("Cannot compute sizing without close price.")
        else:
            st.info("Seven System action is not LONG. Blocked by Seven.")

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
    out["action"] = action
    out["reason"] = gate4.get("reason", "")
    out["prob_up"] = p_up
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download prediction CSV", data=csv_bytes, file_name=f"{ticker}_prediction.csv", mime="text/csv")

if __name__ == "__main__":
    main()
