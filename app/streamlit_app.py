import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Add project root to path for imports (so `import src.*` works)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf
except Exception:
    yf = None

from src.features import build_features
from src.enhanced_features import build_enhanced_features
from src.seven_system import SevenGatesEngine
from src.seven_config import DEFAULT_SEVEN

APP_TITLE = "Hybrid Stock Predictor (Local)"
FALLBACK_VOL = 0.01

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
    st.caption("Runs locally. Uses model proba then applies Seven System gates for ranking + strict risk management.")

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

    # 2) Load model (first, to decide which feature builder to use)
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

    # 3) Build features (basic vs enhanced)
    try:
        feature_names = list(feature_names)
        wants_enhanced = ("atr_14" in feature_names) or (len(feature_names) > 10)
        if wants_enhanced:
            feats_df = build_enhanced_features(candles)
        else:
            feats_df = build_features(candles)

        feats_df = feats_df.dropna().copy()
        if len(feats_df) < 25:
            raise RuntimeError("Not enough rows after feature engineering. Use more history (need >= ~25 for percentiles).")
    except Exception as e:
        st.error(f"Failed to build features: {e}")
        return

    # 4) Prepare X for all clean rows (needed for ranking percentiles)
    feats_df.columns = [c.lower() for c in feats_df.columns]
    feature_names = [str(f).lower() for f in feature_names]

    missing = [f for f in feature_names if f not in feats_df.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.write("Available columns:", list(feats_df.columns))
        return

    X_all = feats_df[list(feature_names)].astype(float)

    # 5) Predict probabilities (NO thresholding; Seven System decides)
    try:
        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(X_all)[:, 1]
            proba_series = pd.Series(proba_all, index=X_all.index)
            p_up = float(proba_series.iloc[-1])
        else:
            # fallback
            pred_all = np.asarray(model.predict(X_all)).reshape(-1)
            proba_series = pd.Series(np.clip(np.abs(pred_all), 0.0, 1.0), index=X_all.index)
            p_up = float(proba_series.iloc[-1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # =========================
    # Seven System integration (Gates 1-4, 6-7)
    # =========================
    engine = SevenGatesEngine(DEFAULT_SEVEN)

    ok1, r1 = engine.gate1_universe(feats_df)
    ok2, r2 = engine.gate2_regime(feats_df)
    ok3, r3 = engine.gate3_event_risk(feats_df)

    if not (ok1 and ok2 and ok3):
        decision = {
            "action": "BLOCKED",
            "reason": f"gate_fail: g1={r1}, g2={r2}, g3={r3}",
            "proba": float(p_up),
        }
    else:
        decision = engine.gate4_decision(p_up, proba_series)

    if "seven_blocks" not in st.session_state:
        st.session_state["seven_blocks"] = []

    # Layout
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Result")
        st.metric("Ticker", ticker)
        st.metric("Prob(UP)", f"{p_up*100:.2f}%")
        st.metric("Seven Action", decision.get("action", ""))
        st.write("Reason:", decision.get("reason", ""))
        if "q_hi" in decision or "q_lo" in decision:
            st.write({"q_hi": decision.get("q_hi"), "q_lo": decision.get("q_lo")})
        st.caption("Decision is based on no_trade_band + percentiles ranking (no 0.5 threshold anywhere).")
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
    st.dataframe(X_all.tail(1), use_container_width=True)

    # Gate 6-7 (only if LONG)
    if decision.get("action") == "LONG":
        # Determine last close price
        last_close = None
        if "close" in feats_df.columns:
            last_close = float(feats_df["close"].iloc[-1])
        else:
            # fallback from candles
            c_map = {c.lower(): c for c in candles.columns}
            close_col = c_map.get("close")
            if close_col is not None:
                last_close = float(pd.to_numeric(candles[close_col], errors="coerce").dropna().iloc[-1])
        if last_close is None:
            st.error("Could not determine last close price for sizing.")
            return

        # ATR norm: prefer enhanced atr_14, else fallback to vol_20
        if "atr_14" in feats_df.columns:
            atr_norm = float(feats_df["atr_14"].iloc[-1])
        else:
            vol_20 = float(feats_df["vol_20"].iloc[-1]) if "vol_20" in feats_df.columns else FALLBACK_VOL
            atr_norm = max(vol_20, FALLBACK_VOL)

        with st.sidebar:
            st.header("Risk (Seven System)")
            equity = st.number_input("Equity (capital)", min_value=0.0, value=10000.0, step=100.0)

        sizing = engine.gate6_size(equity=float(equity), price=last_close, atr_norm=atr_norm)
        exits = engine.gate7_exits(entry_price=last_close, stop_price=sizing["stop_price"])

        st.subheader("Seven System – Risk & Exits")
        st.write({
            "entry_price": last_close,
            "atr_norm": atr_norm,
            "shares": sizing["shares"],
            "stop_price": sizing["stop_price"],
            "take_profit": exits["take_profit"],
            "partial_take_profit": exits["partial_take_profit"],
            "move_stop_to_BE_at": exits["move_stop_to_BE_at"],
            "time_stop_bars": exits["time_stop_bars"],
        })
    else:
        st.session_state["seven_blocks"].append({
            "ticker": ticker,
            "proba_up": float(p_up),
            "action": decision.get("action"),
            "reason": decision.get("reason"),
        })
        st.warning("Blocked by Seven (no sizing shown).")
        if st.session_state["seven_blocks"]:
            st.subheader("Seven System – Block Log")
            st.dataframe(pd.DataFrame(st.session_state["seven_blocks"]).tail(20), use_container_width=True)

    # Download signal snapshot
    out = X_all.tail(1).copy()
    out["ticker"] = ticker
    out["prob_up"] = p_up
    out["seven_action"] = decision.get("action")
    out["seven_reason"] = decision.get("reason")
    if "q_hi" in decision:
        out["q_hi"] = decision.get("q_hi")
    if "q_lo" in decision:
        out["q_lo"] = decision.get("q_lo")
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download prediction CSV", data=csv_bytes, file_name=f"{ticker}_prediction.csv", mime="text/csv")

if __name__ == "__main__":
    main()
