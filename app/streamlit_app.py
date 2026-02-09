import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import yfinance as yf
except Exception:
    yf = None

from features import build_features
from utils import (
    setup_logging, validate_dataframe, validate_model_features,
    ModelLoadError, DataValidationError
)

# Setup logging
logger = setup_logging(level="INFO")

APP_TITLE = "Hybrid Stock Predictor (Professional)"

def load_candles_from_csv(data_dir: str, ticker: str) -> pd.DataFrame:
    """
    Load stock candles from CSV file.
    
    Args:
        data_dir: Directory containing CSV files
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        DataValidationError: If file not found or data is invalid
    """
    try:
        path = os.path.join(data_dir, f"{ticker.upper()}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        
        df = pd.read_csv(path)
        # normalize column names
        df.columns = [c.strip() for c in df.columns]
        
        # Basic validation
        if len(df) < 50:
            raise DataValidationError(
                f"Insufficient data: {len(df)} rows found, minimum 50 required"
            )
        
        logger.info(f"Loaded {ticker} from CSV: {len(df)} rows")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        raise DataValidationError(str(e)) from e
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise DataValidationError(f"Failed to load CSV data: {str(e)}") from e

def load_candles_from_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Load stock candles from yfinance.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (e.g., '2y', '1y')
        interval: Data interval (e.g., '1d', '1h')
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        DataValidationError: If yfinance fails or returns no data
    """
    if yf is None:
        raise DataValidationError(
            "yfinance is not installed or failed to import. "
            "Please use CSV mode or install yfinance."
        )
    
    try:
        logger.info(f"Downloading {ticker} from yfinance (period={period}, interval={interval})")
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        
        if df is None or len(df) == 0:
            raise DataValidationError(
                f"No data returned for ticker={ticker}. "
                "Try CSV mode or check network/ticker symbol."
            )
        
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
        
        logger.info(f"Downloaded {ticker}: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        raise DataValidationError(f"Failed to download from yfinance: {str(e)}") from e

@st.cache_resource
def load_model(model_path: str):
    """
    Load ML model from file with caching.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model object
        
    Raises:
        ModelLoadError: If model loading fails
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise ModelLoadError(f"Failed to load model from {model_path}: {str(e)}") from e

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")
    st.title(APP_TITLE)
    st.caption(
        "Professional stock prediction with hybrid ML models. "
        "Features comprehensive error handling, data validation, and logging."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        ticker = st.text_input("Ticker Symbol", value="TSLA", help="Stock ticker (e.g., AAPL, TSLA)").strip().upper()
        source = st.selectbox("Data Source", ["data (local CSV)", "yfinance (live)"], index=0)
        model_path = st.text_input("Model path", value="models/hybrid_model.pkl")
        data_dir = st.text_input("CSV folder (for local)", value="data")
        
        with st.expander("Advanced Options"):
            period = st.text_input("yfinance period", value="2y", help="e.g., 1y, 2y, 5y")
            interval = st.text_input("yfinance interval", value="1d", help="e.g., 1d, 1h")
        
        run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    if not run_btn:
        st.info("👈 Configure settings in the sidebar then click **Run Prediction**.")
        
        # Display feature information
        st.subheader("📊 Model Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Basic Features (10)**")
            st.markdown("""
            - Returns: `ret_1`, `ret_3`, `ret_5`, `ret_10`, `ret_20`
            - Volatility: `vol_5`, `vol_10`, `vol_20`
            - Drawdown: `dd_20`
            - Range: `range_pct`
            """)
        with col2:
            st.markdown("**Enhanced Features (+8)**")
            st.markdown("""
            - Technical indicators: RSI, MACD, Bollinger Bands
            - Momentum indicators: Momentum, OBV, ATR
            - Trend indicators: EMA ratio, Volume ratio
            - Advanced: Stochastic, ADX
            """)
        return

    # Progress indicator
    with st.spinner("🔄 Loading data and making predictions..."):
        # 1) Load candles
        try:
            if source.startswith("data"):
                candles = load_candles_from_csv(data_dir, ticker)
            else:
                candles = load_candles_from_yfinance(ticker, period=period, interval=interval)
        except (DataValidationError, Exception) as e:
            st.error(f"❌ Failed to load data: {str(e)}")
            logger.error(f"Data loading failed for {ticker}: {e}")
            return

        # 2) Build features
        try:
            feats_df = build_features(candles)
            feats_df = feats_df.dropna().copy()
            if len(feats_df) < 5:
                raise DataValidationError(
                    "Not enough rows after feature engineering. "
                    f"Got {len(feats_df)}, need at least 5. Use more historical data."
                )
        except Exception as e:
            st.error(f"❌ Failed to build features: {str(e)}")
            logger.error(f"Feature engineering failed: {e}")
            return

        # 3) Load model
        try:
            model = load_model(model_path)
            # Get the exact feature order expected by the model (Pipeline's estimator)
            feature_names = validate_model_features(model, feats_df)
        except ModelLoadError as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.info("💡 Run `python scripts/restore_models.py` to restore model files.")
            return
        except Exception as e:
            st.error(f"❌ Model validation failed: {str(e)}")
            logger.error(f"Model validation error: {e}")
            return

        # 4) Latest row prediction
        latest = feats_df.iloc[[-1]].copy()
        # normalize feature names (features.py uses lowercase already)
        for c in list(latest.columns):
            latest.rename(columns={c: c.lower()}, inplace=True)

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
            st.error(f"❌ Prediction failed: {str(e)}")
            logger.error(f"Prediction error: {e}")
            return

    direction = "UP ⬆️" if pred_class == 1 else "DOWN ⬇️"
    direction_color = "green" if pred_class == 1 else "red"

    # Layout
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("📈 Prediction Result")
        st.metric("Ticker", ticker)
        st.markdown(f"**Direction:** :{direction_color}[{direction}]")
        st.metric("Probability (UP)", f"{p_up*100:.2f}%")
        st.metric("Confidence", f"{max(p_up, 1-p_up)*100:.1f}%")
        
        st.divider()
        st.caption(f"Model: {os.path.basename(model_path)}")
        st.caption(f"Features used: {len(feature_names)}")

    with col2:
        st.subheader("📉 Price Chart (Close)")
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
            st.line_chart(chart_df[close_col].dropna(), height=300)
        else:
            st.warning("Close column not found to plot.")

    st.subheader("🔍 Latest Engineered Features")
    st.dataframe(X, use_container_width=True)

    # Download signal snapshot
    out = X.copy()
    out["ticker"] = ticker
    out["direction"] = direction
    out["prob_up"] = p_up
    out["confidence"] = max(p_up, 1-p_up)
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction CSV",
        data=csv_bytes,
        file_name=f"{ticker}_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
