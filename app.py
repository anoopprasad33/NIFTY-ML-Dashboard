# app.py (Upgraded Streamlit dashboard with Buy/Sell markers + UI improvements)

import os
import joblib
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
MODEL_NIFTY_PATH = os.getenv("MODEL_NIFTY_PATH", "model_nifty_xgb.joblib")
MODEL_COMB_PATH = os.getenv("MODEL_COMB_PATH", "model_combined_xgb.joblib")

# -------------- DATA FETCH --------------
@st.cache_data(ttl=60*60)
def download_nifty_daily(start="1996-01-01"):
    df = yf.download("^NSEI", start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# -------------- FEATURES --------------
def add_features_daily(df):
    df = df.copy()
    df['Ret1'] = df['Close'].pct_change(1)
    df['Ret2'] = df['Close'].pct_change(2)
    df['Ret5'] = df['Close'].pct_change(5)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df.dropna(inplace=True)
    return df

def label_signal(prob):
    if prob >= 0.60: return "STRONG BUY", "🟩"
    if prob >= 0.52: return "BUY", "🟢"
    if prob <= 0.40: return "STRONG SELL", "🟥"
    if prob <= 0.48: return "SELL", "🔴"
    return "HOLD", "⚪"

# -------------- LOAD MODEL --------------
def load_models():
    models = {}
    models['nifty'] = joblib.load(MODEL_NIFTY_PATH) if os.path.exists(MODEL_NIFTY_PATH) else None
    models['combined'] = joblib.load(MODEL_COMB_PATH) if os.path.exists(MODEL_COMB_PATH) else None
    return models

# -------------- STREAMLIT UI --------------
st.set_page_config(page_title="NIFTY 5-Day ML Dashboard", layout="wide")
st.title("📈 NIFTY 5-Day ML Prediction Dashboard")

# Layout
left, right = st.columns([2, 1])

# Load data
df = download_nifty_daily()
df_feat = add_features_daily(df)
models = load_models()

# Prepare last input features
feature_cols = ['Ret1','Ret2','Ret5','MA10','MA20','MA50','MA200']
last_X = df_feat.iloc[-1:][feature_cols]

# Predictions
pred_out = {}
for m in models:
    if models[m]:
        p = float(models[m].predict_proba(last_X)[0, 1])
        lbl, icon = label_signal(p)
        pred_out[m] = {"prob": p, "label": lbl, "icon": icon}
    else:
        pred_out[m] = {"prob": None, "label": "No Model", "icon": "⚪"}

# ---------------- LEFT SIDE: CHART ----------------
with left:
    st.subheader("NIFTY Price Chart with Signals")

    dfc = df.copy()
    dfc['Signal'] = None

    # ✅ FIX: Ensure required columns exist before plotting
    dfc['MA50'] = dfc['Close'].rolling(50).mean()
    dfc['MA200'] = dfc['Close'].rolling(200).mean()
    dfc.dropna(inplace=True)

    # Generate signals for chart markers
    for i in range(1, len(dfc)):
        dfc.loc[dfc.index[i], 'Signal'] = "BUY" if dfc['Close'].iloc[i] > dfc['MA50'].iloc[i] else "SELL"

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=dfc.index, open=dfc["Open"], high=dfc["High"],
        low=dfc["Low"], close=dfc["Close"], name="NIFTY"
    ))

    # ✅ Safe MA plotting
    if "MA50" in dfc.columns:
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc["MA50"], name="MA50"))
    if "MA200" in dfc.columns:
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc["MA200"], name="MA200"))

    # Buy/Sell markers
    buys = dfc[dfc.Signal == "BUY"]
    sells = dfc[dfc.Signal == "SELL"]

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys['Close'],
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="green"),
        name="BUY"
    ))

    fig.add_trace(go.Scatter(
        x=sells.index, y=sells['Close'],
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="red"),
        name="SELL"
    ))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- RIGHT SIDE: PREDICTIONS ----------------
with right:
    st.subheader("📊 Model Prediction (Next 5 Days)")

    for key in ["nifty", "combined"]:
        v = pred_out[key]
        if v["prob"] is not None:
            st.metric(
                label=f"{key.upper()} Model",
                value=f"{v['icon']} {v['label']}",
                delta=f"{round(v['prob'] * 100, 2)}% confidence"
            )
        else:
            st.metric(label=f"{key.upper()} Model", value="⚪ No model loaded")

    st.write("---")
    st.subheader("⚙ Actions")

    if st.button("🔁 Reload Model"):
        st.rerun()

    if st.button("♻ Retrain Model (Offline Logic Placeholder)"):
        st.info("Retraining is not configured live. Train offline & replace model files.")

st.write("---")
st.caption("Tip: BUY/SELL signals on chart based on MA trend for visualization. ML predicts next 5 days direction.")
