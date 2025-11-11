# app.py - Upgraded NIFTY ML Dashboard

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

# -------------- INDICATORS --------------
def add_indicators(df):
    df = df.copy()
    df['Ret1'] = df['Close'].pct_change(1)
    df['Ret2'] = df['Close'].pct_change(2)
    df['Ret5'] = df['Close'].pct_change(5)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2*std
    df['BB_lower'] = df['BB_mid'] - 2*std

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
    return {
        "nifty": joblib.load(MODEL_NIFTY_PATH) if os.path.exists(MODEL_NIFTY_PATH) else None,
        "combined": joblib.load(MODEL_COMB_PATH) if os.path.exists(MODEL_COMB_PATH) else None
    }

# -------------- UI --------------
st.set_page_config(page_title="NIFTY ML Dashboard", layout="wide")
st.title("📊 NIFTY 5-Day ML Prediction + Technical Dashboard")

# Sidebar
st.sidebar.header("⚙ Chart Controls")
show_ma10 = st.sidebar.checkbox("MA10", True)
show_ma20 = st.sidebar.checkbox("MA20", False)
show_ma50 = st.sidebar.checkbox("MA50", True)
show_ma200 = st.sidebar.checkbox("MA200", True)
show_bb = st.sidebar.checkbox("Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("RSI", False)
show_macd = st.sidebar.checkbox("MACD", False)

# Data & features
df = download_nifty_daily()
df = add_indicators(df)
models = load_models()

# ML Input
feature_cols = ['Ret1','Ret2','Ret5','MA10','MA20','MA50','MA200']
last_X = df.iloc[-1:][feature_cols]

# Predictions
predictions = {}
for k, model in models.items():
    if model:
        prob = float(model.predict_proba(last_X)[0,1])
        lbl, icon = label_signal(prob)
        predictions[k] = (prob, lbl, icon)
    else:
        predictions[k] = (None, "No Model", "⚪")

# ML Signals on chart
df['ML_signal'] = None
for i in range(200, len(df)):
    try:
        Xrow = df.iloc[i:i+1][feature_cols]
        p = float(models['nifty'].predict_proba(Xrow)[0,1])
        df.loc[df.index[i], 'ML_signal'] = "BUY" if p > 0.55 else ("SELL" if p < 0.45 else None)
    except:
        continue

# -------------- CHART --------------
st.subheader("📈 Price + Indicators + ML Signals")

fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(
    x=df.index, open=df.Open, high=df.High,
    low=df.Low, close=df.Close, name="Price"
))

# MAs
if show_ma10:  fig.add_trace(go.Scatter(x=df.index, y=df.MA10, name="MA10"))
if show_ma20:  fig.add_trace(go.Scatter(x=df.index, y=df.MA20, name="MA20"))
if show_ma50:  fig.add_trace(go.Scatter(x=df.index, y=df.MA50, name="MA50"))
if show_ma200: fig.add_trace(go.Scatter(x=df.index, y=df.MA200, name="MA200"))

# Bollinger
if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_upper, name="BB Upper", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_lower, name="BB Lower", line=dict(dash='dot')))

# ML Buy/Sell markers
buy_ml = df[df.ML_signal == "BUY"]
sell_ml = df[df.ML_signal == "SELL"]

fig.add_trace(go.Scatter(
    x=buy_ml.index, y=buy_ml.Close, mode="markers",
    marker=dict(color="lime", size=13, symbol="triangle-up"), name="ML BUY"
))
fig.add_trace(go.Scatter(
    x=sell_ml.index, y=sell_ml.Close, mode="markers",
    marker=dict(color="red", size=13, symbol="triangle-down"), name="ML SELL"
))

fig.update_layout(height=620, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# -------------- OPTIONAL PANELS --------------
if show_rsi:
    st.subheader("RSI (14)")
    st.line_chart(df['RSI'])

if show_macd:
    st.subheader("MACD")
    st.line_chart(df[['MACD','MACD_signal']])

# -------------- PREDICTIONS --------------
st.subheader("🤖 Model Forecast (Next 5 Days)")
c1, c2 = st.columns(2)

for i,(name,data) in enumerate(predictions.items()):
    prob, label, icon = data
    if prob:
        (c1 if i==0 else c2).metric(
            f"{name.upper()} Model",
            f"{icon} {label}",
            f"{round(prob*100,2)}% confidence"
        )
    else:
        (c1 if i==0 else c2).metric(f"{name.upper()} Model", "⚪ Not Loaded")

st.write("---")
if st.button("🔁 Reload Dashboard"):
    st.rerun()

st.caption("💡 Indicators are toggleable from sidebar | ML signals plotted on chart | Forecast for next 5 days")
