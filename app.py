# app.py (Streamlit dashboard — no Twilio / WhatsApp)
import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -----------------------
# Config / env
# -----------------------
GS_SA_PATH = os.getenv("GS_SERVICE_ACCOUNT_JSON")  # path to service account json
MODEL_NIFTY_PATH = os.getenv("MODEL_NIFTY_PATH", "model_nifty_xgb.joblib")
MODEL_COMB_PATH = os.getenv("MODEL_COMB_PATH", "model_combined_xgb.joblib")

# -----------------------
# Utilities
# -----------------------
@st.cache_data(ttl=60*60)
def download_nifty_daily(start="1996-01-01"):
    df = yf.download("^NSEI", start=start, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def add_features_daily(df):
    df = df.copy()
    df['Ret1'] = df['Close'].pct_change(1)
    df['Ret2'] = df['Close'].pct_change(2)
    df['Ret5'] = df['Close'].pct_change(5)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['Vol20'] = df['Ret1'].rolling(20).std()
    df['Momentum_21'] = df['Close'].pct_change(21)
    df['ATR14'] = (df['High'] - df['Low']).rolling(14).mean()
    df['MA50_200'] = df['MA50'] / df['MA200']
    df['Close_MA20_pct'] = (df['Close'] - df['MA20']) / df['MA20']
    df['Close_MA50_pct'] = (df['Close'] - df['MA50']) / df['MA50']
    df = df.dropna()
    return df

def make_horizon_target(df, horizon=5, threshold=0.0):
    df = df.copy()
    df['FutureClose'] = df['Close'].shift(-horizon)
    df['FutureReturn'] = df['FutureClose'] / df['Close'] - 1
    df['Target'] = (df['FutureReturn'] > threshold).astype(int)
    df = df.dropna(subset=['FutureClose', 'FutureReturn', 'Target'])
    return df

def label_prob(p, buy_th=0.60, sell_th=0.40):
    if p is None: return 'N/A'
    if p >= buy_th: return 'STRONG BUY'
    if p >= 0.52: return 'BUY'
    if p <= sell_th: return 'STRONG SELL'
    if p <= 0.48: return 'SELL'
    return 'HOLD'

# Google Sheets helpers
def gs_auth(json_path):
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
    client = gspread.authorize(creds)
    return client

def append_to_sheet(sheet_id, sheet_name, rows, json_path=GS_SA_PATH):
    client = gs_auth(json_path)
    sh = client.open_by_key(sheet_id)
    try:
        worksheet = sh.worksheet(sheet_name)
    except Exception:
        worksheet = sh.add_worksheet(title=sheet_name, rows=1000, cols=20)
    # if rows is list of dicts, convert
    if isinstance(rows, list) and len(rows) and isinstance(rows[0], dict):
        headers = list(rows[0].keys())
        existing = worksheet.row_values(1)
        if not existing:
            worksheet.append_row(headers)
        for r in rows:
            worksheet.append_row([r.get(h,"") for h in headers])
    else:
        for r in rows:
            worksheet.append_row(r)

# Model load
def load_models(nifty_path=MODEL_NIFTY_PATH, comb_path=MODEL_COMB_PATH):
    models = {}
    if os.path.exists(nifty_path):
        try:
            models['nifty'] = joblib.load(nifty_path)
        except Exception as e:
            models['nifty'] = None
            st.error(f"Failed loading NIFTY model: {e}")
    else:
        models['nifty'] = None
    if os.path.exists(comb_path):
        try:
            models['combined'] = joblib.load(comb_path)
        except Exception as e:
            models['combined'] = None
            st.error(f"Failed loading Combined model: {e}")
    else:
        models['combined'] = None
    return models

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(layout="wide", page_title="NIFTY 5-day ML Dashboard")
st.title("NIFTY 5-day ML Prediction Dashboard (Streamlit only)")

col1, col2 = st.columns([2,1])

with col2:
    st.subheader("Controls")
    model_reload = st.button("Reload Models")
    do_export = st.button("Append current prediction to Google Sheet")
    sheet_id = st.text_input("Google Sheet ID (leave blank to skip export)", value="")
    horizon = st.number_input("Horizon (days)", value=5, min_value=1, max_value=30)

with col1:
    st.subheader("NIFTY Chart (daily)")
    df_daily = download_nifty_daily()
    st.write(f"Data from {df_daily.index[0].date()} to {df_daily.index[-1].date()}")
    st.dataframe(df_daily.tail(3))

    df_feat = add_features_daily(df_daily)
    df_model = make_horizon_target(df_feat, horizon=horizon)

    if model_reload:
        st.experimental_rerun()
    models = load_models()

    days = st.slider("Days to show", 120, 365*3, 365)
    df_plot = df_daily.iloc[-days:].copy()
    df_plot['MA50'] = df_plot['Close'].rolling(50).mean()
    df_plot['MA200'] = df_plot['Close'].rolling(200).mean()

    fig = go.Figure(data=[go.Candlestick(x=df_plot.index,
                                         open=df_plot['Open'],
                                         high=df_plot['High'],
                                         low=df_plot['Low'],
                                         close=df_plot['Close'],
                                         name='NIFTY')])
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], mode='lines', name='MA50'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA200'], mode='lines', name='MA200'))
    fig.update_layout(height=650, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Inference: last observed features
    features = ['Ret1','Ret2','Ret5','MA10','MA20','MA50','MA200','Vol20','Momentum_21','ATR14','MA50_200','Close_MA20_pct','Close_MA50_pct']
    last_feat = df_feat.iloc[-1:][features].fillna(method='ffill').fillna(method='bfill')

    results = {}
    for name, mod in models.items():
        if mod is None:
            results[name] = {"prob": None, "label": None}
            continue
        prob = float(mod.predict_proba(last_feat)[0,1])
        lab = label_prob(prob)
        cls = int(prob > 0.5)
        results[name] = {"prob": prob, "label": lab, "class": cls}

    st.subheader(f"ML Predictions ({horizon}-day horizon)")
    c1, c2 = st.columns(2)
    c1.metric("NIFTY-only P(up next {})".format(horizon),
              f"{(results['nifty']['prob'] if results['nifty']['prob'] is not None else 'N/A'):.3f}" if results['nifty']['prob'] else "N/A",
              results['nifty']['label'] if results['nifty']['label'] else "")
    c2.metric("Combined model P(up next {})".format(horizon),
              f"{(results['combined']['prob'] if results['combined']['prob'] is not None else 'N/A'):.3f}" if results['combined']['prob'] else "N/A",
              results['combined']['label'] if results['combined']['label'] else "")

    if models['nifty'] is not None or models['combined'] is not None:
        st.write("Model results (raw):")
        st.table(pd.DataFrame([results['nifty'], results['combined']]).T)

with col2:
    st.subheader("Actions / Export")
    if sheet_id:
        st.write("Sheet ID:", sheet_id)
    if do_export:
        if not sheet_id:
            st.error("Please enter Google Sheet ID to append.")
        elif not GS_SA_PATH or not os.path.exists(GS_SA_PATH):
            st.error("GS_SERVICE_ACCOUNT_JSON not set or file not found. Set env var and upload JSON.")
        else:
            try:
                row = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "horizon_days": horizon,
                    "nifty_prob": results['nifty']['prob'] if results['nifty']['prob'] is not None else "",
                    "nifty_label": results['nifty']['label'] if results['nifty']['label'] else "",
                    "combined_prob": results['combined']['prob'] if results['combined']['prob'] is not None else "",
                    "combined_label": results['combined']['label'] if results['combined']['label'] else ""
                }
                append_to_sheet(sheet_id, "predictions", [row], json_path=GS_SA_PATH)
                st.success("Appended prediction row to Google Sheet.")
            except Exception as e:
                st.error("Failed to write to Google Sheet: " + str(e))

st.markdown("---")
st.caption("Notes: Put trained joblib models in the working dir or set MODEL_* env var paths. Use last-observed features to predict next H-day outcome. Backtest / retraining should be done offline and models saved as joblib files. Do not share your service account JSON publicly.")
