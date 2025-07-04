import streamlit as st
from data_feed import get_binance_ohlcv
from strategy import preprocess, make_prediction
from portfolio import PaperPortfolio
from datetime import datetime
import time

st.set_page_config(page_title="BTC Trading Bot", layout="wide")
st.title("🧠 BTC Machine Learning Paper Trading Bot")

if "last_run" not in st.session_state:
    st.session_state.last_run = time.time()
elif time.time() - st.session_state.last_run > 300:  # 5 minutes = 300s
    st.session_state.last_run = time.time()
    st.rerun()
# === Session state ===
if "portfolio" not in st.session_state:
    st.session_state.portfolio = PaperPortfolio()

# === Fetch data ===
df = get_binance_ohlcv()
df = preprocess(df)
print(len(df))
#price = df.loc[df.index[-1], 'Close']
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# === Predict ===
proba = make_prediction(df)
st.metric("Prediction Proba", f"{proba:.3f}")
st.metric("BTC/USDT Price", f"{price:.2f}")

# === Strategy ===
threshold = 0.6
portfolio = st.session_state.portfolio

if proba > threshold and portfolio.position == 0:
    portfolio.buy(price, timestamp)
elif proba < (1 - threshold) and portfolio.position == 1:
    portfolio.sell(price, timestamp)

portfolio.update_equity(price)

# === Display ===
st.subheader("📈 Equity Curve")
st.line_chart(portfolio.equity_curve)

st.subheader("📋 Trade Log")
st.dataframe(portfolio.trade_log[::-1])

# === Export ===
if st.button("💾 Export trades to CSV"):
    portfolio.export_trades_csv()
    with open("trades.csv", "rb") as f:
        st.download_button("Download CSV", f, file_name="trades.csv")

# === Metrics ===
st.metric("💰 Cash", f"${portfolio.cash:.2f}")
st.metric("📊 Equity", f"${portfolio.current_equity():.2f}")
