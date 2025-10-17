import streamlit as st
import pandas as pd
import yfinance as yf
import aiohttp
import asyncio

st.title("📈 Live Stock Screener (NASDAQ + NYSE + S&P 500)")

# ------------------------------
# Fetch tickers safely
# ------------------------------
@st.cache_data
def get_all_us_tickers():
    tickers = set()

    # Try S&P 500
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers.update(sp500["Symbol"].tolist())
        st.write(f"✅ Loaded {len(sp500)} S&P 500 tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch S&P 500: {e}")

    # Try open datahub NASDAQ + NYSE
    try:
        nasdaq = pd.read_csv("https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv")
        tickers.update(nasdaq["Symbol"].tolist())
        st.write(f"✅ Loaded {len(nasdaq)} NASDAQ tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch NASDAQ tickers: {e}")

    try:
        nyse = pd.read_csv("https://datahub.io/core/nyse-other-listings/r/nyse-listed-symbols.csv")
        tickers.update(nyse["ACT Symbol"].tolist())
        st.write(f"✅ Loaded {len(nyse)} NYSE tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch NYSE tickers: {e}")

    tickers = sorted(list(tickers))
    st.write(f"📊 Total tickers loaded: {len(tickers)}")
    return tickers


tickers = get_all_us_tickers()

# ------------------------------
# Sidebar filters
# ------------------------------
st.sidebar.header("Filter Settings")

min_price = st.sidebar.slider("Minimum Current Price ($)", 0, 500, 5)
min_change = st.sidebar.slider("Minimum % Price Change Today", -5, 20, 5)
min_volume = st.sidebar.slider("Minimum Volume (Shares)", 0, 10_000_000, 500_000)
min_rel_volume = st.sidebar.slider("Min Volume Multiplier vs Intraday Avg", 1.0, 10.0, 5.0)

# ------------------------------
# Async fetch
# ------------------------------
async def fetch_data(session, ticker):
    try:
        data =

