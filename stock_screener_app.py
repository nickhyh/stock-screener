import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import requests
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, MFIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import asyncio
import aiohttp
import nest_asyncio
import os

# Enable nested event loops for async fetches
nest_asyncio.apply()

st.set_page_config(page_title="📈 Real-Time U.S. Stock Screener", layout="wide")

# ---------------------------------------
# Sidebar Filters
# ---------------------------------------
st.sidebar.title("🔍 Stock Screener Settings")
st.sidebar.markdown("Filter U.S. stocks by price, RSI, volume, and more.")

price_min = st.sidebar.number_input("Min Price ($)", 0.0, 1000.0, 5.0)
price_max = st.sidebar.number_input("Max Price ($)", 0.0, 2000.0, 200.0)
min_volume = st.sidebar.number_input("Min Average Volume", 0, 1_000_000_000, 500_000)
rsi_max = st.sidebar.slider("Max RSI (Overbought Filter)", 0, 100, 70)
rsi_min = st.sidebar.slider("Min RSI (Oversold Filter)", 0, 100, 30)
refresh_minutes = st.sidebar.slider("⏱ Auto-refresh every (minutes)", 1, 15, 5)

st.sidebar.markdown("---")
st.sidebar.info("App auto-refreshes periodically for live data.")

# ---------------------------------------
# Utility Functions
# ---------------------------------------

@st.cache_data(ttl=300)
def get_sp500_tickers():
    """Fetch the list of S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df["Symbol"].to_list()

async def fetch_ticker_data(session, ticker):
    """Fetch live data for one ticker."""
    try:
        data = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if data.empty:
            return None

        data["RSI"] = ta_rsi(data["Close"], 14)
        data["OBV"] = OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()
        data["ADL"] = AccDistIndexIndicator(data["High"], data["Low"], data["Close"], data["Volume"]).acc_dist_index()
        data["MFI"] = MFIIndicator(data["High"], data["Low"], data["Close"], data["Volume"]).money_flow_index()
        data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
        data["Volume_Spike"] = data["Volume"] / data["Volume"].rolling(20).mean()

        latest = data.iloc[-1]
        return {
            "Ticker": ticker,
            "Price": latest["Close"],
            "RSI": latest["RSI"],
            "MFI": latest["MFI"],
            "VWAP_Ratio": latest["Close"] / latest["VWAP"],
            "OBV": latest["OBV"],
            "Volume_Spike": latest["Volume_Spike"],
            "AvgVolume": data["Volume"].mean()
        }
    except Exception:
        return None

def ta_rsi(series, period=14):
    """Compute RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

async def fetch_all_data(tickers):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_data(session, t) for t in tickers]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r]

def get_news_for_ticker(ticker):
    """Fetch recent news headlines for a given ticker using NewsAPI."""
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return []
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        articles = response.json().get("articles", [])
        return [a["title"] for a in articles[:3]]
    except Exception:
        return []

# ---------------------------------------
# Data Processing & Filtering
# ---------------------------------------
st.title("📊 Real-Time U.S. Stock Screener")
st.caption("Featuring real-time volume indicators, technical filters, and news sentiment")

with st.spinner("Fetching live market data..."):
    tickers = get_sp500_tickers()
    results = asyncio.run(fetch_all_data(tickers))

df = pd.DataFrame(results)
if df.empty:
    st.error("No data retrieved. Try again later.")
    st.stop()

# Apply filters
filtered = df[
    (df["Price"].between(price_min, price_max)) &
    (df["AvgVolume"] > min_volume) &
    (df["RSI"].between(rsi_min, rsi_max))
]

# ---------------------------------------
# Display Results
# ---------------------------------------
st.subheader(f"📈 Matching Stocks ({len(filtered)})")

if len(filtered) > 0:
    st.dataframe(
        filtered.sort_values("Volume_Spike", ascending=False),
        use_container_width=True,
    )

    # Display top 3 with news
    st.subheader("📰 Top 3 Volume Movers + Latest News")
    top3 = filtered.sort_values("Volume_Spike", ascending=False).head(3)
    for _, row in top3.iterrows():
        st.markdown(f"### {row['Ticker']} — ${row['Price']:.2f}")
        headlines = get_news_for_ticker(row["Ticker"])
        if headlines:
            for h in headlines:
                st.markdown(f"- {h}")
        else:
            st.markdown("_No recent news available._")
        st.markdown("---")
else:
    st.warning("No stocks match your criteria.")

# ---------------------------------------
# Auto-refresh logic
# ---------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

time.sleep(refresh_minutes * 60)
st.experimental_rerun()
