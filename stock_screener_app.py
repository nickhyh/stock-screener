import streamlit as st
import pandas as pd
import yfinance as yf
import asyncio
import aiohttp
import io
import requests
import time

st.set_page_config(page_title="Live Stock Screener", layout="wide")
st.title("📈 Live Stock Screener — NASDAQ + NYSE + S&P 500")

# -----------------------------------------------------------
# FETCH ALL U.S. TICKERS (robust + cached)
# -----------------------------------------------------------
@st.cache_data
def get_all_us_tickers():
    tickers = set()

    # --- NASDAQ/NYSE/AMEX (official file) ---
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        df = pd.read_csv(io.StringIO(res.text), sep="|")
        symbols = df[df["Test Issue"] == "N"]["Symbol"].dropna().unique().tolist()
        tickers.update(symbols)
        st.write(f"✅ Loaded {len(symbols)} NASDAQ/NYSE tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch NASDAQ/NYSE: {e}")

    # --- S&P 500 (mirror) ---
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        tickers.update(df["Symbol"].dropna().tolist())
        st.write(f"✅ Loaded {len(df)} S&P 500 tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch S&P 500: {e}")

    if not tickers:
        st.error("❌ No remote tickers could be loaded — using fallback.")
        fallback = io.StringIO("AAPL\nMSFT\nGOOG\nAMZN\nTSLA\nNVDA\nMETA\nNFLX\nJPM\nDIS")
        tickers = set(pd.read_csv(fallback, header=None)[0].tolist())

    tickers = sorted(list(tickers))
    st.write(f"📊 Total tickers loaded: {len(tickers)}")
    return tickers


tickers = get_all_us_tickers()

# -----------------------------------------------------------
# SIDEBAR FILTERS (SLIDERS)
# -----------------------------------------------------------
st.sidebar.header("🔧 Screening Criteria (Minimum Thresholds)")

min_price = st.sidebar.slider("Minimum Current Price ($)", 0, 500, 5)
min_change = st.sidebar.slider("Minimum % Price Change Today", -5, 20, 5)
min_volume = st.sidebar.slider("Minimum Volume (Shares)", 0, 10_000_000, 500_000)
min_rel_volume = st.sidebar.slider("Min Volume × vs Intraday Avg", 1.0, 10.0, 5.0)

auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", value=False)

# -----------------------------------------------------------
# ASYNC DATA FETCHING
# -----------------------------------------------------------
async def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="2d", interval="1h", progress=False)
        if data is None or len(data) < 2:
            return None
        return ticker, data
    except Exception:
        return None


async def fetch_all(tickers):
    results = []
    tasks = [fetch_data(t) for t in tickers[:300]]  # limit for efficiency
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
    return results


# -----------------------------------------------------------
# SCREEN LOGIC
# -----------------------------------------------------------
def screen_stocks(tickers):
    found = []
    results = asyncio.run(fetch_all(tickers))
    st.write(f"🔍 Checked {len(results)} tickers.")

    for ticker, data in results:
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["Close"]
        change_pct = ((price - prev["Close"]) / prev["Close"]) * 100
        volume = current["Volume"]
        intraday_avg_vol = data["Volume"].mean()
        rel_vol = volume / intraday_avg_vol if intraday_avg_vol > 0 else 0

        if (
            price >= min_price
            and change_pct >= min_change
            and volume >= min_volume
            and rel_vol >= min_rel_volume
        ):
            found.append(ticker)

    st.write(f"✅ Found {len(found)} matching stocks.")
    return found


# -----------------------------------------------------------
# RUN & REFRESH LOOP
# -----------------------------------------------------------
def run_screener():
    matches = screen_stocks(tickers)
    if matches:
        st.success("📊 Stocks meeting criteria:")
        st.write(matches)
    else:
        st.warning("No stocks currently match your criteria.")


if auto_refresh:
    while True:
        st.experimental_rerun()
        time.sleep(60)
else:
    if st.button("Run Screener Now"):
        run_screener()

