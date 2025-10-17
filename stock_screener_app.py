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
        data = yf.download(ticker, period="2d", interval="1h", progress=False)
        if len(data) < 2:
            return None
        return ticker, data
    except Exception:
        return None

async def fetch_all(tickers):
    async with aiohttp.ClientSession():
        tasks = [asyncio.to_thread(fetch_data, None, t) for t in tickers[:300]]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# ------------------------------
# Screening function
# ------------------------------
def screen_stocks(tickers):
    found = []
    results = asyncio.run(fetch_all(tickers))
    st.write(f"🔍 Checked {len(results)} tickers.")

    for result in results:
        ticker, data = result
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


if st.button("Run Screener"):
    matches = screen_stocks(tickers)
    if matches:
        st.success("Stocks meeting criteria:")
        st.write(matches)
    else:
        st.warning("No stocks currently match your criteria.")
