import streamlit as st
import pandas as pd
import yfinance as yf
import aiohttp
import asyncio
import io

st.title("📈 Live Stock Screener (NASDAQ + NYSE + S&P 500)")

# ------------------------------
# SAFE TICKER FETCHING
# ------------------------------
@st.cache_data
def get_all_us_tickers():
    tickers = set()

    def try_source(name, func):
        try:
            lst = func()
            tickers.update(lst)
            st.write(f"✅ Loaded {len(lst)} {name} tickers.")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch {name}: {e}")

    # --- S&P 500 ---
    try_source(
        "S&P 500",
        lambda: pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
    )

    # --- NASDAQ ---
    try_source(
        "NASDAQ",
        lambda: pd.read_csv(
            "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv"
        )["Symbol"].dropna().tolist()
    )

    # --- NYSE ---
    try_source(
        "NYSE",
        lambda: pd.read_csv(
            "https://datahub.io/core/nyse-other-listings/r/nyse-listed-symbols.csv"
        )["ACT Symbol"].dropna().tolist()
    )

    if not tickers:
        st.error("❌ No remote tickers could be loaded. Using fallback list.")
        fallback_csv = io.StringIO(
            "AAPL\nMSFT\nGOOG\nAMZN\nTSLA\nNVDA\nMETA\nNFLX\nJPM\nDIS"
        )
        tickers = set(pd.read_csv(fallback_csv, header=None)[0].tolist())

    tickers = sorted(list(tickers))
    st.write(f"📊 Total tickers loaded: {len(tickers)}")
    return tickers


tickers = get_all_us_tickers()

# ------------------------------
# SLIDERS
# ------------------------------
st.sidebar.header("Filter Settings")

min_price = st.sidebar.slider("Minimum Current Price ($)", 0, 500, 5)
min_change = st.sidebar.slider("Minimum % Price Change Today", -5, 20, 5)
min_volume = st.sidebar.slider("Minimum Volume (Shares)", 0, 10_000_000, 500_000)
min_rel_volume = st.sidebar.slider("Min Volume Multiplier vs Intraday Avg", 1.0, 10.0, 5.0)

# ------------------------------
# ASYNC DATA FETCH
# ------------------------------
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
    tasks = [fetch_data(t) for t in tickers[:300]]  # limit to 300 for speed
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
    return results

# ------------------------------
# SCREEN FUNCTION
# ------------------------------
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

# ------------------------------
# RUN
# ------------------------------
if st.button("Run Screener"):
    matches = screen_stocks(tickers)
    if matches:
        st.success("Stocks meeting criteria:")
        st.write(matches)
    else:
        st.warning("No stocks currently match your criteria.")

