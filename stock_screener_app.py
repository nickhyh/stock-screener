import streamlit as st
import pandas as pd
import yfinance as yf
import asyncio
import datetime
import requests
import os
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="📈 High Demand Stock Screener", layout="wide")
st.title("📊 High Demand Stock Screener — NASDAQ, NYSE & S&P 500")

st.markdown("""
Filters for stocks with:
- Volume spike ≥ 5× 20-day avg
- Intraday demand increase ≥ 10%
- At least one news event today
- Float < 20M shares
""")

refresh_minutes = st.sidebar.slider("⏱ Auto-refresh every (minutes)", 1, 15, 5)
st.sidebar.info("App auto-refreshes periodically.")

# ----------------------------
# Auto-refresh (non-blocking)
# ----------------------------
st_autorefresh(interval=refresh_minutes*60*1000, key="datarefresh")

# ----------------------------
# Load tickers from local CSVs
# ----------------------------
@st.cache_data(ttl=86400)
def get_all_us_tickers():
    sp500_df = pd.read_csv("sp500.csv")
    nasdaq_df = pd.read_csv("nasdaq.csv")
    nyse_df = pd.read_csv("nyse.csv")

    sp500_tickers = sp500_df['Symbol'].tolist()
    nasdaq_tickers = nasdaq_df['Symbol'].tolist()
    nyse_tickers = nyse_df['ACT Symbol'].tolist()

    all_tickers = list(set(sp500_tickers + nasdaq_tickers + nyse_tickers))
    return all_tickers

# ----------------------------
# Fetch news
# ----------------------------
def get_news_today(ticker):
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return []
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={today}&to={today}&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=5).json()
        articles = response.get("articles", [])
        return [a["title"] for a in articles]
    except:
        return []

# ----------------------------
# Screening function
# ----------------------------
def screen_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="1m")  # intraday 1-minute data
        if hist.empty or len(hist) < 2:
            return None

        # ----------------- Volume filter -----------------
        cumulative_volume = hist['Volume'].sum()
        avg_volume_so_far = hist['Volume'][:-1].mean()  # average of previous intervals
        current_interval_volume = hist['Volume'][-1]
        volume_ratio = current_interval_volume / (avg_volume_so_far if avg_volume_so_far else 1)

        # ----------------- Price / Demand filter ------------
        today_open = hist['Open'][0]
        today_close = hist['Close'][-1]
        percent_increase = (today_close - today_open) / today_open * 100

        # ----------------- Float & News -----------------
        float_shares = stock.info.get("floatShares", 0)
        news_today = get_news_today(ticker)

        # ----------------- Criteria -----------------
        if (
            volume_ratio >= 5
            and percent_increase >= 10
            and float_shares and float_shares < 20_000_000
            and news_today
        ):
            return {
                "Ticker": ticker,
                "Price": today_close,
                "Volume": cumulative_volume,
                "Volume/IntervalRatio": round(volume_ratio, 2),
                "Demand%": round(percent_increase, 2),
                "FloatShares": float_shares,
                "News": news_today[:3]
            }
        return None
    except:
        return None


# ----------------------------
# Async batch fetching
# ----------------------------
async def fetch_all_stocks_batched(tickers, batch_size=50):
    loop = asyncio.get_event_loop()
    results = []
    progress = st.progress(0)
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i, batch_start in enumerate(range(0, len(tickers), batch_size)):
        batch = tickers[batch_start:batch_start+batch_size]
        tasks = [loop.run_in_executor(None, screen_stock, t) for t in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend([r for r in batch_results if r])
        progress.progress((i+1)/total_batches)
    return results

# ----------------------------
# Main logic
# ----------------------------
st.info("Scanning NASDAQ, NYSE & S&P 500 stocks. This may take a few minutes...")

tickers = get_all_us_tickers()
results = asyncio.run(fetch_all_stocks_batched(tickers, batch_size=50))

if results:
    df = pd.DataFrame(results)
    st.subheader(f"Stocks Matching Criteria ({len(df)})")
    st.dataframe(df, use_container_width=True)

    st.subheader("📰 News Headlines")
    for _, row in df.iterrows():
        st.markdown(f"### {row['Ticker']} — ${row['Price']:.2f}")
        for headline in row['News']:
            st.markdown(f"- {headline}")
        st.markdown("---")
else:
    st.warning("No stocks match the criteria today.")

