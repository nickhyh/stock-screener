import streamlit as st
import pandas as pd
import yfinance as yf
import asyncio
import datetime
import requests
from io import StringIO
import os
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="📈 Interactive Stock Screener", layout="wide")
st.title("📊 High Demand Stock Screener — S&P500 + NASDAQ + NYSE")
st.markdown("""
Adjust the sliders in the sidebar to filter stocks dynamically.
Only stocks meeting **all criteria ranges** will appear.
""")

# ----------------------------
# Auto-update ticker CSVs
# ----------------------------
@st.cache_data(ttl=86400)
def update_ticker_csvs():
    # ----- S&P 500 -----
    try:
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_table['Symbol'] = sp500_table['Symbol'].str.strip()
        sp500_table.drop_duplicates(subset='Symbol', inplace=True)
        sp500_table.to_csv("sp500.csv", index=False)
    except:
        pass

    # ----- NASDAQ -----
    try:
        nasdaq_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        nasdaq_txt = requests.get(nasdaq_url).text
        nasdaq_df = pd.read_csv(StringIO(nasdaq_txt), sep="|")
        nasdaq_df = nasdaq_df[nasdaq_df['Test Issue'] == 'N']
        nasdaq_df['Symbol'] = nasdaq_df['Symbol'].str.strip()
        nasdaq_df.drop_duplicates(subset='Symbol', inplace=True)
        nasdaq_df.to_csv("nasdaq.csv", index=False)
    except:
        pass

    # ----- NYSE -----
    try:
        nyse_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        nyse_txt = requests.get(nyse_url).text
        nyse_df = pd.read_csv(StringIO(nyse_txt), sep="|")
        nyse_df = nyse_df[nyse_df['Exchange'] == 'N']
        nyse_df['ACT Symbol'] = nyse_df['ACT Symbol'].str.strip()
        nyse_df.drop_duplicates(subset='ACT Symbol', inplace=True)
        nyse_df.to_csv("nyse.csv", index=False)
    except:
        pass

update_ticker_csvs()

# ----------------------------
# Sidebar sliders for criteria
# ----------------------------
st.sidebar.header("Filter Settings")
vol_min, vol_max = st.sidebar.slider("Volume Spike (x) Range", 1, 20, (5, 20))
demand_min, demand_max = st.sidebar.slider("Intraday Price Increase (%) Range", 1, 50, (10, 50))
news_min, news_max = st.sidebar.slider("News Articles Today Range", 0, 5, (1, 5))
float_min, float_max = st.sidebar.slider("Float (Millions) Range", 1, 100, (1, 20))
refresh_minutes = st.sidebar.slider("Auto-refresh (minutes)", 1, 15, 5)

st_autorefresh(interval=refresh_minutes*60*1000, key="datarefresh")

# ----------------------------
# Load tickers
# ----------------------------
@st.cache_data(ttl=86400)
def get_all_us_tickers():
    sp500_df = pd.read_csv("sp500.csv")
    nasdaq_df = pd.read_csv("nasdaq.csv")
    nyse_df = pd.read_csv("nyse.csv")

    sp500_tickers = sp500_df['Symbol'].tolist()
    nasdaq_tickers = nasdaq_df['Symbol'].tolist()
    nyse_tickers = nyse_df['ACT Symbol'].tolist()

    return list(set(sp500_tickers + nasdaq_tickers + nyse_tickers))

tickers = get_all_us_tickers()
if not tickers:
    st.stop()

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
def screen_stock(ticker, vol_range, demand_range, news_range, float_range):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="1m")
        if hist.empty or len(hist) < 2:
            return None

        avg_volume_so_far = hist['Volume'][:-1].mean()
        current_interval_volume = hist['Volume'][-1]
        volume_ratio = current_interval_volume / (avg_volume_so_far if avg_volume_so_far else 1)

        today_open = hist['Open'][0]
        today_close = hist['Close'][-1]
        percent_increase = (today_close - today_open) / today_open * 100

        float_shares = stock.info.get("floatShares", 0)
        news_today = get_news_today(ticker)

        if (
            vol_range[0] <= volume_ratio <= vol_range[1]
            and demand_range[0] <= percent_increase <= demand_range[1]
            and float_shares and float_range[0]*1_000_000 <= float_shares <= float_range[1]*1_000_000
            and news_range[0] <= len(news_today) <= news_range[1]
        ):
            return {
                "Ticker": ticker,
                "Price": today_close,
                "Volume": hist['Volume'].sum(),
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
async def fetch_all_stocks_batched(tickers, vol_range, demand_range, news_range, float_range, batch_size=50):
    loop = asyncio.get_event_loop()
    results = []
    progress = st.progress(0)
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i, batch_start in enumerate(range(0, len(tickers), batch_size)):
        batch = tickers[batch_start:batch_start+batch_size]
        tasks = [loop.run_in_executor(None, screen_stock, t, vol_range, demand_range, news_range, float_range) for t in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend([r for r in batch_results if r])
        progress.progress((i+1)/total_batches)
    return results

# ----------------------------
# Main logic
# ----------------------------
st.info("Scanning NASDAQ, NYSE & S&P 500 stocks. This may take a few minutes...")

results = asyncio.run(
    fetch_all_stocks_batched(
        tickers,
        vol_range=(vol_min, vol_max),
        demand_range=(demand_min, demand_max),
        news_range=(news_min, news_max),
        float_range=(float_min, float_max)
    )
)

if results:
    df = pd.DataFrame(results)
    st.subheader(f"Stocks Matching Criteria ({len(df)})")
    st.dataframe(df, use_container_width=True)

    st.subheader("📰 News Headlines")
    for _, row in df.iterrows():
        st.markdown(f"### {row['Ticker']} — ${row['Price']:.2f}")
        for headline in row['News']:
            st.markdown(f"- {headline}")
        st.mar
