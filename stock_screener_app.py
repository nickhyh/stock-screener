import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import requests
import os
import time

st.set_page_config(page_title="📈 High Demand Stock Screener", layout="wide")

st.title("📊 High Demand Stock Screener")
st.markdown("""
Filter U.S. stocks that meet the following criteria:
- Volume spike ≥ 5× 20-day average volume
- Price/demand increase ≥ 10% intraday
- At least one news headline today
- Float shares < 20 million
""")

# ----------------------------
# Sidebar settings
# ----------------------------
refresh_minutes = st.sidebar.slider("⏱ Auto-refresh every (minutes)", 1, 15, 5)
st.sidebar.info("App auto-refreshes periodically for live data.")

# ----------------------------
# Helper functions
# ----------------------------
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """Fetch S&P 500 tickers from static CSV"""
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    return df['Symbol'].tolist()

def get_news_today(ticker):
    """Return list of news headlines published today"""
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

def screen_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="21d", interval="1d")
        if hist.empty or len(hist) < 2:
            return None
        
        # Volume spike
        today_volume = hist['Volume'][-1]
        avg_volume = hist['Volume'][:-1].mean()
        if avg_volume == 0:
            return None
        volume_ratio = today_volume / avg_volume
        
        # Demand increase (intraday % change)
        today_open = hist['Open'][-1]
        today_close = hist['Close'][-1]
        percent_increase = (today_close - today_open) / today_open * 100
        
        # Float shares
        float_shares = stock.info.get("floatShares", 0)
        
        # News today
        news_today = get_news_today(ticker)
        
        if (
            volume_ratio >= 5
            and percent_increase >= 10
            and len(news_today) > 0
            and float_shares is not None
            and float_shares < 20_000_000
        ):
            return {
                "Ticker": ticker,
                "Price": today_close,
                "Volume": today_volume,
                "Volume/Avg": round(volume_ratio, 2),
                "Demand%": round(percent_increase, 2),
                "FloatShares": float_shares,
                "News": news_today[:3]
            }
        return None
    except:
        return None

# ----------------------------
# Main logic
# ----------------------------
st.info("Scanning S&P 500 stocks... this may take a few minutes.")
tickers = get_sp500_tickers()
results = []

for t in tickers:
    r = screen_stock(t)
    if r:
        results.append(r)

if results:
    df = pd.DataFrame(results)
    st.subheader(f"Stocks Matching Criteria ({len(df)})")
    st.dataframe(df, use_container_width=True)

    st.subheader("📰 News Headlines for Matching Stocks")
    for _, row in df.iterrows():
        st.markdown(f"### {row['Ticker']} — ${row['Price']:.2f}")
        for headline in row['News']:
            st.markdown(f"- {headline}")
        st.markdown("---")
else:
    st.warning("No stocks match the criteria today.")

# ----------------------------
# Auto-refresh
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

time.sleep(refresh_minutes * 60)
st.experimental_rerun()

