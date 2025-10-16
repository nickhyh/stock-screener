import streamlit as st
import pandas as pd
import yfinance as yf
import asyncio
import requests
import datetime
import os

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
# Fetch tickers
# ----------------------------
@st.cache_data(ttl=86400)
def get_all_us_tickers():
    # S&P 500
    sp500_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    sp500_df = pd.read_csv(sp500_url)
    sp500_tickers = sp500_df['Symbol'].tolist()
    
    # NASDAQ
    nasdaq_url = "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv"
    nasdaq_df = pd.read_csv(nasdaq_url)
    nasdaq_tickers = nasdaq_df['Symbol'].tolist()
    
    # NYSE
    nyse_url = "https://datahub.io/core/nyse-other-listings/r/other-listed.csv"
    nyse_df = pd.read_csv(nyse_url)
    nyse_tickers = nyse_df['ACT Symbol'].tolist()
    
    all_tickers = list(set(sp500_tickers + nasdaq_tickers + nyse_tickers))
    return all_tickers

# ----------------------------
# News fetch
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
        hist = stock.history(period="21d", interval="1d")
        if hist.empty or len(hist) < 2:
            return None
        
        today_volume = hist['Volume'][-1]
        avg_volume = hist['Volume'][:-1].mean()
        if avg_volume == 0:
            return None
        volume_ratio = today_volume / avg_volume
        
        today_open = hist['Open'][-1]
        today_close = hist['Close'][-1]
        percent_increase = (today_close - today_open) / today_open * 100
        
        float_shares = stock.info.get("floatShares", 0)
        news_today = get_news_today(ticker)
        
        if (
            volume_ratio >= 5
            and percent_increase >= 10
            and float_shares and float_shares < 20_000_000
            and news_today
        ):
            return {
                "Ticker": ticker,
                "Price": today_close,
                "Volume": today_volume,
                "Volume/Avg": round(volume_ratio,2),
                "Demand%": round(percent_increase,2),
                "FloatShares": float_shares,
                "News": news_today[:3]
            }
        return None
    except:
        return None

# ----------------------------
# Async fetching
# ---

