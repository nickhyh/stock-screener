import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

st.title("📈 Live Stock Screener")

# ------------------------------
# Fetch ticker lists automatically
# ------------------------------
@st.cache_data
def get_all_us_tickers():
    tickers = []

    # S&P 500 from Wikipedia
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers += sp500["Symbol"].tolist()
        st.write(f"✅ Loaded {len(sp500)} S&P 500 tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch S&P 500: {e}")

    # NASDAQ tickers
    try:
        nasdaq_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?exchange=NASDAQ&render=download"
        nasdaq = pd.read_csv(nasdaq_url)
        tickers += nasdaq["Symbol"].tolist()
        st.write(f"✅ Loaded {len(nasdaq)} NASDAQ tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch NASDAQ tickers: {e}")

    # NYSE tickers
    try:
        nyse_url = "https://old.nasdaq.com/screening/companies-by-name.aspx?exchange=NYSE&render=download"
        nyse = pd.read_csv(nyse_url)
        tickers += nyse["Symbol"].tolist()
        st.write(f"✅ Loaded {len(nyse)} NYSE tickers.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch NYSE tickers: {e}")

    tickers = sorted(list(set(tickers)))
    st.write(f"📊 Total tickers loaded: {len(tickers)}")
    return tickers


tickers = get_all_us_tickers()

# ------------------------------
# User-adjustable screening criteria
# ------------------------------
st.sidebar.header("Filter Settings")

min_price = st.sidebar.slider("Minimum Current Price ($)", 0, 500, 5)
min_change = st.sidebar.slider("Minimum % Price Change Today", -5, 10, 0)
min_volume = st.sidebar.slider("Minimum Volume (Shares)", 0, 10_000_000, 500_000)
min_rel_volume = st.sidebar.slider("Min Volume Multiplier vs Intraday Avg", 1.0, 10.0, 2.0)

# ------------------------------
# Screening function
# ------------------------------
@st.cache_data(show_spinner=False)
def screen_stocks(tickers):
    found = []
    total_checked = 0

    for ticker in tickers[:300]:  # limit for speed
        try:
            data = yf.Ticker(ticker).history(period="2d", interval="1h")
            if len(data) < 2:
                continue

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

            total_checked += 1
        except Exception as e:
            continue

    st.write(f"🔍 Checked {total_checked} stocks.")
    st.write(f"✅ Found {len(found)} matching stocks.")
    return found


if st.button("Run Screener"):
    matches = screen_stocks(tickers)
    if matches:
        st.success("Stocks meeting criteria:")
        st.write(matches)
    else:
        st.warning("No stocks currently match your criteria.")

