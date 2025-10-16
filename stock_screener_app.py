"""
Streamlit Stock Screener
- Screens US stocks (attempts comprehensive list from Nasdaq/NYSE)
- Filters: price, MA, PE, PB, dividend yield, RSI, MACD, avg volume, market cap, sector/exchange
- Volume indicators: VSR, OBV, VWAP, MFI, Accum/Dist
- Real-time supply indicators (approx): up/down volume imbalance over last N minutes
- News screener: uses NewsAPI or Finnhub if API key provided; else uses yfinance news
- Auto-refresh via st_autorefresh
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import time
import math
from datetime import datetime, timedelta
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
import nest_asyncio

# allow nested event loop (for some environments)
nest_asyncio.apply()
load_dotenv()

# Config
DEFAULT_REFRESH_SECONDS = 300  # 5 minutes
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")

st.set_page_config(page_title="Advanced Stock Screener", layout="wide")

# ---------------------------
# Utilities: ticker sources
# ---------------------------
@st.cache_data(ttl=60*60*24)
def fetch_us_tickers():
    """
    Attempt to build a comprehensive US ticker list from NASDAQ Trader symbol directories.
    Fallback to S&P 500 list from Wikipedia if network failures occur.
    Returns DataFrame with columns: symbol, exchange
    """
    sources = {
        "nasdaq": "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "other": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    }
    tickers = []
    try:
        for name,url in sources.items():
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                text = r.text
                df = pd.read_csv(io.StringIO(text), sep="|")
                # select Symbol or ACT Symbol depending on file
                if 'Symbol' in df.columns:
                    syms = df['Symbol'].dropna().tolist()
                elif 'ACT Symbol' in df.columns:
                    syms = df['ACT Symbol'].dropna().tolist()
                else:
                    syms = []
                for s in syms:
                    tickers.append((s.strip(), name.upper()))
        if len(tickers) == 0:
            raise Exception("No tickers found")
        df_out = pd.DataFrame(tickers, columns=["symbol","source"])
        df_out = df_out.drop_duplicates().reset_index(drop=True)
        return df_out
    except Exception as e:
        st.warning(f"Failed to fetch full exchange lists ({e}). Falling back to S&P 500 list.")
        try:
            wiki = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            df_sp = wiki[0]
            df_sp = df_sp[['Symbol','Security','GICS Sector','Headquarters Location']]
            df_sp = df_sp.rename(columns={'Symbol':'symbol'})
            df_sp['source'] = 'SP500'
            return df_sp[['symbol','source']]
        except Exception as e2:
            st.error("Failed to load fallback S&P500 list. Please provide a tickers CSV.")
            return pd.DataFrame(columns=["symbol","source"])

# ---------------------------
# Indicator computations
# ---------------------------
def compute_indicators(history_df):
    """
    given a dataframe with index Datetime and columns ['Open','High','Low','Close','Volume'],
    compute indicators and return a dict (or series) of recent indicator values
    """
    results = {}
    df = history_df.copy()
    if df.empty or len(df) < 5:
        return results

    # Ensure numeric
    df = df[['Open','High','Low','Close','Volume']].astype(float).dropna()

    # VSR: current volume / 20-day avg volume -> but if intraday pass shorter window
    try:
        avg_vol = df['Volume'].tail(20).mean()
        cur_vol = df['Volume'].iloc[-1]
        results['VSR'] = float(cur_vol / avg_vol) if avg_vol and avg_vol>0 else np.nan
    except Exception:
        results['VSR'] = np.nan

    # OBV
    try:
        obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        results['OBV'] = float(obv.iloc[-1])
    except Exception:
        results['OBV'] = np.nan

    # VWAP (works best intraday)
    try:
        vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14)
        results['VWAP'] = float(vwap.volume_weighted_average_price()[-1])
    except Exception:
        results['VWAP'] = np.nan

    # MFI (Money Flow Index) 14-period
    try:
        mfi = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14)
        results['MFI'] = float(mfi.money_flow_index()[-1])
    except Exception:
        results['MFI'] = np.nan

    # Accumulation/Distribution
    try:
        ad = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
        results['AD'] = float(ad.acc_dist_index()[-1])
    except Exception:
        results['AD'] = np.nan

    # RSI (14)
    try:
        rsi = RSIIndicator(close=df['Close'], window=14)
        results['RSI'] = float(rsi.rsi()[-1])
    except Exception:
        results['RSI'] = np.nan

    # MACD
    try:
        macd = MACD(close=df['Close'])
        results['MACD'] = float(macd.macd_diff()[-1])
    except Exception:
        results['MACD'] = np.nan

    # basic price checks
    results['Close'] = float(df['Close'].iloc[-1])
    results['MA50'] = float(df['Close'].rolling(window=50, min_periods=1).mean().iloc[-1])
    results['MA200'] = float(df['Close'].rolling(window=200, min_periods=1).mean().iloc[-1])
    results['AvgVolume'] = float(df['Volume'].tail(20).mean())

    return results

# ---------------------------
# Real-time supply indicator approximation
# ---------------------------
def compute_supply_indicators(minute_df, lookback_minutes=15):
    """
    minute_df: intraday minute bars (index datetime, columns O H L C V)
    Returns:
      - up_volume_pct: percent of recent volume that occurred on upticks
      - down_volume_pct: percent on downticks
      - sell_pressure_ratio: down_volume / (up_volume + down_volume)
    """
    res = {'up_volume_pct': np.nan, 'down_volume_pct': np.nan, 'sell_pressure': np.nan}
    if minute_df is None or minute_df.empty:
        return res
    df = minute_df.copy().dropna()
    if df.empty:
        return res

    # restrict lookback
    try:
        cutoff = df.index.max() - pd.Timedelta(minutes=lookback_minutes)
        df_recent = df[df.index > cutoff]
        if df_recent.empty:
            df_recent = df.tail(lookback_minutes)
    except Exception:
        df_recent = df

    # uptick: close > previous close; downtick: close < previous close
    closes = df_recent['Close']
    volume = df_recent['Volume']
    prev = closes.shift(1)
    uptick_mask = closes > prev
    downtick_mask = closes < prev
    up_vol = volume[uptick_mask].sum()
    down_vol = volume[downtick_mask].sum()
    total = up_vol + down_vol
    if total == 0:
        res['up_volume_pct'] = 0.0
        res['down_volume_pct'] = 0.0
        res['sell_pressure'] = 0.0
    else:
        res['up_volume_pct'] = float(up_vol / total)
        res['down_volume_pct'] = float(down_vol / total)
        res['sell_pressure'] = float(down_vol / total)
    return res

# ---------------------------
# News fetcher (NewsAPI or Finnhub fallback, else yfinance)
# ---------------------------
def fetch_news_for_ticker(ticker, max_items=5):
    # prefer NewsAPI
    if NEWSAPI_KEY:
        try:
            url = ("https://newsapi.org/v2/everything?"
                   f"q={ticker}&pageSize={max_items}&apiKey={NEWSAPI_KEY}")
            r = requests.get(url, timeout=10)
            j = r.json()
            if j.get('status') == 'ok':
                articles = j.get('articles', [])
                return [{'title':a['title'],'url':a['url'],'source':a['source']['name'],'publishedAt':a['publishedAt']} for a in articles]
        except Exception:
            pass
    if FINNHUB_KEY:
        try:
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={(datetime.utcnow()-timedelta(days=7)).date()}&to={datetime.utcnow().date()}&token={FINNHUB_KEY}"
            r = requests.get(url, timeout=10)
            j = r.json()
            if isinstance(j, list):
                out = []
                for item in j[:max_items]:
                    out.append({'title': item.get('headline'), 'url': item.get('url'), 'source': item.get('source'), 'publishedAt': item.get('datetime')})
                return out
        except Exception:
            pass
    # fallback: yfinance Ticker.news (not always present)
    try:
        t = yf.Ticker(ticker)
        news = t.news
        out = []
        for a in news[:max_items]:
            out.append({'title': a.get('title'), 'url': a.get('link') or a.get('link'), 'source': a.get('publisher') or a.get('publisher'), 'publishedAt': a.get('providerPublishTime')})
        return out
    except Exception:
        return []

# ---------------------------
# Screening routine (single ticker)
# ---------------------------
def analyze_ticker(ticker, intraday_minutes=60):
    """
    Returns a dict with fundamentals (if available) and computed indicators.
    """
    out = {'symbol': ticker}
    try:
        t = yf.Ticker(ticker)
        # fundamentals
        info = {}
        try:
            info = t.get_info()
        except Exception:
            try:
                info = t.info
            except Exception:
                info = {}
        out['shortName'] = info.get('shortName', '')
        out['sector'] = info.get('sector', '')
        out['marketCap'] = info.get('marketCap', np.nan)
        out['previousClose'] = info.get('previousClose', np.nan)
        out['trailingPE'] = info.get('trailingPE', np.nan)
        out['priceToBook'] = info.get('priceToBook', np.nan)
        out['dividendYield'] = info.get('dividendYield', np.nan)
        # historical daily (for MA, PE, etc)
        hist = t.history(period="6mo", interval="1d", actions=False)
        if hist is not None and not hist.empty:
            hist = hist[['Open','High','Low','Close','Volume']].dropna()
            ind = compute_indicators(hist)
            out.update(ind)
        else:
            # try smaller set
            hist = t.history(period="60d", interval="1d")
            if hist is not None and not hist.empty:
                hist = hist[['Open','High','Low','Close','Volume']].dropna()
                ind = compute_indicators(hist)
                out.update(ind)

        # intraday minute bars to compute supply indicators
        try:
            minute = t.history(period="1d", interval="1m")
            if minute is not None and not minute.empty:
                minute = minute[['Open','High','Low','Close','Volume']].dropna()
                supply = compute_supply_indicators(minute, lookback_minutes=intraday_minutes)
                out.update(supply)
            else:
                out.update({'up_volume_pct': np.nan, 'down_volume_pct': np.nan, 'sell_pressure': np.nan})
        except Exception:
            out.update({'up_volume_pct': np.nan, 'down_volume_pct': np.nan, 'sell_pressure': np.nan})
        # news (only headline count here)
        try:
            news = fetch_news_for_ticker(ticker, max_items=3)
            out['news_count_3d'] = len(news)
            out['news_headlines'] = news
        except Exception:
            out['news_count_3d'] = 0
            out['news_headlines'] = []
    except Exception as e:
        out['error'] = str(e)
    return out

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📡 Advanced Real-Time Stock Screener (Streamlit)")

# sidebar controls
st.sidebar.header("Screener Settings")
tickers_df = fetch_us_tickers()
st.sidebar.write(f"Detected {len(tickers_df)} tickers from exchange lists (may include ETFs).")
exchange_filter = st.sidebar.multiselect("Source(s) to include (exchange source)", options=sorted(tickers_df['source'].unique().tolist()), default=None)
if exchange_filter:
    symbol_list = tickers_df[tickers_df['source'].isin(exchange_filter)]['symbol'].tolist()
else:
    symbol_list = tickers_df['symbol'].tolist()

# allow user to upload a custom CSV of tickers (single column 'symbol' or 'Ticker')
uploaded = st.sidebar.file_uploader("Optional: upload CSV of tickers (column 'symbol')", type=["csv"])
if uploaded is not None:
    df_custom = pd.read_csv(uploaded)
    col = None
    for possible in ['symbol','Ticker','ticker','SYMBOL']:
        if possible in df_custom.columns:
            col = possible
            break
    if col:
        symbol_list = df_custom[col].dropna().astype(str).str.upper().tolist()
        st.sidebar.success(f"Loaded {len(symbol_list)} tickers from uploaded CSV.")
    else:
        st.sidebar.error("Couldn't find a 'symbol' column in uploaded CSV. Using exchange list.")

# core numeric filters
st.sidebar.header("Numeric Filters (leave blank to ignore)")
price_min = st.sidebar.number_input("Price >= ", value=0.0, step=0.01)
price_max = st.sidebar.number_input("Price <= ", value=1e9, step=0.01)
pe_max = st.sidebar.number_input("P/E <= ", value=1e9, step=0.1)
pb_max = st.sidebar.number_input("P/B <= ", value=1e9, step=0.1)
div_yield_min = st.sidebar.number_input("Dividend yield >= ", value=0.0, step=0.001, format="%.4f")
rsi_min = st.sidebar.number_input("RSI >= ", value=0.0, max_value=100.0)
rsi_max = st.sidebar.number_input("RSI <= ", value=100.0, max_value=100.0)
macd_min = st.sidebar.number_input("MACD diff >= ", value=-1e9, step=0.01)
vol_avg_min = st.sidebar.number_input("Avg daily volume >= ", value=0.0, step=1000.0)
marketcap_min = st.sidebar.number_input("Market cap >= ", value=0.0, step=1e6)

# Volume indicator thresholds
st.sidebar.header("Volume Indicator Thresholds")
vsr_min = st.sidebar.number_input("VSR (curVol / 20dAvg) >= ", value=0.0, step=0.1)
obv_positive = st.sidebar.checkbox("Require OBV positive?", value=False)
mfi_min = st.sidebar.number_input("MFI >= ", value=0.0, step=1.0)
ad_positive = st.sidebar.checkbox("Require A/D positive?", value=False)

# Supply indicator thresholds
st.sidebar.header("Real-time Supply (approx)")
sell_pressure_max = st.sidebar.number_input("Sell pressure (0-1) <= ", value=1.0, min_value=0.0, max_value=1.0, step=0.01)
lookback_mins = st.sidebar.number_input("Supply lookback minutes", value=15, min_value=1, max_value=120, step=1)

# news filter
st.sidebar.header("News Event Screener")
require_news = st.sidebar.checkbox("Require recent news?", value=False)
news_days = st.sidebar.number_input("News lookback (days)", value=3, min_value=1, max_value=30)

# additional options
st.sidebar.header("Other")
autorefresh_sec = st.sidebar.number_input("Auto-refresh interval (seconds)", value=DEFAULT_REFRESH_SECONDS, min_value=30, step=30)
max_results = st.sidebar.number_input("Max tickers to evaluate (0 = all)", value=200, step=10)
run_button = st.sidebar.button("Run Screener Now")

st.sidebar.markdown("""
**Notes**
- For comprehensive exchange lists the first run may take several minutes.
- For truly real-time/level-II data, connect a paid data provider and update `analyze_ticker` accordingly.
""")

# autorefresh widget (Streamlit)
count = st.experimental_get_query_params().get("refresh_count", [0])
# use st.experimental_rerun alternative: st_autorefresh
from streamlit.runtime.scriptrunner import add_script_run_ctx
try:
    from streamlit_autorefresh import st_autorefresh
    autoref = st_autorefresh(interval=autorefresh_sec * 1000, limit=None, key="autorefresh")
except Exception:
    # fallback: simple timer display
    autoref = None
    st.sidebar.info("Auto-refresh library not installed; press 'Run Screener Now' to refresh manually.")

# Show top bar
st.write("Filters applied. Click **Run Screener Now** to start scanning tickers (may take a while).")
placeholder = st.empty()

# screening loop (single run)
if run_button:
    start_time = time.time()
    symbol_list = list(dict.fromkeys([s.upper() for s in symbol_list]))  # unique
    if max_results > 0:
        symbol_list = symbol_list[:max_results]
    st.info(f"Analyzing {len(symbol_list)} tickers. This can take time; results will appear below as they finish.")
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    N = len(symbol_list)

    for i, sym in enumerate(symbol_list):
        status_text.text(f"Processing {i+1}/{N}: {sym}")
        try:
            out = analyze_ticker(sym, intraday_minutes=int(lookback_mins))
            # apply simple filters quickly to avoid heavy work if missing price
            close = out.get('Close', np.nan) or out.get('previousClose', np.nan) or np.nan
            if np.isnan(close):
                # skip
                pass
            else:
                # apply user filters
                if (close < price_min) or (close > price_max):
                    pass
                elif (out.get('trailingPE') and out.get('trailingPE') > pe_max):
                    pass
                elif (out.get('priceToBook') and out.get('priceToBook') > pb_max):
                    pass
                elif (out.get('dividendYield') and out.get('dividendYield') < div_yield_min):
                    pass
                elif (out.get('RSI') is not None and (out.get('RSI') < rsi_min or out.get('RSI') > rsi_max)):
                    pass
                elif (out.get('MACD') is not None and out.get('MACD') < macd_min):
                    pass
                elif (out.get('AvgVolume') is not None and out.get('AvgVolume') < vol_avg_min):
                    pass
                elif (out.get('marketCap') is not None and out.get('marketCap') < marketcap_min):
                    pass
                elif (out.get('VSR') is not None and out.get('VSR') < vsr_min):
                    pass
                elif obv_positive and out.get('OBV', 0) <= 0:
                    pass
                elif out.get('MFI', 0) < mfi_min:
                    pass
                elif ad_positive and out.get('AD', 0) <= 0:
                    pass
                elif out.get('sell_pressure', 0) > sell_pressure_max:
                    pass
                elif require_news and out.get('news_count_3d', 0) == 0:
                    pass
                else:
                    results.append(out)
        except Exception as e:
            # ignore ticker errors
            pass

        progress_bar.progress(int((i+1)/N * 100))

    elapsed = time.time() - start_time
    st.success(f"Scan finished in {elapsed:.1f}s — {len(results)} tickers passed filters.")
    if len(results) == 0:
        st.write("No tickers matched your filters. Try loosening thresholds.")
    else:
        df_res = pd.DataFrame(results)
        # expand news headlines count
        if 'news_headlines' in df_res.columns:
            df_res['news_count_3'] = df_res['news_count_3d']
        # sor table by VSR descending
        df_display = df_res.sort_values(by='VSR', ascending=False).reset_index(drop=True)
        st.dataframe(df_display[['symbol','shortName','Close','VSR','OBV','VWAP','RSI','MACD','AvgVolume','marketCap','sell_pressure','news_count_3d']].fillna('N/A'), height=600)
        csv = df_display.to_csv(index=False)
        st.download_button("Download results CSV", csv, file_name=f"screener_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

        # show headlines for first selected item
        sel_idx = st.number_input("Show news for result index (0-based)", min_value=0, max_value=len(df_display)-1, value=0)
        selected = df_display.iloc[int(sel_idx)]
        st.subheader(f"News & indicators for {selected['symbol']}")
        st.write(f"Short name: {selected.get('shortName','')}")
        st.write(f"Latest close: {selected.get('Close','')}")
        st.write("Recent headlines:")
        for item in selected.get('news_headlines', []) or []:
            title = item.get('title','')
            url = item.get('url','')
            src = item.get('source','')
            published = item.get('publishedAt','')
            st.markdown(f"- [{title}]({url}) — {src} — {published}")

# End of file
