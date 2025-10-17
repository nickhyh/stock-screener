# stock_screener_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import asyncio
import requests
import io
import datetime
import os
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Strict Criteria Stock Screener", layout="wide")
st.title("🔎 Strict-Criteria Stock Screener — S&P500 + NASDAQ + NYSE")
st.caption("Shows only ticker symbols that meet ALL minimum thresholds you set in the sidebar.")

# -------------------------
# Sidebar: sliders (minimum thresholds)
# -------------------------
st.sidebar.header("Minimum thresholds (stock must meet ALL)")

vol_multiplier_min = st.sidebar.slider("Volume spike multiplier (current interval ≥ x × avg so far)", 1.0, 50.0, 5.0, step=0.5)
demand_pct_min = st.sidebar.slider("Intraday demand increase (%) from open", 0.0, 200.0, 10.0, step=0.5)
news_min = st.sidebar.slider("Minimum news items today (NewsAPI)", 0, 5, 1)
max_float_millions = st.sidebar.slider("Maximum float (millions)", 0.1, 500.0, 20.0, step=0.1)

batch_size = st.sidebar.number_input("Async batch size (tune for performance)", min_value=10, max_value=500, value=50, step=10)
limit_tickers = st.sidebar.number_input("Max tickers to scan (0 = all)", min_value=0, value=0, step=100)

autorefresh = st.sidebar.checkbox("Auto-refresh every minute", value=False)
st.sidebar.markdown("Note: News filtering requires `NEWS_API_KEY` in Streamlit secrets or environment variables.")

# non-blocking auto-refresh
if autorefresh:
    st_autorefresh(interval=60_000, key="autorefresh_minute")

# -------------------------
# Fetch ticker universe robustly
# -------------------------
@st.cache_data(ttl=24*3600)
def load_tickers():
    tickers = set()
    # 1) NASDAQ Trader official combined feed (includes many exchanges)
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[df["Test Issue"] == "N"]
        tickers.update(df["Symbol"].dropna().astype(str).tolist())
        st.write(f"✅ Loaded {len(df)} tickers from nasdaqtrader feed.")
    except Exception as e:
        st.warning(f"Could not load nasdaqtrader feed: {e}")

    # 2) S&P 500 mirror (datahub)
    try:
        sp_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        sp = pd.read_csv(sp_url)
        tickers.update(sp["Symbol"].dropna().astype(str).tolist())
        st.write(f"✅ Loaded {len(sp)} S&P 500 tickers.")
    except Exception as e:
        st.warning(f"Could not load S&P 500 list: {e}")

    # 3) As a fallback, include a small set so app never returns zero
    if not tickers:
        st.error("No remote tickers available. Falling back to a small sample list.")
        tickers.update(["AAPL","MSFT","AMZN","GOOGL","TSLA","NVDA","META","JPM","DIS","NFLX"])

    tickers = sorted([t.upper().strip() for t in tickers if isinstance(t, str) and t.strip() != ""])
    return tickers

all_tickers = load_tickers()
st.sidebar.write(f"Tickers available: {len(all_tickers)}")
if limit_tickers and limit_tickers > 0:
    scan_tickers = all_tickers[:limit_tickers]
else:
    scan_tickers = all_tickers

# -------------------------
# Helper: news fetch for today
# -------------------------
def fetch_news_count_today(ticker, max_items=10):
    api_key = os.environ.get("NEWS_API_KEY") or st.secrets.get("NEWS_API_KEY", None) if "st" in globals() and hasattr(st, "secrets") else None
    if not api_key:
        return 0, []
    today = datetime.datetime.utcnow().date().isoformat()
    url = ("https://newsapi.org/v2/everything?"
           f"q={ticker}&from={today}&to={today}&pageSize={max_items}&apiKey={api_key}")
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        js = r.json()
        arts = js.get("articles", []) or []
        titles = [a.get("title","") for a in arts]
        return len(titles), titles
    except Exception:
        return 0, []

# -------------------------
# Screening: single ticker (intraday)
# -------------------------
def evaluate_ticker_intraday(ticker, vol_mult_min, demand_min_pct, news_min_count, max_float_m):
    """
    Returns True if ticker meets ALL criteria, else False.
    Uses intraday 1m bars when available; falls back to 5m or 1d if needed.
    """
    try:
        # prefer 1m intraday data for today
        intraday = None
        try:
            intraday = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
        except Exception:
            intraday = None

        if intraday is None or intraday.empty:
            # try 5m
            try:
                intraday = yf.download(ticker, period="1d", interval="5m", progress=False, threads=False)
            except Exception:
                intraday = None

        if intraday is None or intraday.empty:
            # no intraday data available — fail the strict intraday checks
            return False, None

        intraday = intraday.dropna(subset=["Open","Close","Volume"])
        if intraday.empty:
            return False, None

        # compute avg interval volume so far (exclude last incomplete interval)
        if len(intraday) >= 2:
            avg_so_far = intraday["Volume"][:-1].mean() if len(intraday) > 1 else intraday["Volume"].mean()
            cur_interval_vol = intraday["Volume"].iloc[-1]
            volume_ratio = (cur_interval_vol / avg_so_far) if avg_so_far and avg_so_far > 0 else 0
        else:
            return False, None

        # intraday demand from today's open
        today_open = intraday["Open"].iloc[0]
        latest_price = intraday["Close"].iloc[-1]
        demand_pct = ((latest_price - today_open) / today_open) * 100 if today_open != 0 else 0

        # float shares (may be in info)
        info = {}
        try:
            t = yf.Ticker(ticker)
            info = t.get_info() if hasattr(t, "get_info") else t.info
        except Exception:
            info = {}
        float_shares = info.get("floatShares") or info.get("float") or 0
        # some providers give float as integer, sometimes None

        # news today
        news_count, news_titles = fetch_news_count_today(ticker, max_items=5)

        # apply criteria (minimum thresholds)
        meets = (
            (volume_ratio >= vol_mult_min)
            and (demand_pct >= demand_min_pct)
            and (news_count >= news_min_count)
            and (float_shares is not None and float_shares > 0 and (float_shares <= max_float_m * 1_000_000))
        )

        result_meta = {
            "volume_ratio": volume_ratio,
            "demand_pct": demand_pct,
            "news_count": news_count,
            "news_titles": news_titles,
            "float_shares": float_shares
        }
        return meets, result_meta
    except Exception:
        return False, None

# -------------------------
# Async batch runner (uses thread executor)
# -------------------------
async def run_batches(ticker_list, vol_mult_min, demand_min_pct, news_min_count, max_float_m, batch_size=50):
    loop = asyncio.get_event_loop()
    results = []
    progress = st.progress(0)
    total = len(ticker_list)
    batches = [ticker_list[i:i+batch_size] for i in range(0, total, batch_size)]
    processed = 0

    for bi, batch in enumerate(batches):
        # run evaluations in threads to avoid blocking event loop
        tasks = [loop.run_in_executor(None, evaluate_ticker_intraday, t, vol_mult_min, demand_min_pct, news_min_count, max_float_m) for t in batch]
        batch_results = await asyncio.gather(*tasks)
        # batch_results is a list of tuples (meets(bool), meta or None)
        for ticker, (meets, meta) in zip(batch, batch_results):
            if meets:
                results.append(ticker)
        processed += len(batch)
        progress.progress(min(1.0, processed/total))
    return results

# -------------------------
# Main: run scan on button
# -------------------------
st.write("Press **Run Screener** to scan tickers (async batched). This may take a few minutes for thousands of tickers.")
if st.button("Run Screener"):
    with st.spinner("Scanning tickers..."):
        # limit scan set earlier (scan_tickers)
        to_scan = scan_tickers
        matches = asyncio.run(run_batches(to_scan, vol_multiplier_min, demand_pct_min, news_min, max_float_millions, batch_size))
    if matches:
        st.success(f"{len(matches)} tickers meet ALL criteria:")
        st.write(", ".join(matches))
    else:
        st.warning("No tickers met the criteria on this run.")

# optional: show last-run settings
st.sidebar.markdown("---")
st.sidebar.write("Last run settings:")
st.sidebar.write(f"Vol × ≥ {vol_multiplier_min}, Demand% ≥ {demand_pct_min}, News ≥ {news_min}, Float ≤ {max_float_millions}M")
st.sidebar.write(f"Tickers scanned: {len(scan_tickers)} (batch {batch_size})")
