import pandas as pd
import requests
from io import StringIO

# ---------------------------
# 1️⃣ S&P 500 from Wikipedia
# ---------------------------
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(sp500_url)[0]
sp500_table['Symbol'] = sp500_table['Symbol'].str.strip()
sp500_table.drop_duplicates(subset='Symbol', inplace=True)
sp500_table.to_csv("sp500.csv", index=False)
print(f"S&P 500 tickers saved: {len(sp500_table)}")

# ---------------------------
# 2️⃣ NASDAQ tickers (from NASDAQ FTP)
# ---------------------------
nasdaq_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
nasdaq_txt = requests.get(nasdaq_url).text
nasdaq_df = pd.read_csv(StringIO(nasdaq_txt), sep="|")
nasdaq_df = nasdaq_df[nasdaq_df['Test Issue'] == 'N']  # remove test symbols
nasdaq_df['Symbol'] = nasdaq_df['Symbol'].str.strip()
nasdaq_df.drop_duplicates(subset='Symbol', inplace=True)
nasdaq_df.to_csv("nasdaq.csv", index=False)
print(f"NASDAQ tickers saved: {len(nasdaq_df)}")

# ---------------------------
# 3️⃣ NYSE tickers (from NASDAQ FTP)
# ---------------------------
nyse_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
nyse_txt = requests.get(nyse_url).text
nyse_df = pd.read_csv(StringIO(nyse_txt), sep="|")
nyse_df = nyse_df[nyse_df['Exchange'] == 'N']  # keep NYSE
nyse_df['ACT Symbol'] = nyse_df['ACT Symbol'].str.strip()
nyse_df.drop_duplicates(subset='ACT Symbol', inplace=True)
nyse_df.to_csv("nyse.csv", index=False)
print(f"NYSE tickers saved: {len(nyse_df)}")
