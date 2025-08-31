from pathlib import Path
import pandas as pd, numpy as np, yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

START, END = "2000-01-01", None

# Map course assets -> Yahoo tickers (adjust if any fail in your region)
TICKERS = {
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "SP500": "^GSPC",
    "NIKKEI225": "^N225",
    "CAC40": "^FCHI",          # index (close to CAC40 futures)
    "US30Y_FUT": "ZB=F",       # 30Y T-Bond futures
    "CA10Y_FUT": "CGB=F",      # Canada Gov Bond 10Y futures (if missing, download CSV manually)
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "COPPER": "HG=F",
    "BRENT": "BZ=F",
    "WTI": "CL=F",
    "NATGAS": "NG=F",
    "WHEAT": "ZW=F",
    "SOY_OIL": "ZL=F",
    "COFFEE_C": "KC=F",
}

def fetch_ret_daily(ticker):
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if df.empty: 
        return None
    s = df["Close"].asfreq("B").ffill().pct_change().dropna()
    return s

wide = pd.DataFrame(index=pd.bdate_range(START, pd.Timestamp.today()))
for name, t in TICKERS.items():
    s = fetch_ret_daily(t)
    if s is None:
        print(f"⚠️  Skipping {name}: {t} not available.")
        continue
    wide[name] = s

# Align and fill non-trading days with 0 to mirror the lecture style
wide = wide.loc[wide.index.min():wide.index.max()].fillna(0.0)
wide.index.name = "Date"

wide.reset_index().to_csv(PROC/"daily_returns_wide.csv", index=False)
wide.stack().rename("ret").rename_axis(["Date","asset"]).reset_index() \
    .to_parquet(PROC/"daily_returns_long.parquet", index=False)

print("✅ saved:", PROC/"daily_returns_wide.csv")
print("✅ saved:", PROC/"daily_returns_long.parquet")
