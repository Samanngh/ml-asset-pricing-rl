import time
from pathlib import Path
import pandas as pd, yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CACHE = RAW / "yf_cache"
CACHE.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

START, END = "2000-01-01", None
PAUSE = 1.5
MAX_RETRIES = 6

# Start with just 3 tickers

TICKERS = {
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "SP500": "^GSPC",
    "NIKKEI225": "^N225",
    "CAC40": "^FCHI",
    "US30Y_FUT": "ZB=F",
    "US10Y_FUT": "ZN=F",   # 🔄 replaced CA10Y_FUT
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



def fetch_series(name, ticker):
    cache_file = CACHE / f"{name}.parquet"
    


    if cache_file.exists():
        dfc = pd.read_parquet(cache_file)
        if "ret" in dfc.columns:
            s = dfc["ret"]
        else:
            # fallback: old cache style (Series saved directly)
            s = dfc.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        s.name = "ret"
        return s

    backoff = 2.0
    for k in range(MAX_RETRIES):
        try:
            df = yf.download(
                ticker, start=START, end=END, auto_adjust=True,
                progress=False, threads=False, timeout=60
            )
            if df.empty:
                return None
            s = df["Close"].asfreq("B").ffill().pct_change().dropna()
            s.name = "ret"   # ensure column name is consistent
            pd.DataFrame(s).to_parquet(cache_file)
            return s

        except Exception as e:
            if k == MAX_RETRIES - 1:
                print(f"❌ {name} failed: {e}")
                return None
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 60)

def main():
    idx = pd.bdate_range(START, pd.Timestamp.today().normalize())
    wide = pd.DataFrame(index=idx); wide.index.name = "Date"
    for i, (name, t) in enumerate(TICKERS.items(), 1):
        print(f"[{i:02d}/{len(TICKERS)}] {name} ← {t}")
        s = fetch_series(name, t)
        if s is None:
            print(f"   ⚠️  Skipping {name}")
        else:
            wide[name] = s.reindex(idx).fillna(0.0)
        time.sleep(PAUSE)
    if len(wide.columns) == 0:
        print("⚠️ No assets fetched. Try again later or switch to CSV ingestion.")
    wide.reset_index().to_csv(PROC/"daily_returns_wide.csv", index=False)
    wide.stack().rename("ret").rename_axis(["Date","asset"]).reset_index() \
        .to_parquet(PROC/"daily_returns_long.parquet", index=False)
    print("✅ saved:", PROC/"daily_returns_wide.csv")
    print("✅ saved:", PROC/"daily_returns_long.parquet")
    print("Columns:", list(wide.columns))

if __name__ == "__main__":
    main()
