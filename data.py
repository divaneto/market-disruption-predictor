"""Historical price data fetching with on-disk caching."""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_SECONDS = 6 * 3600  # refetch at most every 6 hours


def fetch_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Return OHLCV history for a ticker, using a local CSV cache to avoid hammering Yahoo Finance."""
    safe_name = ticker.replace("=", "_").replace("/", "_")
    cache_path = CACHE_DIR / f"{safe_name}_{interval}.csv"

    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < CACHE_TTL_SECONDS:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(df) > 50:
            return df

    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        raise ValueError(f"No data returned for ticker '{ticker}' and no cache available.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(cache_path)
    return df
