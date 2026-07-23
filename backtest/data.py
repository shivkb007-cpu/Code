"""Historical OHLCV loading, with local caching so repeated backtests don't re-hit the network."""

import os
from typing import Dict, List

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _cache_path(symbol: str, interval: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_symbol(symbol: str, start: str, end: str, interval: str = "1d",
                  use_cache: bool = True) -> pd.DataFrame:
    """Download (or load cached) OHLCV for one symbol. Requires network access to Yahoo Finance."""
    cache_file = _cache_path(symbol, interval)
    if use_cache and os.path.exists(cache_file):
        cached = pd.read_parquet(cache_file)
        # only trust the cache if it actually spans the requested range — otherwise
        # a narrower prior run's cache silently starves a wider later request
        if not cached.empty and cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            return cached.loc[(cached.index >= start) & (cached.index <= end)]

    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({interval}, {start}..{end})")
    df = _flatten_columns(df)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_file)
    return df


def fetch_universe(symbols: List[str], start: str, end: str, interval: str = "1d",
                    use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for a list of symbols. Raises if any symbol fails rather than
    silently dropping it — a strategy backtest should not quietly run on a smaller
    universe than requested."""
    out = {}
    for symbol in symbols:
        out[symbol] = fetch_symbol(symbol, start, end, interval, use_cache)
    return out
