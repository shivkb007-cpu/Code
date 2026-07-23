"""Synthetic OHLCV generator — for exercising the backtest engine's mechanics only.

This sandbox cannot reach Yahoo Finance, Alpaca, or Finnhub, so this module produces
fake-but-plausible price series purely to smoke-test that data.py/signals.py/engine.py
wire together correctly. It has no bearing on whether the real strategy is profitable —
never treat results run on this data as evidence about the actual watchlist.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def make_symbol(symbol: str, start: str, end: str, seed: int, daily_vol: float = 0.02,
                 drift: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rets = rng.normal(drift, daily_vol, size=n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, daily_vol / 2, size=n)))
    low = close * (1 - np.abs(rng.normal(0, daily_vol / 2, size=n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                        "Volume": volume}, index=dates)
    return df


def make_universe(symbols: List[str], start: str, end: str, seed: int = 42) -> Dict[str, pd.DataFrame]:
    out = {}
    for i, symbol in enumerate(symbols):
        out[symbol] = make_symbol(symbol, start, end, seed=seed + i)
    return out
