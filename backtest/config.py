"""Strategy parameters, mirrored from the live bot so backtest and live logic stay comparable."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategyConfig:
    watchlist: List[str] = field(default_factory=lambda: [
        "AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT", "AMZN", "GOOGL", "PLTR", "COIN",
    ])

    max_positions: int = 3
    position_pct: float = 0.05

    profit_target: float = 0.02
    stop_loss: float = 0.01

    # Original bot's time stop is 120 minutes on a live, minute-resolution clock.
    # This backtest operates on discrete bars (daily or hourly); time_stop_bars
    # is the number of bars held (with no profit) before a time-stop exit fires.
    # For interval="60m", 2 corresponds to the original 120-minute stop.
    # For interval="1d", 1 is the closest daily-bar equivalent.
    time_stop_bars: int = 2

    spy_drawdown_gate: float = -0.005  # block new entries if SPY day change <= this
    confidence_1d: float = 0.65
    confidence_3d: float = 0.60

    starting_cash: float = 100_000.0

    # Cost model — Alpaca is commission-free, so the default commission is 0.
    # Slippage approximates spread + market-order impact; tune per symbol liquidity.
    commission_bps: float = 0.0
    slippage_bps: float = 5.0

    # Chronos forecast horizon, in bars, used for the 1d/3d confidence checks.
    forecast_horizon: int = 5
    forecast_samples: int = 100
    lookback_bars: int = 90
