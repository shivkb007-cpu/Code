"""Event-driven, walk-forward backtest of the live bot's entry/exit rules.

Known simplifications vs. the live bot (see README for detail):
  - Operates on discrete bars (daily or hourly), not a continuous 1-minute clock.
    Profit target / stop loss are detected via each bar's High/Low, not tick-by-tick,
    so a bar that touches both levels is resolved conservatively (stop checked first).
  - No historical news feed is wired in, so the live bot's has_severe_news() filter
    is not modeled here. Backtest results are therefore optimistic relative to live
    in that one respect.
  - The live bot's "wait 15 minutes after open" gate has no equivalent at daily/hourly
    granularity and is omitted.

Decisions are made using data available through a bar's close and executed at the
*next* bar's open — the engine never lets a signal computed on bar i affect the
price paid on bar i.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    qty: int
    reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def pnl_pct(self) -> float:
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade] = field(default_factory=list)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "symbol": t.symbol, "entry_date": t.entry_date, "entry_price": t.entry_price,
            "exit_date": t.exit_date, "exit_price": t.exit_price, "qty": t.qty,
            "reason": t.reason, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
        } for t in self.trades])


def run_backtest(price_data: Dict[str, pd.DataFrame], spy_data: pd.DataFrame,
                  config: StrategyConfig, signal) -> BacktestResult:
    symbols = list(price_data.keys())

    common_index = spy_data.index
    for df in price_data.values():
        common_index = common_index.intersection(df.index)
    common_index = common_index.sort_values()

    if len(common_index) <= config.lookback_bars + 2:
        raise ValueError("Not enough overlapping history for the requested lookback")

    spy = spy_data.loc[common_index]
    bars = {s: price_data[s].loc[common_index] for s in symbols}

    cash = config.starting_cash
    positions: Dict[str, dict] = {}
    trades: List[Trade] = []
    equity_curve = []
    equity_dates = []
    pending_entry: Optional[str] = None

    slip = config.slippage_bps / 10_000
    comm = config.commission_bps / 10_000

    n = len(common_index)
    start_i = config.lookback_bars

    for i in range(start_i, n):
        date = common_index[i]

        # The Chronos calls below are the slow part (real model inference per
        # candidate symbol per day) and can silently run for a long time with
        # zero output otherwise, which is indistinguishable from being stuck.
        if (i - start_i) % 20 == 0:
            print(f"  ...{i - start_i}/{n - start_i} trading days processed ({date.date()})")

        if pending_entry is not None and pending_entry not in positions:
            symbol = pending_entry
            open_price = float(bars[symbol]["Open"].iloc[i])
            fill_price = open_price * (1 + slip)
            portfolio_value = cash + sum(
                p["qty"] * float(bars[s]["Close"].iloc[i - 1]) for s, p in positions.items()
            )
            qty = max(1, int(portfolio_value * config.position_pct / fill_price))
            cost = qty * fill_price * (1 + comm)
            if cost <= cash:
                cash -= cost
                positions[symbol] = {"qty": qty, "entry_price": fill_price, "entry_bar": i}
        pending_entry = None

        for symbol in list(positions.keys()):
            pos = positions[symbol]
            row = bars[symbol].iloc[i]
            entry_price = pos["entry_price"]
            bars_held = i - pos["entry_bar"]
            stop_price = entry_price * (1 - config.stop_loss)
            target_price = entry_price * (1 + config.profit_target)

            exit_price = None
            reason = None
            if row["Low"] <= stop_price:
                exit_price = stop_price * (1 - slip)
                reason = "stop_loss"
            elif row["High"] >= target_price:
                exit_price = target_price * (1 - slip)
                reason = "profit_target"
            elif bars_held >= config.time_stop_bars and row["Close"] <= entry_price:
                exit_price = float(row["Close"]) * (1 - slip)
                reason = "time_stop"

            if exit_price is not None:
                proceeds = pos["qty"] * exit_price * (1 - comm)
                cash += proceeds
                trades.append(Trade(symbol, common_index[pos["entry_bar"]], entry_price,
                                     date, exit_price, pos["qty"], reason))
                del positions[symbol]

        equity = cash + sum(p["qty"] * float(bars[s]["Close"].iloc[i]) for s, p in positions.items())
        equity_curve.append(equity)
        equity_dates.append(date)

        if i < n - 1 and len(positions) < config.max_positions:
            spy_change = float(spy["Close"].iloc[i]) / float(spy["Close"].iloc[i - 1]) - 1
            if spy_change > config.spy_drawdown_gate:
                candidates = []
                for symbol in symbols:
                    if symbol in positions:
                        continue
                    closes = bars[symbol]["Close"].iloc[: i + 1].values.astype(float)
                    day_change = closes[-1] / closes[-2] - 1
                    if day_change <= 0:
                        continue
                    conf = signal.get_confidence(closes, config.forecast_horizon,
                                                  config.forecast_samples)
                    if conf is None:
                        continue
                    conf_1d, conf_3d = conf
                    if conf_1d >= config.confidence_1d and conf_3d >= config.confidence_3d:
                        candidates.append((symbol, day_change, conf_1d + conf_3d))
                if candidates:
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    pending_entry = candidates[0][0]

    # liquidate anything still open at the end of the backtest window
    last_i = n - 1
    last_date = common_index[last_i]
    for symbol, pos in list(positions.items()):
        exit_price = float(bars[symbol]["Close"].iloc[last_i]) * (1 - slip)
        cash += pos["qty"] * exit_price * (1 - comm)
        trades.append(Trade(symbol, common_index[pos["entry_bar"]], pos["entry_price"],
                             last_date, exit_price, pos["qty"], "end_of_backtest"))
    positions.clear()
    if equity_curve:
        equity_curve[-1] = cash

    return BacktestResult(
        equity_curve=pd.Series(equity_curve, index=pd.Index(equity_dates, name="date"), name="equity"),
        trades=trades,
    )
