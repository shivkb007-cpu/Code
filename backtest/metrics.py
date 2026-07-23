"""Performance metrics computed from a BacktestResult."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestResult


@dataclass
class Metrics:
    start_equity: float
    end_equity: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy_pct: float
    avg_bars_held: float

    def summary(self) -> str:
        return (
            f"Equity:         ${self.start_equity:,.0f} -> ${self.end_equity:,.0f}\n"
            f"Total return:   {self.total_return:+.2%}\n"
            f"CAGR:           {self.cagr:+.2%}\n"
            f"Sharpe:         {self.sharpe:.2f}\n"
            f"Max drawdown:   {self.max_drawdown:.2%}\n"
            f"Trades:         {self.num_trades}\n"
            f"Win rate:       {self.win_rate:.1%}\n"
            f"Avg win:        {self.avg_win_pct:+.2%}\n"
            f"Avg loss:       {self.avg_loss_pct:+.2%}\n"
            f"Profit factor:  {self.profit_factor:.2f}\n"
            f"Expectancy:     {self.expectancy_pct:+.2%} per trade\n"
            f"Avg hold:       {self.avg_bars_held:.1f} bars\n"
        )


def compute_metrics(result: BacktestResult, bars_per_year: int = 252) -> Metrics:
    curve = result.equity_curve
    if len(curve) < 2:
        raise ValueError("Equity curve too short to compute metrics")

    start_equity = float(curve.iloc[0])
    end_equity = float(curve.iloc[-1])
    total_return = end_equity / start_equity - 1

    n_bars = len(curve)
    years = n_bars / bars_per_year
    cagr = (end_equity / start_equity) ** (1 / years) - 1 if years > 0 else 0.0

    bar_returns = curve.pct_change().dropna()
    sharpe = (bar_returns.mean() / bar_returns.std() * np.sqrt(bars_per_year)
              if bar_returns.std() > 0 else 0.0)

    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    max_drawdown = float(drawdown.min())

    trades_df = result.trades_df()
    num_trades = len(trades_df)
    if num_trades == 0:
        return Metrics(start_equity, end_equity, total_return, cagr, sharpe,
                        max_drawdown, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / num_trades
    avg_win_pct = float(wins["pnl_pct"].mean()) if len(wins) else 0.0
    avg_loss_pct = float(losses["pnl_pct"].mean()) if len(losses) else 0.0
    gross_profit = float(wins["pnl"].sum())
    gross_loss = float(-losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy_pct = float(trades_df["pnl_pct"].mean())

    entry_idx = curve.index.get_indexer(trades_df["entry_date"])
    exit_idx = curve.index.get_indexer(trades_df["exit_date"])
    avg_bars_held = float(np.mean(exit_idx - entry_idx))

    return Metrics(start_equity, end_equity, total_return, cagr, sharpe, max_drawdown,
                    num_trades, win_rate, avg_win_pct, avg_loss_pct, profit_factor,
                    expectancy_pct, avg_bars_held)
