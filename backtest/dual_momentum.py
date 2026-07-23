"""Backtest for the Dual Momentum bot: weekly rebalance into the single highest-
momentum ticker, gated by SPY's 200-day moving average and a monthly -5% circuit
breaker. Mirrors live_bot/dual_momentum_bot.py's logic exactly (same weights, same
lookback windows, same thresholds) so this is a faithful check, not an approximation.

Usage:
    python -m backtest.dual_momentum --start 2018-01-01 --end 2026-01-01
    python -m backtest.dual_momentum --synthetic --start 2020-01-01 --end 2022-01-01
"""

import argparse
import os

import numpy as np
import pandas as pd

from .engine import Trade, BacktestResult
from .metrics import compute_metrics

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMD", "AMZN", "COIN", "PLTR", "MSTR"]

ONE_MONTH_WEIGHT   = 2.0
THREE_MONTH_WEIGHT = 1.5
SIX_MONTH_WEIGHT   = 0.5

MONTHLY_DRAWDOWN_LIMIT = -0.05
INVEST_PCT             = 0.95
SLIPPAGE_BPS           = 10

MIN_LOOKBACK_BARS = 200  # 200-day SPY MA is the binding constraint

# 200 trading days plus the 6-month momentum window need real calendar room before
# --start, or the first many weeks have nothing to evaluate against.
WARMUP_CALENDAR_DAYS = 320


def momentum_score(closes: np.ndarray) -> float:
    current = closes[-1]
    ret_1m = current / closes[-22] - 1 if len(closes) >= 22 else 0.0
    ret_3m = current / closes[-64] - 1 if len(closes) >= 64 else 0.0
    ret_6m = current / closes[-127] - 1 if len(closes) >= 127 else 0.0
    return ONE_MONTH_WEIGHT * ret_1m + THREE_MONTH_WEIGHT * ret_3m + SIX_MONTH_WEIGHT * ret_6m


def run_backtest(price_data: dict, spy_data: pd.DataFrame, start_cash: float = 100_000.0) -> BacktestResult:
    symbols = list(price_data.keys())
    common_index = spy_data.index
    for df in price_data.values():
        common_index = common_index.intersection(df.index)
    common_index = common_index.sort_values()

    if len(common_index) <= MIN_LOOKBACK_BARS + 2:
        raise ValueError("Not enough overlapping history for the 200-day lookback")

    bars = {s: price_data[s].loc[common_index] for s in symbols}
    spy = spy_data.loc[common_index]

    cash = start_cash
    held = None  # {"symbol", "qty", "entry_price", "entry_bar"}
    trades = []
    equity_curve = []
    equity_dates = []

    month_start_equity = None
    current_month = None
    circuit_breaker_month = None
    last_rebalance_week = None

    slip = SLIPPAGE_BPS / 10_000
    n = len(common_index)

    for i in range(n):
        date = common_index[i]
        if i < MIN_LOOKBACK_BARS:
            equity_curve.append(cash)
            equity_dates.append(date)
            continue

        prior_close_equity = cash + (held["qty"] * float(bars[held["symbol"]]["Close"].iloc[i - 1]) if held else 0.0)

        month_key = (date.year, date.month)
        if current_month != month_key:
            current_month = month_key
            month_start_equity = prior_close_equity
            if circuit_breaker_month is not None and circuit_breaker_month != month_key:
                circuit_breaker_month = None

        iso_week = date.isocalendar()[:2]
        if last_rebalance_week != iso_week:
            last_rebalance_week = iso_week

            monthly_return = (prior_close_equity - month_start_equity) / month_start_equity if month_start_equity else 0.0
            do_liquidate = False
            new_winner = None

            if monthly_return <= MONTHLY_DRAWDOWN_LIMIT:
                circuit_breaker_month = month_key
                do_liquidate = held is not None
            elif circuit_breaker_month == month_key:
                do_liquidate = held is not None
            else:
                spy_closes = spy["Close"].iloc[:i].values.astype(float)
                spy_above = len(spy_closes) >= 200 and spy_closes[-1] > spy_closes[-200:].mean()
                if not spy_above:
                    do_liquidate = held is not None
                else:
                    scores = []
                    for symbol in symbols:
                        closes = bars[symbol]["Close"].iloc[:i].values.astype(float)
                        if len(closes) < 127:
                            continue
                        scores.append((symbol, momentum_score(closes)))
                    if scores:
                        winner_symbol, winner_score = max(scores, key=lambda x: x[1])
                        if winner_score > 0:
                            new_winner = winner_symbol
                        else:
                            do_liquidate = held is not None
                    else:
                        do_liquidate = held is not None

            if held is not None and (do_liquidate or (new_winner and new_winner != held["symbol"])):
                exit_price = float(bars[held["symbol"]]["Open"].iloc[i]) * (1 - slip)
                cash += held["qty"] * exit_price
                trades.append(Trade(held["symbol"], common_index[held["entry_bar"]], held["entry_price"],
                                     date, exit_price, held["qty"], "rebalance_exit"))
                held = None

            if new_winner and held is None:
                entry_price = float(bars[new_winner]["Open"].iloc[i]) * (1 + slip)
                qty = max(1, int((cash * INVEST_PCT) / entry_price))
                cost = qty * entry_price
                if cost <= cash:
                    cash -= cost
                    held = {"symbol": new_winner, "qty": qty, "entry_price": entry_price, "entry_bar": i}

        equity = cash + (held["qty"] * float(bars[held["symbol"]]["Close"].iloc[i]) if held else 0.0)
        equity_curve.append(equity)
        equity_dates.append(date)

    if held is not None:
        last_i = n - 1
        exit_price = float(bars[held["symbol"]]["Close"].iloc[last_i]) * (1 - slip)
        cash += held["qty"] * exit_price
        trades.append(Trade(held["symbol"], common_index[held["entry_bar"]], held["entry_price"],
                             common_index[last_i], exit_price, held["qty"], "end_of_backtest"))
        if equity_curve:
            equity_curve[-1] = cash

    return BacktestResult(
        equity_curve=pd.Series(equity_curve, index=pd.Index(equity_dates, name="date"), name="equity"),
        trades=trades,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out-dir", default="backtest_out_dual_momentum")
    args = parser.parse_args()

    requested_start = pd.Timestamp(args.start)
    fetch_start = (requested_start - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)).date().isoformat()

    if args.synthetic:
        from . import synthetic
        universe = synthetic.make_universe(TICKERS, fetch_start, args.end)
        spy = synthetic.make_symbol("SPY", fetch_start, args.end, seed=1)
    else:
        from . import data
        universe = data.fetch_universe(TICKERS, fetch_start, args.end)
        spy = data.fetch_symbol("SPY", fetch_start, args.end)

    result = run_backtest(universe, spy)
    result.equity_curve = result.equity_curve[result.equity_curve.index >= requested_start]
    result.trades = [t for t in result.trades if t.entry_date >= requested_start]

    metrics = compute_metrics(result, bars_per_year=252)
    print(metrics.summary())

    os.makedirs(args.out_dir, exist_ok=True)
    result.trades_df().to_csv(os.path.join(args.out_dir, "trades.csv"), index=False)
    result.equity_curve.to_csv(os.path.join(args.out_dir, "equity_curve.csv"))
    print(f"Wrote trades.csv and equity_curve.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
