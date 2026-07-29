"""Backtest for the Dual Momentum bot: weekly rebalance into the top-N highest-
momentum tickers (N=1 matches the original single-winner design), gated by SPY's
200-day moving average and a monthly -5% circuit breaker, plus a daily stop-loss
independent of the weekly cadence.
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
STOP_LOSS_PCT          = 0.15

MIN_LOOKBACK_BARS = 200
WARMUP_CALENDAR_DAYS = 320


def momentum_score(closes: np.ndarray) -> float:
    current = closes[-1]
    ret_1m = current / closes[-22] - 1 if len(closes) >= 22 else 0.0
    ret_3m = current / closes[-64] - 1 if len(closes) >= 64 else 0.0
    ret_6m = current / closes[-127] - 1 if len(closes) >= 127 else 0.0
    return ONE_MONTH_WEIGHT * ret_1m + THREE_MONTH_WEIGHT * ret_3m + SIX_MONTH_WEIGHT * ret_6m


def run_backtest(price_data: dict, spy_data: pd.DataFrame, top_n: int = 1,
                  start_cash: float = 100_000.0) -> BacktestResult:
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
    positions: dict = {}  # symbol -> {"qty", "entry_price", "entry_bar"}
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

        if (i - MIN_LOOKBACK_BARS) % 250 == 0:
            print(f"  ...{i - MIN_LOOKBACK_BARS}/{n - MIN_LOOKBACK_BARS} trading days processed ({date.date()})")

        # Daily stop-loss, independent of the weekly rebalance — a violent single-name
        # move shouldn't have to wait until the next Monday to get cut.
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            day_low = float(bars[symbol]["Low"].iloc[i])
            stop_price = pos["entry_price"] * (1 - STOP_LOSS_PCT)
            if day_low <= stop_price:
                exit_price = stop_price * (1 - slip)
                cash += pos["qty"] * exit_price
                trades.append(Trade(symbol, common_index[pos["entry_bar"]], pos["entry_price"],
                                     date, exit_price, pos["qty"], "stop_loss"))
                del positions[symbol]

        prior_close_equity = cash + sum(
            p["qty"] * float(bars[s]["Close"].iloc[i - 1]) for s, p in positions.items()
        )

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
            target_symbols = set()

            if monthly_return <= MONTHLY_DRAWDOWN_LIMIT:
                circuit_breaker_month = month_key
            elif circuit_breaker_month == month_key:
                pass  # stay in cash for the rest of this month
            else:
                spy_closes = spy["Close"].iloc[:i].values.astype(float)
                spy_above = len(spy_closes) >= 200 and spy_closes[-1] > spy_closes[-200:].mean()
                if spy_above:
                    scores = []
                    for symbol in symbols:
                        closes = bars[symbol]["Close"].iloc[:i].values.astype(float)
                        if len(closes) < 127:
                            continue
                        score = momentum_score(closes)
                        if score > 0:
                            scores.append((symbol, score))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    target_symbols = {s for s, _ in scores[:top_n]}

            currently_held = set(positions.keys())
            to_sell = currently_held - target_symbols
            to_buy = target_symbols - currently_held

            for symbol in to_sell:
                pos = positions[symbol]
                exit_price = float(bars[symbol]["Open"].iloc[i]) * (1 - slip)
                cash += pos["qty"] * exit_price
                trades.append(Trade(symbol, common_index[pos["entry_bar"]], pos["entry_price"],
                                     date, exit_price, pos["qty"], "rebalance_exit"))
                del positions[symbol]

            if to_buy:
                per_slot = (cash * INVEST_PCT) / max(1, len(target_symbols))
                for symbol in to_buy:
                    entry_price = float(bars[symbol]["Open"].iloc[i]) * (1 + slip)
                    qty = max(1, int(per_slot / entry_price))
                    cost = qty * entry_price
                    if cost <= cash:
                        cash -= cost
                        positions[symbol] = {"qty": qty, "entry_price": entry_price, "entry_bar": i}

        equity = cash + sum(p["qty"] * float(bars[s]["Close"].iloc[i]) for s, p in positions.items())
        equity_curve.append(equity)
        equity_dates.append(date)

    last_i = n - 1
    for symbol, pos in list(positions.items()):
        exit_price = float(bars[symbol]["Close"].iloc[last_i]) * (1 - slip)
        cash += pos["qty"] * exit_price
        trades.append(Trade(symbol, common_index[pos["entry_bar"]], pos["entry_price"],
                             common_index[last_i], exit_price, pos["qty"], "end_of_backtest"))
    positions.clear()
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
    parser.add_argument("--top-n", type=int, default=1)
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

    result = run_backtest(universe, spy, top_n=args.top_n)
    result.equity_curve = result.equity_curve[result.equity_curve.index >= requested_start]
    result.trades = [t for t in result.trades if t.entry_date >= requested_start]

    metrics = compute_metrics(result, bars_per_year=252)
    print(metrics.summary())

    out_dir = f"{args.out_dir}_top{args.top_n}"
    os.makedirs(out_dir, exist_ok=True)
    result.trades_df().to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    result.equity_curve.to_csv(os.path.join(out_dir, "equity_curve.csv"))
    print(f"Wrote trades.csv and equity_curve.csv to {out_dir}/")


if __name__ == "__main__":
    main()
