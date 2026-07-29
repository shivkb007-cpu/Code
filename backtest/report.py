"""CLI: run the backtest and print/save a report.

Examples:
  # real data (needs network access to Yahoo Finance; fallback signal, no torch/chronos needed)
  python -m backtest.report --start 2022-01-01 --end 2024-01-01

  # real data + real Chronos model (needs torch + chronos-forecasting installed,
  # and network access to download model weights from Hugging Face on first run)
  python -m backtest.report --start 2022-01-01 --end 2024-01-01 --use-chronos

  # offline smoke test with synthetic data (no network required, no signal edge implied)
  python -m backtest.report --synthetic --start 2022-01-01 --end 2024-01-01
"""

import argparse
import os

import pandas as pd

from .config import StrategyConfig
from .engine import run_backtest
from .metrics import compute_metrics
from .signals import load_signal

# engine.py needs `lookback_bars` (90) trading days of history before it can evaluate
# anything, so fetch this much extra calendar time before --start and trim the warmup
# back off afterward — otherwise a --start/--end window under ~5 months has nowhere
# for that lookback to come from and run_backtest raises "not enough overlapping history".
WARMUP_CALENDAR_DAYS = 170


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="1d", choices=["1d", "60m"])
    parser.add_argument("--use-chronos", action="store_true",
                         help="Use the real Chronos model instead of the offline fallback signal")
    parser.add_argument("--synthetic", action="store_true",
                         help="Use synthetic price data instead of Yahoo Finance (offline smoke test only)")
    parser.add_argument("--out-dir", default="backtest_out")
    args = parser.parse_args()

    config = StrategyConfig()
    requested_start = pd.Timestamp(args.start)
    fetch_start = (requested_start - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)).date().isoformat()

    if args.synthetic:
        from . import synthetic
        universe = synthetic.make_universe(config.watchlist, fetch_start, args.end)
        spy = synthetic.make_symbol("SPY", fetch_start, args.end, seed=1)
    else:
        from . import data
        print("Fetching historical data...")
        universe = data.fetch_universe(config.watchlist, fetch_start, args.end, args.interval)
        print("  Fetching SPY...")
        spy = data.fetch_symbol("SPY", fetch_start, args.end, args.interval)

    if args.interval == "60m":
        config.time_stop_bars = 2  # ~120 minutes
    else:
        config.time_stop_bars = 1

    print("Loading signal model..." if args.use_chronos else "Loading fallback signal...")
    signal = load_signal(args.use_chronos)

    print("Running backtest (this is the slow part — progress prints every 20 simulated trading days)...")
    result = run_backtest(universe, spy, config, signal)

    # drop the warmup-only portion so reported metrics reflect the requested window
    result.equity_curve = result.equity_curve[result.equity_curve.index >= requested_start]
    result.trades = [t for t in result.trades if t.entry_date >= requested_start]

    metrics = compute_metrics(result, bars_per_year=252 if args.interval == "1d" else 252 * 7)

    print(metrics.summary())

    os.makedirs(args.out_dir, exist_ok=True)
    result.trades_df().to_csv(os.path.join(args.out_dir, "trades.csv"), index=False)
    result.equity_curve.to_csv(os.path.join(args.out_dir, "equity_curve.csv"))
    print(f"Wrote trades.csv and equity_curve.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
