# Backtest harness for the Kronos/Chronos day-trading bot

Measures whether the live bot's entry logic (Chronos confidence + SPY filter + day-change
filter) has any real edge, before trusting it with money — paper or otherwise.

## Why this exists

The live bot (`level4_kronos_bot.py`) went straight to live paper trading with no
historical validation. This harness replays the same entry/exit rules against
historical prices with a cost model, so you get win rate, expectancy, drawdown, and
Sharpe before tuning anything further.

## Install

```
pip install -r backtest/requirements.txt
```

`torch` + `chronos-forecasting` are only required if you pass `--use-chronos`; without
it, a lightweight momentum-based fallback signal is used instead (see "Limitations").

## Run

```bash
# offline smoke test, synthetic random-walk data, no network needed, no real signal
python -m backtest.report --synthetic --start 2022-01-01 --end 2024-01-01

# real historical data (Yahoo Finance) with the fallback heuristic signal
python -m backtest.report --start 2022-01-01 --end 2024-01-01

# real historical data + the actual Chronos model (downloads weights from
# Hugging Face on first run)
python -m backtest.report --start 2022-01-01 --end 2024-01-01 --use-chronos

# hourly bars, closer to the live bot's 120-minute time stop (yfinance limits
# 60m history to roughly the trailing 2 years)
python -m backtest.report --start 2024-01-01 --end 2026-01-01 --interval 60m --use-chronos
```

Each run prints a summary and writes `trades.csv` + `equity_curve.csv` to `--out-dir`
(default `backtest_out/`).

## This sandbox specifically

This Claude Code session's network policy blocks Yahoo Finance, Alpaca, and Finnhub
directly (confirmed via `curl` — all three time out / get rejected at the proxy). So
`--synthetic` is the only mode that runs *here*. Run the real-data commands above on a
machine or environment with normal internet access.

## Limitations vs. the live bot — read before trusting results

- **Bar granularity, not a continuous clock.** The engine walks day-by-day (or bar-by-bar
  in `60m` mode), not minute-by-minute. Profit target / stop loss are detected by
  checking each bar's High/Low against the trigger price, not the actual intraday path,
  so fills are an approximation, not a tick-accurate replay.
- **No historical news filter.** The live bot skips entries when Finnhub shows severe
  headlines; this backtest has no historical news feed wired in, so it will take some
  entries the live bot would have blocked. Real performance is likely somewhat worse
  than what you see here for that reason alone.
- **No "wait 15 minutes after open" gate** — no equivalent at daily/hourly granularity.
- **The fallback signal (`FallbackSignal` in `signals.py`) is not a trading signal.**
  It's a deterministic momentum stand-in used only so the harness can be exercised
  without `torch`/model access. Only `--use-chronos` results say anything about whether
  the real model has edge. Don't read fallback-signal profitability as validation of
  the strategy — it's smoke-testing the code, not the idea.
- **Position sizing** uses total portfolio value (matching the live bot), not available
  cash, so a rejected order due to insufficient buying power is possible in principle;
  the backtest instead just skips the fill if `cost > cash`.
- **Costs are a flat bps assumption** (`slippage_bps`, `commission_bps` in `config.py`),
  not a real order-book/liquidity model. Tighten or loosen these to stress-test how
  sensitive the strategy's edge is to execution quality — this bot's 1%/2% stop/target
  is tight enough that a few bps of slippage matters a lot.

## What to look at

- **Win rate vs. avg win/loss** — expectancy is what matters, not win rate alone.
- **Sharpe and max drawdown** across at least one down-market window, not just a bull run.
- **Sensitivity to `slippage_bps`** — if profitability disappears at realistic slippage,
  the live version will lose money even if the raw signal has some directional skill.
- **`--use-chronos` vs. the fallback signal**, same date range — if Chronos doesn't
  meaningfully beat the momentum fallback, the model isn't adding value over something
  far simpler and cheaper to run.
