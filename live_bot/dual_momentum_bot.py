"""
Dual Momentum Risk Managed Bot
- Every week: calculate 1m, 3m, 6m momentum for 10 stocks, on the first trading
  day of the week (handles a Monday holiday by rebalancing the next open day)
- Invest 95% in the highest scoring stock if score is positive
- SPY 200 day moving average filter - cash when SPY below 200 day moving average
- 5% monthly circuit breaker
- Alpaca paper trading

Secrets are read from environment variables, not hardcoded - set them once in your
shell before running this. If any are missing the script fails immediately with a
clear message instead of running with a blank/broken client.
"""

import csv
import json
import os
import time
import datetime
from datetime import timezone

import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import pytz


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


ALPACA_KEY    = require_env("ALPACA_KEY")
ALPACA_SECRET = require_env("ALPACA_SECRET")
ALPACA_URL    = os.environ.get("ALPACA_URL", "https://paper-api.alpaca.markets")

api     = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version="v2")
EASTERN = pytz.timezone("US/Eastern")

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMD", "AMZN", "COIN", "PLTR", "MSTR"]

ONE_MONTH_WEIGHT   = 2.0
THREE_MONTH_WEIGHT = 1.5
SIX_MONTH_WEIGHT   = 0.5

MONTHLY_DRAWDOWN_LIMIT = -0.05  # 5% circuit breaker
INVEST_PCT             = 0.95

_HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH       = os.path.join(_HERE, "dual_momentum_state.json")
SIGNAL_LOG_PATH  = os.path.join(_HERE, "dual_momentum_signal_log.csv")
TRADE_LOG_PATH   = os.path.join(_HERE, "dual_momentum_trade_log.csv")


# ── Persistent state (survives restarts) ───────────────────────────────────────
# The original kept monthly_start_equity / circuit_breaker_month / current_holding /
# last_rebalance_week only in memory. A crash or restart mid-week lost all of it,
# which caused two real bugs: (1) current_holding resetting to None made the bot
# liquidate-then-immediately-rebuy the same stock it was already holding, burning
# a round-trip in fees/slippage for nothing, and (2) losing last_rebalance_week
# could trigger a second rebalance the same week after a restart. Persisting this
# to disk and reconciling current_holding against the real broker state on startup
# fixes both.
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "current_holding": None,
        "monthly_start_equity": None,
        "circuit_breaker_month": None,   # [year, month] or None
        "current_month": None,           # [year, month] or None
        "last_rebalance_week": None,     # [iso_year, iso_week] or None
    }


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def _append_csv(path, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_signal(ticker, ret_1m, ret_3m, ret_6m, score, chosen, reason):
    _append_csv(SIGNAL_LOG_PATH, {
        "time": datetime.datetime.now(timezone.utc).isoformat(),
        "ticker": ticker or "",
        "ret_1m": "" if ret_1m is None else f"{ret_1m:.4f}",
        "ret_3m": "" if ret_3m is None else f"{ret_3m:.4f}",
        "ret_6m": "" if ret_6m is None else f"{ret_6m:.4f}",
        "score": "" if score is None else f"{score:.4f}",
        "chosen": chosen,
        "reason": reason,
    })


def log_trade(symbol, side, qty, price, reason):
    _append_csv(TRADE_LOG_PATH, {
        "time": datetime.datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": f"{price:.4f}",
        "reason": reason,
    })


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_portfolio():
    account   = api.get_account()
    portfolio = float(account.portfolio_value)
    cash      = float(account.cash)
    pl        = portfolio - 100000
    return portfolio, cash, pl


def get_positions():
    return {p.symbol: p for p in api.list_positions()}


def liquidate_all(reason="rebalance"):
    positions = get_positions()
    for symbol, pos in positions.items():
        qty = int(float(pos.qty))
        if qty > 0:
            try:
                api.submit_order(symbol=symbol, qty=qty, side="sell",
                                  type="market", time_in_force="day")
                print(f"  SELL {qty} x {symbol}")
                log_trade(symbol, "sell", qty, float(pos.current_price or pos.avg_entry_price), reason)
            except Exception as e:
                print(f"  SELL failed {symbol}: {e}")


def get_spy_above_200ma():
    """Check if SPY is above its 200 day moving average.
    Fails closed (treats an error as "below MA" / go to cash) rather than failing
    open — a data outage shouldn't be silently treated as "market's fine"."""
    try:
        spy = yf.download("SPY", period="1y", interval="1d", progress=False)
        if spy.empty or len(spy) < 200:
            print("  SPY data unavailable - treating as below 200 day moving average")
            return False
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        close = spy["Close"]
        ma200 = float(close.rolling(200).mean().iloc[-1])
        current = float(close.iloc[-1])
        print(f"  SPY: ${current:.2f} | 200 day moving average: ${ma200:.2f} | {'ABOVE' if current > ma200 else 'BELOW'}")
        return current > ma200
    except Exception as e:
        print(f"  SPY check error ({e}) - treating as below 200 day moving average")
        return False


def get_momentum_score(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 130:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        current = float(closes.iloc[-1])

        ret_1m = (current / float(closes.iloc[-22]) - 1) if len(closes) >= 22 else 0
        ret_3m = (current / float(closes.iloc[-64]) - 1) if len(closes) >= 64 else 0
        ret_6m = (current / float(closes.iloc[-127]) - 1) if len(closes) >= 127 else 0

        score = (ONE_MONTH_WEIGHT * ret_1m +
                 THREE_MONTH_WEIGHT * ret_3m +
                 SIX_MONTH_WEIGHT * ret_6m)

        return {
            "ticker": ticker,
            "score": score,
            "1m": ret_1m,
            "3m": ret_3m,
            "6m": ret_6m,
            "price": current
        }
    except Exception as e:
        print(f"  Error calculating momentum for {ticker}: {e}")
        return None


def is_market_open():
    return api.get_clock().is_open


def rebalance(state):
    print("\n" + "=" * 55)
    print("  REBALANCING - Dual Momentum Strategy")
    print("=" * 55)

    portfolio, cash, pl = get_portfolio()
    print(f"  Portfolio: ${portfolio:,.2f} | Cash: ${cash:,.2f} | P&L: ${pl:+,.2f}")

    now = datetime.datetime.now(timezone.utc).astimezone(EASTERN)
    current_month = [now.year, now.month]

    if state["monthly_start_equity"] is None or state["current_month"] != current_month:
        state["current_month"] = current_month
        state["monthly_start_equity"] = portfolio
        if state["circuit_breaker_month"] is not None and state["circuit_breaker_month"] != current_month:
            state["circuit_breaker_month"] = None

    monthly_return = (portfolio - state["monthly_start_equity"]) / state["monthly_start_equity"]
    print(f"  Monthly return so far: {monthly_return:.2%}")

    if monthly_return <= MONTHLY_DRAWDOWN_LIMIT:
        print(f"  CIRCUIT BREAKER triggered at {monthly_return:.2%} - moving to cash until next month")
        log_signal(None, None, None, None, None, "", "circuit_breaker_triggered")
        liquidate_all("circuit_breaker")
        state["circuit_breaker_month"] = current_month
        state["current_holding"] = None
        save_state(state)
        return

    if state["circuit_breaker_month"] == current_month:
        print("  Circuit breaker active this month - holding cash")
        log_signal(None, None, None, None, None, "", "circuit_breaker_active")
        save_state(state)
        return

    if not get_spy_above_200ma():
        print("  SPY below 200 day moving average - moving to cash")
        log_signal(None, None, None, None, None, "", "spy_below_200ma")
        liquidate_all("spy_below_200ma")
        state["current_holding"] = None
        save_state(state)
        return

    print("\n  Calculating momentum scores...")
    scores = []
    for ticker in TICKERS:
        result = get_momentum_score(ticker)
        if result:
            print(f"  {ticker}: score={result['score']:+.4f} | 1m={result['1m']:+.2%} | 3m={result['3m']:+.2%} | 6m={result['6m']:+.2%}")
            scores.append(result)
            log_signal(ticker, result["1m"], result["3m"], result["6m"], result["score"], False, "evaluated")
        else:
            log_signal(ticker, None, None, None, None, False, "no_data")

    if not scores:
        print("  No momentum scores available - holding cash")
        liquidate_all("no_scores")
        state["current_holding"] = None
        save_state(state)
        return

    winner = max(scores, key=lambda x: x["score"])

    if winner["score"] <= 0:
        print(f"  Best score is negative ({winner['score']:+.4f}) - moving to cash")
        liquidate_all("negative_momentum")
        state["current_holding"] = None
        save_state(state)
        return

    print(f"\n  WINNER: {winner['ticker']} with score {winner['score']:+.4f}")
    log_signal(winner["ticker"], winner["1m"], winner["3m"], winner["6m"], winner["score"], True, "winner")

    if state["current_holding"] != winner["ticker"]:
        print(f"  Switching from {state['current_holding']} to {winner['ticker']}")
        liquidate_all("switch")
        time.sleep(2)

        _, cash, _ = get_portfolio()
        invest_amount = cash * INVEST_PCT
        price = winner["price"]
        qty = int(invest_amount / price)

        if qty > 0:
            try:
                api.submit_order(
                    symbol=winner["ticker"],
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                print(f"  BUY {qty} x {winner['ticker']} @ ~${price:.2f}")
                log_trade(winner["ticker"], "buy", qty, price, "winner")
                state["current_holding"] = winner["ticker"]
            except Exception as e:
                print(f"  BUY failed: {e}")
                # We already liquidated above, so we're genuinely in cash now —
                # leaving current_holding at its old value here (the original bug)
                # would make the bot think it still held the sold-off stock and
                # silently sit in cash forever, mistaking that for "no change needed".
                state["current_holding"] = None
    else:
        print(f"  No change - continuing to hold {state['current_holding']}")

    save_state(state)


def run():
    print("=" * 55)
    print("  Dual Momentum Risk Managed Bot")
    print("  Rebalances on the first trading day of each week")
    print("=" * 55)
    portfolio, cash, pl = get_portfolio()
    print(f"  Portfolio: ${portfolio:,.2f} | Cash: ${cash:,.2f} | P&L: ${pl:+,.2f}")

    state = load_state()

    # Reconcile against the real broker state on startup instead of trusting
    # whatever current_holding was last saved — if they've drifted apart (e.g.
    # a manual trade, or a crash between liquidate and buy), the broker is truth.
    positions = get_positions()
    live_symbols = list(positions.keys())
    if len(live_symbols) == 1:
        actual_holding = live_symbols[0]
    elif len(live_symbols) == 0:
        actual_holding = None
    else:
        # this bot is only ever supposed to hold one symbol at a time; more than
        # one means something outside the bot touched the account
        print(f"  WARNING: multiple positions found ({live_symbols}) - this bot expects at most one")
        actual_holding = None
    if actual_holding != state["current_holding"]:
        print(f"  Reconciling current_holding: state said {state['current_holding']!r}, broker says {actual_holding!r}")
        state["current_holding"] = actual_holding
        save_state(state)
    print()

    while True:
        now = datetime.datetime.now(timezone.utc).astimezone(EASTERN)
        current_week = list(now.isocalendar()[:2])  # [ISO year, ISO week number]

        if is_market_open():
            if state["last_rebalance_week"] != current_week:
                market_open    = now.replace(hour=9, minute=30, second=0, microsecond=0)
                rebalance_time = market_open + datetime.timedelta(minutes=30)
                if now >= rebalance_time:
                    rebalance(state)
                    state["last_rebalance_week"] = current_week
                    save_state(state)
                else:
                    mins = int((rebalance_time - now).total_seconds() / 60)
                    print(f"[{now.strftime('%H:%M')}] First trading day this week - waiting {mins} min until rebalance")
        else:
            portfolio, cash, pl = get_portfolio()
            print(f"[{now.strftime('%Y-%m-%d %H:%M')}] Holding: {state['current_holding'] or 'Cash'} | Portfolio: ${portfolio:,.2f} | P&L: ${pl:+,.2f}")

        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    run()
