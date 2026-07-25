"""
Dual Momentum Risk Managed Bot (top-N variant)
- Every week: calculate 1m, 3m, 6m momentum for 10 stocks, on the first trading
  day of the week (handles a Monday holiday by rebalancing the next open day)
- Invest 95% split evenly across the top TOP_N positive-scoring stocks
- SPY 200 day moving average filter - cash when SPY below 200 day moving average
- 5% monthly circuit breaker
- A daily-equivalent stop-loss checked every 5-minute loop tick, independent of
  the weekly rebalance — backtesting showed the weekly-only check let a single
  concentrated position (MSTR, -41% over two weeks including the Aug 2024
  selloff) run uncaught until the next rebalance, which was the main driver of
  a -52% max drawdown. Splitting across top-2 names plus this stop-loss brought
  the backtested drawdown down to roughly -28%, at a cost of ~1pt of CAGR.
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
TOP_N                  = 2      # backtested sweet spot: best Sharpe/drawdown for only ~1pt less CAGR than top-1
STOP_LOSS_PCT          = 0.15

_HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH       = os.path.join(_HERE, "dual_momentum_state.json")
SIGNAL_LOG_PATH  = os.path.join(_HERE, "dual_momentum_signal_log.csv")
TRADE_LOG_PATH   = os.path.join(_HERE, "dual_momentum_trade_log.csv")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "holdings": {},                  # symbol -> {"entry_price": float, "entry_time": iso str}
        "monthly_start_equity": None,
        "circuit_breaker_month": None,    # [year, month] or None
        "current_month": None,            # [year, month] or None
        "last_rebalance_week": None,      # [iso_year, iso_week] or None
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


def get_live_price(symbol):
    try:
        return float(api.get_position(symbol).current_price)
    except Exception:
        try:
            return float(yf.Ticker(symbol).fast_info["last_price"])
        except Exception:
            return None


def sell_symbol(symbol, reason):
    positions = get_positions()
    if symbol not in positions:
        return
    qty = int(float(positions[symbol].qty))
    if qty <= 0:
        return
    try:
        api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day")
        print(f"  SELL {qty} x {symbol} - {reason}")
        log_trade(symbol, "sell", qty, float(positions[symbol].current_price or positions[symbol].avg_entry_price), reason)
    except Exception as e:
        print(f"  SELL failed {symbol}: {e}")


def liquidate_all(reason="rebalance"):
    for symbol in list(get_positions().keys()):
        sell_symbol(symbol, reason)


def check_stop_losses(state):
    """Runs on every loop tick (every 5 min), independent of the weekly rebalance —
    see module docstring for why the weekly-only check wasn't enough."""
    for symbol in list(state["holdings"].keys()):
        price = get_live_price(symbol)
        if not price:
            continue
        entry = state["holdings"][symbol]["entry_price"]
        change = (price - entry) / entry
        if change <= -STOP_LOSS_PCT:
            print(f"  {symbol} {change*100:.2f}% - STOP LOSS")
            sell_symbol(symbol, "stop_loss")
            del state["holdings"][symbol]
            save_state(state)


def get_spy_above_200ma():
    """Fails closed (treats an error as "below MA" / go to cash) rather than failing
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

        return {"ticker": ticker, "score": score, "1m": ret_1m, "3m": ret_3m, "6m": ret_6m, "price": current}
    except Exception as e:
        print(f"  Error calculating momentum for {ticker}: {e}")
        return None


def is_market_open():
    return api.get_clock().is_open


def rebalance(state):
    print("\n" + "=" * 55)
    print("  REBALANCING - Dual Momentum Strategy (top-%d)" % TOP_N)
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

    target_symbols = set()

    if monthly_return <= MONTHLY_DRAWDOWN_LIMIT:
        print(f"  CIRCUIT BREAKER triggered at {monthly_return:.2%} - moving to cash until next month")
        log_signal(None, None, None, None, None, "", "circuit_breaker_triggered")
        state["circuit_breaker_month"] = current_month
    elif state["circuit_breaker_month"] == current_month:
        print("  Circuit breaker active this month - holding cash")
        log_signal(None, None, None, None, None, "", "circuit_breaker_active")
    elif not get_spy_above_200ma():
        print("  SPY below 200 day moving average - moving to cash")
        log_signal(None, None, None, None, None, "", "spy_below_200ma")
    else:
        print("\n  Calculating momentum scores...")
        scores = []
        for ticker in TICKERS:
            result = get_momentum_score(ticker)
            if result:
                print(f"  {ticker}: score={result['score']:+.4f} | 1m={result['1m']:+.2%} | 3m={result['3m']:+.2%} | 6m={result['6m']:+.2%}")
                if result["score"] > 0:
                    scores.append(result)
                log_signal(ticker, result["1m"], result["3m"], result["6m"], result["score"], False, "evaluated")
            else:
                log_signal(ticker, None, None, None, None, False, "no_data")

        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[:TOP_N]
        for r in top:
            log_signal(r["ticker"], r["1m"], r["3m"], r["6m"], r["score"], True, "chosen")
        target_symbols = {r["ticker"] for r in top}
        if target_symbols:
            print(f"\n  TARGET: {sorted(target_symbols)}")
        else:
            print("\n  No positive-momentum tickers - holding cash")

    currently_held = set(state["holdings"].keys())
    to_sell = currently_held - target_symbols
    to_buy = target_symbols - currently_held

    for symbol in to_sell:
        sell_symbol(symbol, "rebalance")
        del state["holdings"][symbol]

    if to_buy:
        time.sleep(2)  # let the sells above settle before sizing the buys
        _, cash, _ = get_portfolio()
        per_slot = (cash * INVEST_PCT) / max(1, len(target_symbols))
        for symbol in to_buy:
            price = get_live_price(symbol)
            if not price:
                continue
            qty = max(1, int(per_slot / price))
            try:
                api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day")
                print(f"  BUY {qty} x {symbol} @ ~${price:.2f}")
                log_trade(symbol, "buy", qty, price, "chosen")
                state["holdings"][symbol] = {
                    "entry_price": price,
                    "entry_time": datetime.datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                print(f"  BUY failed {symbol}: {e}")

    if not to_sell and not to_buy:
        print(f"  No change - continuing to hold {sorted(state['holdings'].keys()) or 'cash'}")

    save_state(state)


def run():
    print("=" * 55)
    print("  Dual Momentum Risk Managed Bot (top-%d)" % TOP_N)
    print("  Rebalances on the first trading day of each week")
    print("=" * 55)
    portfolio, cash, pl = get_portfolio()
    print(f"  Portfolio: ${portfolio:,.2f} | Cash: ${cash:,.2f} | P&L: ${pl:+,.2f}")

    state = load_state()

    # Reconcile against the real broker state on startup instead of trusting
    # whatever holdings were last saved.
    positions = get_positions()
    if set(positions.keys()) != set(state["holdings"].keys()):
        print(f"  Reconciling holdings: state said {list(state['holdings'].keys())}, broker says {list(positions.keys())}")
        new_holdings = {}
        for symbol, pos in positions.items():
            if symbol in state["holdings"]:
                new_holdings[symbol] = state["holdings"][symbol]
            else:
                new_holdings[symbol] = {
                    "entry_price": float(pos.avg_entry_price),
                    "entry_time": datetime.datetime.now(timezone.utc).isoformat(),
                }
        state["holdings"] = new_holdings
        save_state(state)
    print()

    while True:
        try:
            now = datetime.datetime.now(timezone.utc).astimezone(EASTERN)

            if is_market_open():
                check_stop_losses(state)

                current_week = list(now.isocalendar()[:2])
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
                print(f"[{now.strftime('%Y-%m-%d %H:%M')}] Holding: {sorted(state['holdings'].keys()) or 'Cash'} | Portfolio: ${portfolio:,.2f} | P&L: ${pl:+,.2f}")
        except Exception as e:
            # A dropped connection to Alpaca or Yahoo Finance mid-request shouldn't
            # kill a process meant to run unattended for weeks — log it and retry
            # next cycle instead of crashing the whole bot.
            now = datetime.datetime.now(timezone.utc).astimezone(EASTERN)
            print(f"[{now.strftime('%H:%M')}] Unexpected error this cycle ({e}) - will retry next cycle")

        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    run()
