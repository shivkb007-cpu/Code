"""Level 4 - Kronos AI day-trading bot (paper trading via Alpaca).

Secrets are read from environment variables, not hardcoded — set them once in your
shell (see live_bot/README.md) before running this. If any are missing the script
fails immediately with a clear message instead of running with a blank/broken client.
"""

import os
import time
import datetime
from datetime import timezone

import torch
import numpy as np
import pandas as pd
import yfinance as yf
import finnhub
import alpaca_trade_api as tradeapi
import pytz


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


ALPACA_KEY    = require_env("ALPACA_KEY")
ALPACA_SECRET = require_env("ALPACA_SECRET")
ALPACA_URL    = os.environ.get("ALPACA_URL", "https://paper-api.alpaca.markets")
FINNHUB_KEY   = require_env("FINNHUB_KEY")

api     = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version="v2")
fh      = finnhub.Client(api_key=FINNHUB_KEY)
EASTERN = pytz.timezone("US/Eastern")

WATCHLIST = ["AAPL", "NVDA", "TSLA", "AMD", "META", "MSFT", "AMZN", "GOOGL", "PLTR", "COIN"]

MAX_POSITIONS        = 3
POSITION_PCT         = 0.05
PROFIT_TARGET        = 0.02
STOP_LOSS            = 0.01
TIME_STOP_MINUTES    = 120
NO_ENTRY_BEFORE_MINS = 15
CONFIDENCE_1D        = 0.65
CONFIDENCE_3D        = 0.60
SLIPPAGE_BPS         = 15  # max slippage tolerated on limit orders, in basis points

print("Loading Kronos model...")
try:
    from chronos import ChronosPipeline
    kronos = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)
    print("Kronos loaded successfully")
    KRONOS_AVAILABLE = True
except Exception as e:
    print(f"Kronos not available: {e}")
    KRONOS_AVAILABLE = False


def _run_chronos_predict(context_tensor, prediction_length, num_samples):
    """chronos-forecasting renamed predict()'s first parameter across versions
    (context -> inputs), which is what broke this bot last time. Try the current
    signature first and fall back for older installs instead of hardcoding one."""
    try:
        return kronos.predict(inputs=context_tensor, prediction_length=prediction_length, num_samples=num_samples)
    except TypeError:
        return kronos.predict(context_tensor, prediction_length=prediction_length, num_samples=num_samples)


def is_market_open():
    return api.get_clock().is_open


def mins_since_open():
    now = datetime.datetime.now(timezone.utc).astimezone(EASTERN)
    return int((now - now.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60)


def mins_to_close():
    now = datetime.datetime.now(timezone.utc).astimezone(EASTERN)
    return int((now.replace(hour=16, minute=0, second=0, microsecond=0) - now).total_seconds() / 60)


def get_portfolio():
    a = api.get_account()
    return float(a.portfolio_value), float(a.cash), float(a.portfolio_value) - 100000


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


def get_spy_positive():
    try:
        spy = yf.download("SPY", period="2d", interval="1d", progress=False)
        if spy.empty or len(spy) < 2:
            print("  SPY data unavailable - blocking new entries")
            return False
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        change = float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[-2]) - 1
        print(f"  SPY: {change:+.2%}")
        return change > -0.005
    except Exception as e:
        print(f"  SPY check failed ({e}) - blocking new entries")
        return False


def has_severe_news(symbol):
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        news  = fh.company_news(symbol, _from=today, to=today)
        two_hrs_ago = time.time() - 7200
        for a in (news or [])[:5]:
            if a.get("datetime", 0) < two_hrs_ago:
                continue
            if any(w in a.get("headline", "").lower() for w in ["lawsuit", "fraud", "bankruptcy", "hack", "crash", "recall"]):
                return True
        return False
    except Exception as e:
        print(f"  News check failed for {symbol} ({e}) - blocking entry")
        return True


def get_kronos_confidence(symbol):
    if not KRONOS_AVAILABLE:
        return None
    try:
        df = yf.download(symbol, period="90d", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna().values.astype(float)
        current = closes[-1]
        context = torch.tensor(closes, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            forecast = _run_chronos_predict(context, prediction_length=5, num_samples=100)
        samples = forecast[0].numpy()
        return float(np.mean(samples[:, 0] > current)), float(np.mean(samples[:, 2] > current))
    except Exception as e:
        print(f"  Kronos error {symbol}: {e}")
        return None


def get_day_change(symbol):
    try:
        df = yf.download(symbol, period="2d", interval="1d", progress=False)
        if df.empty or len(df) < 2:
            return 0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1
    except Exception:
        return 0


def buy(symbol, price, portfolio, cash):
    size = min(cash, portfolio * POSITION_PCT)
    qty = max(1, int(size / price))
    limit_price = round(price * (1 + SLIPPAGE_BPS / 10_000), 2)
    try:
        api.submit_order(symbol=symbol, qty=qty, side="buy", type="limit",
                          limit_price=limit_price, time_in_force="day")
        print(f"  BUY {qty} x {symbol} @ limit ${limit_price:.2f}")
        return qty
    except Exception as e:
        print(f"  BUY failed {symbol}: {e}")
        return 0


def sell(symbol, qty, price, reason=""):
    limit_price = round(price * (1 - SLIPPAGE_BPS / 10_000), 2)
    try:
        api.submit_order(symbol=symbol, qty=qty, side="sell", type="limit",
                          limit_price=limit_price, time_in_force="day")
        print(f"  SELL {qty} x {symbol} @ limit ${limit_price:.2f} - {reason}")
    except Exception as e:
        print(f"  SELL failed {symbol}: {e}")


def close_all():
    # Forced end-of-day liquidation must fill, so this uses a market order
    # unlike the limit orders sell() uses during normal exits.
    for symbol, pos in get_positions().items():
        qty = int(float(pos.qty))
        try:
            api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day")
            print(f"  SELL {qty} x {symbol} - market close")
        except Exception as e:
            print(f"  SELL failed {symbol}: {e}")


def run():
    print("=" * 60)
    print("  Level 4 - Kronos AI Day Trading Bot")
    print(f"  Kronos: {'ACTIVE' if KRONOS_AVAILABLE else 'FALLBACK'}")
    print("=" * 60)
    portfolio, cash, pl = get_portfolio()
    print(f"  Portfolio: ${portfolio:,.2f} | Cash: ${cash:,.2f} | P&L: ${pl:+,.2f}\n")

    # Reconcile any positions already open (e.g. bot restarted mid-session) so they
    # get profit-target/stop-loss/time-stop management instead of sitting un-managed
    # until the 30-minutes-to-close liquidation.
    held = {}
    now0 = datetime.datetime.now(timezone.utc)
    for symbol, pos in get_positions().items():
        held[symbol] = {"entry": float(pos.avg_entry_price), "entry_time": now0}
        print(f"  Reconciled existing position: {symbol} @ ${float(pos.avg_entry_price):.2f}")

    while True:
        now = datetime.datetime.now(timezone.utc)
        if not is_market_open():
            print(f"[{now.strftime('%H:%M:%S')}] Market closed. Waiting 60s...")
            time.sleep(60)
            continue
        since_open = mins_since_open()
        to_close   = mins_to_close()
        if to_close <= 30:
            print("\n30 min to close - closing all")
            close_all()
            held = {}
            time.sleep(60)
            continue
        portfolio, cash, pl = get_portfolio()
        print(f"\n[{now.strftime('%H:%M:%S')}] Scanning... ({to_close} min to close | {since_open} min since open)")
        print(f"  Portfolio: ${portfolio:,.2f} | Cash: ${cash:,.2f} | P&L: ${pl:+,.2f}")
        if since_open < NO_ENTRY_BEFORE_MINS:
            print(f"  Waiting {NO_ENTRY_BEFORE_MINS - since_open} min to settle")
            time.sleep(60)
            continue
        positions = get_positions()
        for s in list(held.keys()):
            if s not in positions:
                del held[s]
        for symbol in list(held.keys()):
            if symbol not in positions:
                continue
            price = get_live_price(symbol)
            if not price:
                continue
            entry     = held[symbol]["entry"]
            mins_held = (now - held[symbol]["entry_time"]).total_seconds() / 60
            change    = (price - entry) / entry
            if change >= PROFIT_TARGET:
                print(f"  {symbol} +{change*100:.2f}% - PROFIT TARGET")
                sell(symbol, int(float(positions[symbol].qty)), price, "profit target")
                del held[symbol]
            elif change <= -STOP_LOSS:
                print(f"  {symbol} {change*100:.2f}% - STOP LOSS")
                sell(symbol, int(float(positions[symbol].qty)), price, "stop loss")
                del held[symbol]
            elif mins_held >= TIME_STOP_MINUTES and change <= 0:
                print(f"  {symbol} {change*100:.2f}% - TIME STOP")
                sell(symbol, int(float(positions[symbol].qty)), price, "time stop")
                del held[symbol]
            else:
                print(f"  Holding {symbol} | {change*100:+.2f}% | {mins_held:.0f} min")
        positions = get_positions()
        if len(positions) >= MAX_POSITIONS:
            print("  Max positions reached")
            time.sleep(60)
            continue
        if not get_spy_positive():
            print("  SPY negative - no new entries")
            time.sleep(60)
            continue
        signals = []
        for symbol in WATCHLIST:
            if symbol in held or symbol in positions:
                continue
            if has_severe_news(symbol):
                continue
            day_change = get_day_change(symbol)
            if day_change <= 0:
                continue
            result = get_kronos_confidence(symbol)
            if result is None:
                continue
            conf_1d, conf_3d = result
            print(f"  {symbol}: day={day_change:+.2%} | 1d={conf_1d:.0%} | 3d={conf_3d:.0%}")
            if conf_1d >= CONFIDENCE_1D and conf_3d >= CONFIDENCE_3D:
                signals.append((symbol, day_change, conf_1d + conf_3d))
        if signals:
            signals.sort(key=lambda x: x[2], reverse=True)
            best = signals[0]
            price = get_live_price(best[0])
            if price:
                print(f"\n  Best signal: {best[0]} - buying")
                qty = buy(best[0], price, portfolio, cash)
                if qty:
                    held[best[0]] = {"entry": price, "entry_time": now}
        time.sleep(60)


if __name__ == "__main__":
    run()
