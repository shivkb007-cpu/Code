# Kronos day-trading bot (paper trading)

## Before anything else: rotate your keys

The Alpaca and Finnhub keys from earlier in this project were pasted into a chat
transcript, which makes them compromised regardless of what this script does with
them. Generate new ones before running this:

- Alpaca: https://app.alpaca.markets/paper/dashboard/overview -> API Keys -> regenerate
- Finnhub: https://finnhub.io/dashboard -> regenerate API key

## What changed from the version that crashed

- **Secrets are read from environment variables**, not hardcoded in the file — see
  setup below. If a required variable is missing, the script exits immediately with
  a clear message instead of running with a broken client.
- **Fixed the Chronos crash** (`predict() got an unexpected keyword argument 'context'`).
  `chronos-forecasting` renamed `predict()`'s first parameter across versions; the bot
  now tries the current name first and falls back automatically instead of assuming one.
- **SPY / news filters now fail closed.** If the SPY check or Finnhub news check errors
  out (rate limit, network blip), the bot now blocks new entries / treats it as
  potential severe news, instead of silently defaulting to "all clear."
- **Reconciles existing positions on startup.** If the bot restarts while it's already
  holding something, that position gets picked back up for profit-target/stop-loss/
  time-stop management instead of being ignored until the 30-minute close-out.
- **Limit orders instead of market orders** for entries/exits, capped at `SLIPPAGE_BPS`
  (15 bps by default) from the reference price — protects the tight 1%/2% stop/target
  from open-ended slippage. End-of-day forced liquidation still uses a market order
  since that one has to fill no matter what.
- **Position sizing is cash-aware** (`min(cash, portfolio * POSITION_PCT)`), so it won't
  try to size off buying power it doesn't actually have.

## Setup (Windows PowerShell)

```powershell
cd Code
pip install -r live_bot\requirements.txt

# Session-only (simplest — set these each time you open a new PowerShell window):
$env:ALPACA_KEY    = "your-rotated-alpaca-key"
$env:ALPACA_SECRET = "your-rotated-alpaca-secret"
$env:FINNHUB_KEY   = "your-rotated-finnhub-key"

python live_bot\kronos_bot.py
```

To avoid re-typing them every session, set them persistently instead (requires
reopening PowerShell afterward to take effect):

```powershell
setx ALPACA_KEY    "your-rotated-alpaca-key"
setx ALPACA_SECRET "your-rotated-alpaca-secret"
setx FINNHUB_KEY   "your-rotated-finnhub-key"
```

`ALPACA_URL` defaults to the paper-trading endpoint; only set it as an environment
variable if you intend to point this at something else.

## Still true from the earlier review

This is a backtested-nowhere strategy running on 1-minute polling with real order
placement. See `backtest/README.md` in this repo for the harness to validate the
entry logic against history before trusting it further, and see the conversation
history for the fuller list of risk caveats (delayed yfinance quotes, no fine-tuning
of the model to this watchlist, correlation between simultaneous positions, etc.).
