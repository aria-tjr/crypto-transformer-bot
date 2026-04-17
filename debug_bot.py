
import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# Config — set via environment; never commit real keys.
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
IS_DEMO = os.getenv("BYBIT_DEMO", "True").lower() == "true"
if not API_KEY or not API_SECRET:
    raise SystemExit("Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables before running.")

session = HTTP(
    demo=IS_DEMO,
    api_key=API_KEY,
    api_secret=API_SECRET,
)

symbols = ["LTCUSDT", "DASHUSDT", "XPLUSDT", "BANKUSDT", "BEATUSDT"]

print("--- SPREAD ANALYSIS ---")
for sym in symbols:
    try:
        ticker = session.get_tickers(category="linear", symbol=sym)['result']['list'][0]
        bid = float(ticker['bid1Price'])
        ask = float(ticker['ask1Price'])
        last = float(ticker['lastPrice'])
        
        spread = ask - bid
        spread_pct = (spread / last) * 100
        
        print(f"{sym}: Bid={bid}, Ask={ask}, Spread={spread_pct:.4f}%")
    except Exception as e:
        print(f"{sym}: Error {e}")

print("\n--- RECENT CLOSED PNL ---")
try:
    # Fetch last 20 closed trades
    resp = session.get_closed_pnl(category="linear", limit=20)
    for item in resp['result']['list']:
        symbol = item['symbol']
        side = item['side']
        qty = item['qty']
        entry = item['avgEntryPrice']
        exit_price = item['avgExitPrice']
        pnl = item['closedPnl']
        roi = float(pnl) / (float(entry) * float(qty) / 5) * 100 # Approx ROI based on 5x lev
        
        print(f"{symbol} {side}: PnL=${item['closedPnl']} (ROI: {roi:.2f}%) | Entry: {entry} -> Exit: {exit_price}")
except Exception as e:
    print(f"Error fetching PnL: {e}")
