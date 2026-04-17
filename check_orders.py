import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
IS_DEMO = os.getenv("BYBIT_DEMO", "True").lower() == "true"

session = HTTP(
    demo=IS_DEMO,
    api_key=API_KEY,
    api_secret=API_SECRET,
)

print(f"Connected to {'Demo' if IS_DEMO else 'Live'} Bybit")

try:
    # Fetch open orders (active limit/stop orders)
    # settleCoin="USDT" is required to fetch all USDT perp orders if symbol is not provided
    resp = session.get_open_orders(category="linear", settleCoin="USDT", limit=50)
    if resp['retCode'] == 0:
        orders = resp['result']['list']
        if not orders:
            print("No open orders found.")
        else:
            print(f"Found {len(orders)} open orders:")
            for o in orders:
                trigger_price = o.get('triggerPrice', 'N/A')
                print(f"- {o['symbol']} {o['side']} {o['qty']} @ {o['price']} ({o['orderType']}) [Status: {o['orderStatus']}, Trigger: {trigger_price}]")
    else:
        print(f"Error fetching orders: {resp}")
except Exception as e:
    print(f"Exception: {e}")
