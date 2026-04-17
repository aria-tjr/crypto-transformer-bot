import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# Load Environment Variables
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
IS_DEMO = os.getenv("BYBIT_DEMO", "True").lower() == "true"

def close_all_positions():
    print(f"🚨 CLOSING ALL POSITIONS (Demo: {IS_DEMO})")
    
    session = HTTP(
        demo=IS_DEMO,
        api_key=API_KEY,
        api_secret=API_SECRET,
    )

    try:
        # 1. Get Open Positions
        resp = session.get_positions(category="linear", settleCoin="USDT")
        if isinstance(resp, tuple): resp = resp[0]
        
        positions = resp.get('result', {}).get('list', [])
        active_positions = [p for p in positions if float(p['size']) > 0]
        
        if not active_positions:
            print("   ✅ No open positions to close.")
            return

        print(f"   Found {len(active_positions)} active positions. Closing now...")

        for p in active_positions:
            symbol = p['symbol']
            size = p['size']
            side = p['side']
            
            print(f"   🔻 Closing {symbol} ({side} {size})...")
            
            # Cancel existing orders first
            try:
                session.cancel_all_orders(category="linear", symbol=symbol)
            except Exception as e:
                print(f"      Warning: Could not cancel orders for {symbol}: {e}")

            # Place Market Close Order
            try:
                close_side = "Sell" if side == "Buy" else "Buy"
                session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=close_side,
                    orderType="Market",
                    qty=size,
                    reduceOnly=True
                )
                print(f"      ✅ Closed {symbol}")
            except Exception as e:
                print(f"      ❌ Failed to close {symbol}: {e}")

    except Exception as e:
        print(f"   ❌ Error fetching positions: {e}")

if __name__ == "__main__":
    close_all_positions()
