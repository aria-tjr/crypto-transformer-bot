import os
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# Load Environment Variables
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
IS_DEMO = os.getenv("BYBIT_DEMO", "True").lower() == "true"

def get_peak_pnl(session, symbol, side, qty, entry_price, start_ts, end_ts=None):
    """Estimate the Maximum Favorable Excursion (Peak P&L) during the trade."""
    if end_ts is None:
        end_ts = int(time.time() * 1000)
    
    # Use 1-minute candles for better resolution
    interval = "1"
    interval_ms = 60 * 1000
    
    # Align start time to the beginning of the interval to ensure we capture the candle covering the start
    aligned_start = start_ts - (start_ts % interval_ms)
    
    try:
        # Fetch klines
        # Add buffer to end_ts to ensure we get the last candle
        resp = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            start=aligned_start,
            end=end_ts + interval_ms, 
            limit=1000
        )
        if isinstance(resp, tuple): resp = resp[0]
        list_data = resp.get('result', {}).get('list', [])
        
        if not list_data:
            # Fallback to 5 min if 1 min fails or is empty
            interval = "5"
            interval_ms = 5 * 60 * 1000
            aligned_start = start_ts - (start_ts % interval_ms)
            resp = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                start=aligned_start,
                end=end_ts + interval_ms,
                limit=1000
            )
            if isinstance(resp, tuple): resp = resp[0]
            list_data = resp.get('result', {}).get('list', [])

        if not list_data:
            return 0.0
            
        # list is [startTime, open, high, low, close, ...]
        highs = [float(x[2]) for x in list_data]
        lows = [float(x[3]) for x in list_data]
        
        if not highs: return 0.0

        if side == "Buy":
            peak_price = max(highs)
            return (peak_price - entry_price) * qty
        else: # Sell
            peak_price = min(lows)
            return (entry_price - peak_price) * qty
    except Exception as e:
        print(f"Error calc peak pnl for {symbol}: {e}")
        return 0.0

def check_pnl(hours=24):
    print(f"🚀 Checking P&L for the last {hours} hours (Demo: {IS_DEMO})")
    
    session = HTTP(
        demo=IS_DEMO,
        api_key=API_KEY,
        api_secret=API_SECRET,
    )

    # 1. Get Wallet Balance (Total Equity)
    try:
        resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if isinstance(resp, tuple): resp = resp[0]
        equity = float(resp['result']['list'][0]['totalEquity'])
        print(f"\n💰 Current Total Equity: ${equity:,.2f}")
    except Exception as e:
        print(f"Error fetching equity: {e}")

    # 2. Get Closed P&L (Realized)
    start_dt = datetime.now() - timedelta(hours=hours)
    start_ts = int(start_dt.timestamp() * 1000)
    print(f"\n📜 Closed Trades since {start_dt.strftime('%Y-%m-%d %H:%M')}:")
    
    total_realized_pnl = 0.0
    all_trades = []
    cursor = None
    
    try:
        while True:
            params = {
                "category": "linear",
                "limit": 50, # Limit to 50 to avoid slow peak calc
                "startTime": start_ts
            }
            if cursor:
                params["cursor"] = cursor

            resp = session.get_closed_pnl(**params)
            if isinstance(resp, tuple): resp = resp[0]
            
            trades = resp.get('result', {}).get('list', [])
            if not trades:
                break
                
            all_trades.extend(trades)
            cursor = resp.get('result', {}).get('nextPageCursor')
            
            if not cursor:
                break
                
        if not all_trades:
            print("   No closed trades found in this period.")
        else:
            # Filter and process
            data = []
            print("   Calculating Peak P&L for closed trades (this may take a moment)...")
            for t in all_trades:
                trade_ts = int(t['updatedTime'])
                created_ts = int(t['createdTime'])
                if trade_ts < start_ts:
                    continue
                    
                closed_pnl = float(t['closedPnl'])
                total_realized_pnl += closed_pnl
                
                qty = float(t['qty'])
                entry = float(t['avgEntryPrice'])
                
                # Calculate Peak P&L
                peak_pnl = get_peak_pnl(session, t['symbol'], t['side'], qty, entry, created_ts, trade_ts)
                
                data.append({
                    'Time': datetime.fromtimestamp(trade_ts/1000).strftime('%Y-%m-%d %H:%M'),
                    'Symbol': t['symbol'],
                    'Side': t['side'],
                    'Qty': qty,
                    'Entry': entry,
                    'Exit': float(t['avgExitPrice']),
                    'Realized P&L': closed_pnl,
                    'Peak P&L': peak_pnl,
                    'Left on Table': peak_pnl - closed_pnl
                })
            
            if data:
                df = pd.DataFrame(data)
                # Sort by Realized P&L desc
                df = df.sort_values('Realized P&L', ascending=False)
                print(df.to_string(index=False))
                print(f"\n   💵 Total Realized P&L ({len(data)} trades): ${total_realized_pnl:,.2f}")
            else:
                print("   No trades found in the exact time window.")

    except Exception as e:
        print(f"Error fetching closed P&L: {e}")

    # 3. Get Open Positions (Unrealized P&L)
    print("\n🔓 Open Positions (Unrealized P&L):")
    total_unrealized_pnl = 0.0
    try:
        resp = session.get_positions(category="linear", settleCoin="USDT")
        if isinstance(resp, tuple): resp = resp[0]
        
        positions = resp.get('result', {}).get('list', [])
        active_positions = [p for p in positions if float(p['size']) > 0]
        
        if not active_positions:
            print("   No open positions.")
        else:
            data = []
            print("   Calculating Peak P&L for open positions...")
            for p in active_positions:
                unrealized = float(p['unrealisedPnl'])
                total_unrealized_pnl += unrealized
                
                qty = float(p['size'])
                entry = float(p['avgPrice'])
                created_ts = int(p['createdTime'])
                
                peak_pnl = get_peak_pnl(session, p['symbol'], p['side'], qty, entry, created_ts)
                
                data.append({
                    'Symbol': p['symbol'],
                    'Side': p['side'],
                    'Size': qty,
                    'Entry': entry,
                    'Mark': float(p['markPrice']),
                    'Unrealized P&L': unrealized,
                    'Peak P&L': peak_pnl,
                    'Drawdown from Peak': unrealized - peak_pnl
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('Unrealized P&L', ascending=False)
            print(df.to_string(index=False))
            print(f"\n   💎 Total Unrealized P&L: ${total_unrealized_pnl:,.2f}")

    except Exception as e:
        print(f"Error fetching positions: {e}")

    # Summary
    print("\n" + "="*40)
    print(f"SUMMARY (Last {hours} Hours)")
    print("="*40)
    print(f"Realized P&L:            ${total_realized_pnl:,.2f}")
    print(f"Unrealized P&L:          ${total_unrealized_pnl:,.2f}")
    print(f"Net P&L Impact:          ${(total_realized_pnl + total_unrealized_pnl):,.2f}")
    print("="*40)

if __name__ == "__main__":
    check_pnl(hours=24)
