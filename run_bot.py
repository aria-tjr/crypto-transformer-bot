import requests
import pandas as pd
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
]
ENTRY_WINDOW = 20   # "The Boss" (Standard Trend)
SCOUT_WINDOW = 10   # "The Scout" (Early Entry)
EXIT_WINDOW = 10    # Stop Loss
MOMENTUM_WINDOW = 90

def get_binance_data(symbol, limit=100):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1d", "limit": limit}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data, columns=['open_time', 'o', 'h', 'l', 'c', 'v', 'ct', 'qv', 't', 'tbv', 'tbq', 'i'])
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        for c in ['h', 'l', 'c']:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def analyze_market():
    print("=" * 70)
    print(f"🤖 MASTER TRADING BOT (HYBRID MODE) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Strategy: Rotational Turtle + Scout Entry")
    print("=" * 70)
    
    print("   Scanning Market...", end="")
    
    market_data = {}
    momentum_scores = {}
    
    for sym in SYMBOLS:
        df = get_binance_data(sym)
        if df is None: continue
        
        # 1. Calculate Momentum (Relative Strength)
        current_price = df['c'].iloc[-1]
        lookback = min(MOMENTUM_WINDOW, len(df)-1)
        past_price = df['c'].iloc[-lookback]
        mom = (current_price - past_price) / past_price
        momentum_scores[sym] = mom
        
        # 2. Calculate Levels
        # Shift by 1 because we trade based on YESTERDAY'S close
        entry_price = df['h'].iloc[-ENTRY_WINDOW-1:-1].max() # 20-Day High
        scout_price = df['h'].iloc[-SCOUT_WINDOW-1:-1].max() # 10-Day High
        exit_price = df['l'].iloc[-EXIT_WINDOW-1:-1].min()   # 10-Day Low
        
        # 3. Determine Current State (Simulation)
        pos = 0
        start_idx = ENTRY_WINDOW
        for i in range(start_idx, len(df)):
            p = df['c'].iloc[i]
            h = df['h'].iloc[i-ENTRY_WINDOW:i].max()
            l = df['l'].iloc[i-EXIT_WINDOW:i].min()
            
            if pos == 0 and p > h: pos = 1
            elif pos == 1 and p < l: pos = 0
            
        market_data[sym] = {
            'price': current_price,
            'entry': entry_price,
            'scout': scout_price,
            'exit': exit_price,
            'pos': pos
        }
        
    print(" Done!\n")
    
    # Sort by Momentum
    sorted_mom = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = [x[0] for x in sorted_mom[:3]]
    
    # OUTPUT: TOP 3
    print("=" * 70)
    print(f"🏆 TOP 3 ASSETS (Relative Strength)")
    print(f"   Focus Capital Here (33% Each)")
    print("-" * 70)
    
    for rank, (sym, mom) in enumerate(sorted_mom[:3], 1):
        data = market_data[sym]
        status = "LONG ✅" if data['pos'] == 1 else "CASH ⚪"
        
        if data['pos'] == 1:
            action = f"HOLD. Stop Loss at ${data['exit']:,.4f}"
        else:
            # Hybrid Logic Display
            if data['scout'] < data['entry']:
                action = f"WAIT. Scout (30%): ${data['scout']:,.2f} | Full (70%): ${data['entry']:,.2f}"
            else:
                action = f"WAIT. Buy Stop at ${data['entry']:,.2f}"
            
        print(f"   {rank}. {sym:<10} (Mom: {mom*100:>5.1f}%) -> {status} | {action}")
        
    # OUTPUT: IGNORE LIST
    print("\n" + "-"*70)
    print("🗑️ IGNORE LIST (Weak Momentum)")
    print("   Do not trade these, even if they breakout.")
    print("-" * 70)
    for sym, mom in sorted_mom[3:]:
        print(f"   x. {sym:<10} (Mom: {mom*100:>5.1f}%)")
        
    # OUTPUT: ACTION PLAN
    print("\n" + "="*70)
    print("📝 FINAL TRADING PLAN FOR TOMORROW")
    print("=" * 70)
    
    active_buys = []
    active_holds = []
    
    for sym in top_3:
        data = market_data[sym]
        if data['pos'] == 1:
            active_holds.append(f"Keep holding {sym}. Sell if price < ${data['exit']:,.2f}.")
        else:
            if data['scout'] < data['entry']:
                 active_buys.append(f"Watch {sym}. Scout Entry (30%) > ${data['scout']:,.2f}. Full Entry > ${data['entry']:,.2f}.")
            else:
                 active_buys.append(f"Watch {sym}. Buy if price > ${data['entry']:,.2f}.")
            
    if not active_buys and not active_holds:
        print("   😴 NO ACTION. The market is weak. Stay in USDT.")
    else:
        for instruction in active_holds:
            print(f"   • {instruction}")
        for instruction in active_buys:
            print(f"   • {instruction}")
            
if __name__ == "__main__":
    analyze_market()
