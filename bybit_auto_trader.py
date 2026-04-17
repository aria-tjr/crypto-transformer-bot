import os
import time
import pandas as pd
import schedule
from datetime import datetime
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from models.tcn import TCNAttention, TCNConfig

# Load Environment Variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
IS_DEMO = os.getenv("BYBIT_DEMO", "True").lower() == "true"

# Bybit Demo URL (Unified Trading)
DEMO_URL = "https://api-demo.bybit.com" if IS_DEMO else "https://api.bybit.com"

# Default symbol universe (used if dynamic universe is disabled)
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'MATICUSDT', 'OPUSDT', 'ARBUSDT', 'ATOMUSDT', 'LTCUSDT',
    'BCHUSDT', 'NEARUSDT', 'UNIUSDT', 'AAVEUSDT', 'SUIUSDT',
    '1000PEPEUSDT', 'RNDRUSDT', 'INJUSDT', 'FTMUSDT', 'ETCUSDT'
]

# If enabled, the bot will auto-select the top N most liquid USDT perpetuals by 24h turnover.
USE_DYNAMIC_SYMBOL_UNIVERSE = os.getenv("USE_DYNAMIC_SYMBOL_UNIVERSE", "True").lower() == "true"
TOP_N_SYMBOLS = int(os.getenv("TOP_N_SYMBOLS", "100"))

# Strategy Parameters
ENTRY_WINDOW = 5    # Boss Entry (5-Day High) - Aggressive Entry
SCOUT_WINDOW = 3    # Scout Entry (3-Day High) - Very Aggressive
EXIT_WINDOW = 5     # Stop Loss (5-Day Low) - Tight Stop
MOMENTUM_WINDOW = 20 # Short-term Momentum
LEVERAGE = 5        # 5x Leverage
RISK_PER_ASSET = 1.50 # 150% of Equity per Top 3 Asset (Total 450% exposure)

# Filter Parameters (Win Rate Optimization)
ATR_WINDOW = 14
ATR_MA_WINDOW = 20

# Tunable strictness (trade-frequency friendly default)
# Condition becomes: ATR(14) > ATR_MA(20) * ATR_RISING_MULTIPLIER
# - Higher multiplier => fewer trades, higher quality
# - Lower multiplier  => more trades, slightly more chop
ATR_RISING_MULTIPLIER = float(os.getenv("ATR_RISING_MULTIPLIER", "0.80"))

# SMA 200 Trend Filter (Safety)
# If True, only buy if Price > SMA 200.
ENABLE_SMA_FILTER = os.getenv("ENABLE_SMA_FILTER", "False").lower() == "true"

# Breakout-quality filter (Edge) used as a *soft* gate:
# only require Edge when ATR expansion is weak.
# Edge = (Close - 20DHigh) / ATR.
EDGE_FILTER_MODE = os.getenv("EDGE_FILTER_MODE", "off").lower()  # off|soft|on
BREAKOUT_EDGE_THRESHOLD = float(os.getenv("BREAKOUT_EDGE_THRESHOLD", "0.01"))

# If ATR expansion multiplier is below this level, require Edge to avoid tiny breakouts.
EDGE_REQUIRED_BELOW_MULTIPLIER = float(os.getenv("EDGE_REQUIRED_BELOW_MULTIPLIER", "0.95"))

# Safety: dry-run disables order placement (still scans, prints decisions)
DRY_RUN = os.getenv("DRY_RUN", "False").lower() == "true"

# AI Filter Configuration
ENABLE_AI_FILTER = os.getenv("ENABLE_AI_FILTER", "True").lower() == "true"
AI_MODEL_PATH = "checkpoints/transformer_best.pt"
AI_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to trust the model
AI_ALLOW_NEUTRAL = True  # If True, allow trades when AI predicts "Neutral" (Class 1)

class BybitAutoTrader:
    def __init__(self):
        print(f"🚀 Initializing Bybit Auto Trader (Demo: {IS_DEMO})")
        
        self.session = HTTP(
            demo=IS_DEMO,
            api_key=API_KEY,
            api_secret=API_SECRET,
        )
        print(f"   Endpoint: {self.session.endpoint}")
        print(
            "   Config: "
            f"ATRx={ATR_RISING_MULTIPLIER:.2f}, "
            f"EdgeFilterMode={EDGE_FILTER_MODE}({BREAKOUT_EDGE_THRESHOLD:.2f}), "
            f"EdgeReqBelow={EDGE_REQUIRED_BELOW_MULTIPLIER:.2f}x, "
            f"DRY_RUN={DRY_RUN}"
        )
        
        # Verify Connection
        try:
            # Get Server Time to check connection
            self.session.get_server_time()
            print("   ✅ Connected to Bybit API")
        except Exception as e:
            print(f"   ❌ Connection Failed: {e}")
            exit()

        # Cache instrument info for qty precision
        self._instrument_meta: Dict[str, Dict[str, float]] = {}

        # Load AI Model
        self.ai_model = None
        if ENABLE_AI_FILTER:
            self.load_ai_model()

    def load_ai_model(self):
        """Load the trained AI model for trend confirmation."""
        try:
            if not os.path.exists(AI_MODEL_PATH):
                print(f"   ⚠️ AI Model not found at {AI_MODEL_PATH}. AI Filter disabled.")
                return

            print(f"   🧠 Loading AI Model from {AI_MODEL_PATH}...")
            
            # Initialize model architecture (Must match training config)
            # Training used TCN Attention with d_model=256, n_layers=4
            config = TCNConfig(
                input_dim=16,  # 16 features as per training
                num_channels=[256] * 4,
                kernel_size=3,
                dropout=0.2,
                output_dim=3
            )
            self.ai_model = TCNAttention(config)
            
            # Load weights
            checkpoint = torch.load(AI_MODEL_PATH, map_location=torch.device('cpu'))
            if 'model' in checkpoint:
                self.ai_model.load_state_dict(checkpoint['model'])
            else:
                self.ai_model.load_state_dict(checkpoint)
                
            self.ai_model.eval()
            print("   ✅ AI Model Loaded Successfully")
            
        except Exception as e:
            print(f"   ❌ Failed to load AI model: {e}")
            self.ai_model = None

    def get_ai_confirmation(self, symbol: str, klines_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run AI inference to confirm trend.
        Returns: {'allowed': bool, 'prediction': str, 'confidence': float}
        """
        if not self.ai_model or len(klines_df) < 100:
            return {'allowed': True, 'prediction': 'N/A', 'confidence': 0.0}

        try:
            # 1. Feature Engineering (Must match training exactly)
            df = klines_df.copy().sort_values('startTime').reset_index(drop=True)
            
            # Calculate features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['volume_ma'] = df['volume'].rolling(20).mean()
            
            # ATR
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # Momentum
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_6'] = df['close'].pct_change(6)
            df['momentum_12'] = df['close'].pct_change(12)
            
            # Ratios
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
            df['price_sma10_ratio'] = df['close'] / (df['sma_10'] + 1e-8) - 1
            df['price_sma20_ratio'] = df['close'] / (df['sma_20'] + 1e-8) - 1
            
            # MACD
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_mid'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-8)
            
            # Select Features
            feature_cols = [
                'close', 'returns', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_position', 'bb_width', 'atr_pct', 'momentum_3', 'momentum_6', 'momentum_12',
                'volume_ratio', 'price_sma10_ratio', 'price_sma20_ratio'
            ]
            
            # Prepare Sequence (Last 72 candles)
            df = df.dropna()
            if len(df) < 72:
                return {'allowed': True, 'prediction': 'Insufficient Data', 'confidence': 0.0}
                
            seq = df[feature_cols].tail(72).copy()
            
            # Normalize (Z-Score)
            for col in feature_cols[1:]:
                mean = seq[col].mean()
                std = seq[col].std()
                if std > 0:
                    seq[col] = (seq[col] - mean) / std
            
            # Inference
            x = torch.FloatTensor(seq.values).unsqueeze(0) # (1, 72, 16)
            with torch.no_grad():
                output = self.ai_model(x)
                probs = output['direction_probs'][0] # [Down, Neutral, Up]
                
            # Interpret
            down_prob = probs[0].item()
            neutral_prob = probs[1].item()
            up_prob = probs[2].item()
            
            # Logic: Block only if Strong Down Signal
            # Allow if Up OR (Neutral AND Not Bearish)
            
            prediction = "Neutral"
            if up_prob > down_prob and up_prob > neutral_prob:
                prediction = "Up"
            elif down_prob > up_prob and down_prob > neutral_prob:
                prediction = "Down"
                
            allowed = True
            reason = "AI Approved"
            
            if prediction == "Down":
                allowed = False
                reason = f"AI Bearish ({down_prob:.2%})"
            elif prediction == "Neutral" and not AI_ALLOW_NEUTRAL:
                allowed = False
                reason = "AI Neutral (Strict Mode)"
                
            return {
                'allowed': allowed,
                'prediction': prediction,
                'confidence': max(down_prob, neutral_prob, up_prob),
                'reason': reason,
                'probs': (down_prob, neutral_prob, up_prob)
            }
            
        except Exception as e:
            print(f"      ⚠️ AI Inference Error: {e}")
            return {'allowed': True, 'prediction': 'Error', 'confidence': 0.0}

    def get_instrument_meta(self, symbol: str) -> Dict[str, float]:
        """Fetch qtyStep/minOrderQty for a symbol (best-effort).

        Bybit returns these under instruments info. We cache results to avoid repeated calls.
        If unavailable, fall back to conservative defaults.
        """
        if symbol in self._instrument_meta:
            return self._instrument_meta[symbol]

        meta = {"qtyStep": 0.001, "minOrderQty": 0.001, "maxOrderQty": 1000000.0}
        try:
            resp = self.session.get_instruments_info(category="linear", symbol=symbol)
            if isinstance(resp, tuple):
                resp = resp[0]
            if isinstance(resp, dict) and resp.get("retCode") == 0:
                items = (resp.get("result", {}) or {}).get("list", [])
                if items:
                    lot = items[0].get("lotSizeFilter", {}) or {}
                    qty_step = float(lot.get("qtyStep", meta["qtyStep"]))
                    min_qty = float(lot.get("minOrderQty", meta["minOrderQty"]))
                    
                    # Prefer maxMktOrderQty for safety since we use Market orders
                    # If maxMktOrderQty is missing/zero, fall back to maxOrderQty
                    mkt_max = float(lot.get("maxMktOrderQty", 0))
                    limit_max = float(lot.get("maxOrderQty", 0))
                    
                    if mkt_max > 0:
                        meta["maxOrderQty"] = mkt_max
                    elif limit_max > 0:
                        meta["maxOrderQty"] = limit_max
                    
                    # Guard against weird zeros
                    if qty_step > 0:
                        meta["qtyStep"] = qty_step
                    if min_qty > 0:
                        meta["minOrderQty"] = min_qty
        except Exception:
            # best-effort; keep defaults
            pass

        self._instrument_meta[symbol] = meta
        return meta

    @staticmethod
    def _quantize_down(value: float, step: float) -> float:
        if step <= 0:
            return value
        return (value // step) * step

    def format_qty(self, symbol: str, qty: float) -> Optional[str]:
        """Round qty down to the exchange step and validate min/max qty."""
        try:
            qty = float(qty)
        except Exception:
            return None
        if qty <= 0:
            return None

        meta = self.get_instrument_meta(symbol)
        step = float(meta.get("qtyStep", 0.001))
        min_qty = float(meta.get("minOrderQty", 0.001))
        max_qty = float(meta.get("maxOrderQty", 1000000.0))

        # Cap at max_qty
        # if qty > max_qty:
        #    print(f"      ⚠️ Capping qty {qty} to max {max_qty} for {symbol}")
        #    qty = max_qty

        q = self._quantize_down(qty, step)
        # Round to a reasonable number of decimals based on step
        decimals = max(0, min(8, len(str(step).split(".")[-1]) if "." in str(step) else 0))
        q = round(q, decimals)
        if q < min_qty or q <= 0:
            return None
        return str(q)

    def get_max_order_qty(self, symbol: str) -> float:
        meta = self.get_instrument_meta(symbol)
        return float(meta.get("maxOrderQty", 1000000.0))

    def get_symbol_universe(self) -> List[str]:
        """Return the symbol universe to scan.

        If dynamic universe is enabled, this pulls Bybit tickers and selects
        the top N USDT symbols by 24h turnover (liquidity proxy).
        """
        if not USE_DYNAMIC_SYMBOL_UNIVERSE:
            return SYMBOLS

        try:
            resp = self.session.get_tickers(category="linear")
            if isinstance(resp, tuple):
                resp = resp[0]
            if not isinstance(resp, dict) or resp.get("retCode") != 0:
                return SYMBOLS

            items = (resp.get("result", {}) or {}).get("list", [])
            scored = []
            for it in items:
                sym = it.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                try:
                    turnover = float(it.get("turnover24h", 0.0))
                except Exception:
                    turnover = 0.0
                scored.append((turnover, sym))
            scored.sort(reverse=True)
            dynamic_syms = [s for _, s in scored[: max(5, TOP_N_SYMBOLS)]]

            # De-dupe while preserving order
            seen = set()
            out = []
            for s in dynamic_syms:
                if s not in seen:
                    out.append(s)
                    seen.add(s)
            return out if out else SYMBOLS
        except Exception:
            return SYMBOLS

    def get_kline_data(self, symbol: str, interval: str = "D", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            # pybit may return either a dict or a tuple like (dict, headers)
            if isinstance(response, tuple):
                response = response[0]
            
            if response.get('retCode') != 0:
                print(f"   Error fetching {symbol}: {response.get('retMsg')}")
                return None
                
            result = response.get('result', {})
            data_list = result.get('list', [])
            
            if not data_list:
                return None

            # Bybit returns [time, open, high, low, close, volume, turnover]
            # Reverse to get chronological order
            df = pd.DataFrame(data_list, columns=['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.iloc[::-1].reset_index(drop=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            print(f"   Exception fetching {symbol}: {e}")
            return None

    def get_wallet_balance(self) -> float:
        """Get Total Equity (USDT)"""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if isinstance(response, tuple):
                response = response[0]
            if response.get('retCode') == 0:
                result = response.get('result', {})
                account_list = result.get('list', [])
                if account_list:
                    return float(account_list[0]['totalEquity'])
            return 0.0
        except Exception as e:
            print(f"   Error getting balance: {e}")
            return 0.0

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current open positions"""
        positions = {}
        try:
            response = self.session.get_positions(category="linear", settleCoin="USDT")
            if isinstance(response, tuple):
                response = response[0]
            if response.get('retCode') == 0:
                result = response.get('result', {})
                pos_list = result.get('list', [])
                for pos in pos_list:
                    size = float(pos['size'])
                    if size > 0:
                        positions[pos['symbol']] = {
                            'size': size,
                            'side': pos['side'],
                            'avgPrice': float(pos['avgPrice'])
                        }
            return positions
        except Exception as e:
            print(f"   Error getting positions: {e}")
            return {}

    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol"""
        try:
            self.session.cancel_all_orders(category="linear", symbol=symbol)
        except:
            pass

    def place_conditional_order(self, symbol: str, side: str, qty: float, trigger_price: float, order_type: str = "Market"):
        """Place a conditional (stop) order"""
        try:
            qty_str = self.format_qty(symbol, qty)
            if not qty_str:
                print(f"      ⚠️ Skipping order for {symbol}: qty invalid after rounding")
                return

            if DRY_RUN:
                print(f"      🧪 DRY_RUN: would place {side} conditional for {symbol} qty={qty_str} trigger={trigger_price}")
                return
            
            self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType=order_type,
                qty=qty_str,
                triggerPrice=str(trigger_price),
                triggerDirection=1 if side == "Buy" else 2, # 1: Rise, 2: Fall
                triggerBy="LastPrice",
                positionIdx=0, # One-Way Mode
                reduceOnly=False
            )
            print(f"      Placed {side} Stop Order for {symbol} @ {trigger_price}")
        except Exception as e:
            print(f"      Error placing order for {symbol}: {e}")

    def set_leverage(self, symbol: str, leverage: int = 1):
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
        except:
            pass # Leverage might already be set

    def get_active_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch all active orders (limit and conditional) for a symbol."""
        orders = []
        try:
            # 1. Normal Orders (Limit/Market working)
            resp = self.session.get_open_orders(category="linear", symbol=symbol, openOnly=0)
            if isinstance(resp, tuple): resp = resp[0]
            if resp.get("retCode") == 0:
                orders.extend(resp.get("result", {}).get("list", []))
            
            # 2. Conditional Orders (Stop Loss / Breakout Entries)
            # Note: In Unified Account, get_open_orders usually covers everything, 
            # but sometimes stop orders are separate depending on API version. 
            # For 'linear', get_open_orders covers both if we don't filter.
            # We'll assume the above call gets them.
        except Exception as e:
            print(f"      ⚠️ Error fetching orders for {symbol}: {e}")
        return orders

    def sync_orders(self, symbol: str, desired_orders: List[Dict[str, Any]]):
        """
        Smart Order Management:
        1. Fetch current active orders.
        2. Compare with desired orders.
        3. Cancel unneeded orders.
        4. Place missing orders.
        5. (Optional) Amend modified orders (for now, cancel & replace is safer).
        """
        active_orders = self.get_active_orders(symbol)
        
        # 0. Pre-process: Split Desired Orders if they exceed maxOrderQty
        split_desired = []
        meta = self.get_instrument_meta(symbol)
        max_qty = meta.get("maxOrderQty", 1000000.0)
        
        for order in desired_orders:
            raw_qty = float(order['qty'])
            remaining = raw_qty
            
            # If it's small enough, keep as is
            if raw_qty <= max_qty:
                split_desired.append(order)
                continue
                
            # Otherwise split
            print(f"      🔄 Splitting Desired Order: {order['side']} {raw_qty} (Max: {max_qty})")
            while remaining > 0:
                chunk = min(remaining, max_qty)
                # We need to format it to ensure it's valid, but keep it as float/str for the dict
                chunk_str = self.format_qty(symbol, chunk)
                if not chunk_str or float(chunk_str) <= 0: 
                    break
                
                new_order = order.copy()
                new_order['qty'] = float(chunk_str)
                split_desired.append(new_order)
                
                remaining -= float(chunk_str)
                if remaining < (float(chunk_str) * 0.001): 
                    break
        
        desired_orders = split_desired

        # Helper to create a signature for an order to check equality
        # Signature: (side, orderType, triggerPrice, qty)
        def get_order_sig(order):
            # Handle both API response format and our internal desired format
            # API: 'side', 'orderType', 'triggerPrice', 'qty'
            # Desired: 'side', 'order_type', 'trigger_price', 'qty'
            
            # Normalize Side
            side = order.get('side')
            
            # Normalize Type
            o_type = order.get('orderType') or order.get('order_type')
            
            # Normalize Price (Trigger or Price)
            # For conditional orders, we care about triggerPrice. For Limit, price.
            # Our strategy mainly uses Conditional Market orders.
            trig = order.get('triggerPrice') or order.get('trigger_price')
            if trig:
                trig = float(trig)
            else:
                trig = 0.0
                
            # Normalize Qty
            qty = float(order.get('qty', 0))
            
            return (side, o_type, f"{trig:.4f}", f"{qty:.4f}")

        # 1. Identify Orders to Keep vs Cancel
        active_sigs = {get_order_sig(o): o['orderId'] for o in active_orders}
        desired_sigs = [get_order_sig(o) for o in desired_orders]
        
        # Orders to Cancel: Active orders whose sig is NOT in desired
        # (This is a strict equality check. If price moves slightly, we cancel & replace. Good.)
        for sig, order_id in active_sigs.items():
            if sig not in desired_sigs:
                print(f"      ❌ Cancelling obsolete order {order_id} (Sig: {sig})")
                try:
                    self.session.cancel_order(category="linear", symbol=symbol, orderId=order_id)
                except Exception as e:
                    print(f"         Error cancelling: {e}")

        # 2. Identify Orders to Place
        for i, desired in enumerate(desired_orders):
            sig = desired_sigs[i]
            if sig in active_sigs:
                # Already exists
                continue
                
            # Place New
            print(f"      ✨ Placing NEW order: {desired['side']} {desired['qty']} @ {desired.get('trigger_price')}")
            self.place_conditional_order(
                symbol=symbol,
                side=desired['side'],
                qty=desired['qty'],
                trigger_price=desired['trigger_price'],
                order_type=desired['order_type']
            )

    def run_strategy(self):
        print("\n" + "="*70)
        print(f"🤖 RUNNING STRATEGY LOOP - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # 1. Account Info
        equity = self.get_wallet_balance()
        print(f"   💰 Total Equity: ${equity:,.2f}")
        
        current_positions = self.get_positions()
        print(f"   🚩 Open Positions: {list(current_positions.keys())}")

        # 2. Market Scan & Momentum
        symbols = self.get_symbol_universe()
        
        # Ensure we scan our current positions even if they dropped out of Top N
        for pos_sym in current_positions.keys():
            if pos_sym not in symbols:
                symbols.append(pos_sym)
                
        print(f"   📊 Scanning Market ({len(symbols)} symbols) & Calculating Momentum...")
        momentum_scores = {}
        market_data = {}

        for sym in symbols:
            df = self.get_kline_data(sym, limit=250) # Need 200 for SMA
            if df is None: continue
            
            # Momentum (90d)
            current_price = df['close'].iloc[-1]
            lookback = min(MOMENTUM_WINDOW, len(df)-1)
            past_price = df['close'].iloc[-lookback]
            mom = (current_price - past_price) / past_price
            momentum_scores[sym] = mom
            
            # Levels
            # Shift by 1 (Yesterday's High/Low)
            entry_price = df['high'].iloc[-ENTRY_WINDOW-1:-1].max()
            scout_price = df['high'].iloc[-SCOUT_WINDOW-1:-1].max()
            exit_price = df['low'].iloc[-EXIT_WINDOW-1:-1].min()
            
            # --- FILTERS ---
            # 1. SMA 200 Trend Filter
            sma_200 = df['close'].rolling(window=200).mean().iloc[-1]
            sma_ok = True
            if pd.notna(sma_200):
                sma_ok = current_price > sma_200
            
            # 2. ATR Filter
            df['tr0'] = abs(df['high'] - df['low'])
            df['tr1'] = abs(df['high'] - df['close'].shift(1))
            df['tr2'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=ATR_WINDOW).mean()
            df['atr_ma'] = df['atr'].rolling(window=ATR_MA_WINDOW).mean()
            
            # Filter Condition (Rising Volatility)
            atr_rising = df['atr'].iloc[-1] > (df['atr_ma'].iloc[-1] * ATR_RISING_MULTIPLIER)

            # 3. Breakout Edge
            last_atr = df['atr'].iloc[-1]
            breakout_edge = None
            edge_ok = True
            if EDGE_FILTER_MODE != "off":
                if pd.notna(last_atr) and last_atr > 0:
                    breakout_edge = (current_price - entry_price) / last_atr
                    edge_ok = breakout_edge > BREAKOUT_EDGE_THRESHOLD
                else:
                    edge_ok = False
            
            market_data[sym] = {
                'price': current_price,
                'entry': entry_price,
                'scout': scout_price,
                'exit': exit_price,
                'atr': last_atr, # Added ATR for smart management
                'atr_rising': atr_rising,
                'breakout_edge': breakout_edge,
                'edge_ok': edge_ok,
                'sma_ok': sma_ok,
                'sma_200': sma_200
            }
            
            # Ensure Leverage is 1x
            self.set_leverage(sym, LEVERAGE)

        # 3. Select Top 3 (Smart Selection)
        # We want to fill 3 slots with the best Momentum assets that are EITHER:
        # a) Already in our portfolio (Keep winners)
        # b) Valid new entries (Pass filters)
        
        sorted_mom = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = []
        
        print(f"   🕵️ Selecting Top 3 Candidates from {len(sorted_mom)} sorted by Momentum...")
        
        for sym, mom in sorted_mom:
            if len(top_3) >= 3:
                break
                
            # 1. Priority: Keep High Momentum Positions
            if sym in current_positions:
                top_3.append(sym)
                print(f"      ✅ Keeping {sym} (Rank #{sorted_mom.index((sym, mom))+1}, Mom: {mom:.2%}) - Already Holding")
                continue
                
            # 2. Check Filters for New Entries
            data = market_data.get(sym)
            if not data: continue
            
            # SMA Filter
            if ENABLE_SMA_FILTER and not data['sma_ok']:
                print(f"      ⚠️ Skipping {sym} (Rank #{sorted_mom.index((sym, mom))+1}): Price < SMA200")
                continue
                
            # ATR Filter
            if not data['atr_rising']:
                print(f"      ⚠️ Skipping {sym} (Rank #{sorted_mom.index((sym, mom))+1}): ATR not rising")
                continue
                
            # Edge Filter
            if EDGE_FILTER_MODE in ("on", "soft"):
                require_edge = (EDGE_FILTER_MODE == "on") or (ATR_RISING_MULTIPLIER < EDGE_REQUIRED_BELOW_MULTIPLIER)
                if require_edge and not data.get('edge_ok', True):
                    print(f"      ⚠️ Skipping {sym} (Rank #{sorted_mom.index((sym, mom))+1}): Poor Breakout Edge")
                    continue
            
            # If we got here, it's a valid new entry
            top_3.append(sym)
            print(f"      ✅ Selected {sym} (Rank #{sorted_mom.index((sym, mom))+1}, Mom: {mom:.2%}) - Valid Entry Signal")

        print(f"   🏆 Final Top 3 Portfolio: {top_3}")

        # 4. Execution Logic
        
        # A. Manage Existing Positions (Trailing Stops & Add-ons)
        print(f"   �️ Managing {len(current_positions)} Open Positions...")
        
        target_allocation = equity * RISK_PER_ASSET
        
        for sym in current_positions:
            data = market_data.get(sym)
            if not data:
                print(f"   ⚠️ Warning: No market data for held position {sym}. Skipping management.")
                continue
                
            pos_size = current_positions[sym]['size']
            curr_price = data['price']
            pos_entry = current_positions[sym]['avgPrice']
            
            # 1. Dynamic Trailing Stop Logic
            # Base Stop: 10-Day Low (exit_price)
            stop_price = data['exit']
            
            # 1. Smart ATR-Based Trailing Stop Logic
            # This adapts to the asset's volatility rather than using fixed percentages.
            
            atr = data.get('atr', 0)
            if atr <= 0: atr = curr_price * 0.05 # Fallback
            
            # Base Stop: 10-Day Low (Structural Stop)
            stop_price = data['exit']
            
            # Calculate Profit in "R" (Risk Units based on ATR)
            dist_from_entry = curr_price - pos_entry
            r_multiple = dist_from_entry / atr if atr > 0 else 0
            
            # Debug Risk Info
            print(f"      📊 {sym} Risk Info: Entry={pos_entry:.4f}, Price={curr_price:.4f}, ATR={atr:.4f}, R={r_multiple:.2f}x")
            
            # Tier 1: Break-Even (Profit > 0.5 ATR)
            # If we are up 0.5 ATR, move stop to Entry + small buffer to cover fees
            if r_multiple > 0.5:
                be_price = pos_entry + (atr * 0.1)
                if be_price > stop_price:
                    stop_price = be_price
                    print(f"      🔒 Smart Guard: Locked Break-Even for {sym} (Profit > 0.5 ATR)")

            # Tier 2: Trailing Stop (Profit > 1.0 ATR)
            # If we are up 1 ATR, trail at Price - 1.0 ATR
            if r_multiple > 1.0:
                trail_price = curr_price - (atr * 1.0)
                if trail_price > stop_price:
                    stop_price = trail_price
                    print(f"      🔒 Smart Guard: Trailing Active for {sym} (Profit > 1.0 ATR)")
            
            # Tier 3: Parabolic Protection (Profit > 2.0 ATR)
            # If price goes parabolic (2 ATR move), tighten stop to Price - 0.5 ATR
            if r_multiple > 2.0:
                tight_price = curr_price - (atr * 0.5)
                if tight_price > stop_price:
                    stop_price = tight_price
                    print(f"      🔒 Smart Guard: Parabolic Protection for {sym} (Profit > 2.0 ATR)")

            desired_orders = [{
                'side': 'Sell', 
                'qty': pos_size, 
                'trigger_price': stop_price, 
                'order_type': 'Market'
            }]
            
            # 2. Check for Add-on (Boss Entry) if we are undersized
            current_value = pos_size * curr_price
            if current_value < (target_allocation * 0.5):
                remaining_usd = target_allocation - current_value
                qty_to_buy = remaining_usd / data['entry']
                
                if curr_price >= data['entry']:
                    print(f"      🚀 Adding Boss Position for {sym}: Price ({curr_price}) >= Entry ({data['entry']}). Buying NOW.")
                    
                    # AI Confirmation for Add-on
                    allow_add = True
                    if ENABLE_AI_FILTER and self.ai_model:
                        klines_df = self.get_kline_data(sym, interval="5", limit=200)
                        if klines_df is not None:
                            ai_result = self.get_ai_confirmation(sym, klines_df)
                            if not ai_result['allowed']:
                                print(f"      ⛔ Boss Add Blocked by AI: {ai_result['reason']}")
                                allow_add = False
                    
                    if allow_add and not DRY_RUN:
                         self.place_split_market_order(sym, "Buy", qty_to_buy)
                else:
                    desired_orders.append({
                        'side': 'Buy', 
                        'qty': qty_to_buy, 
                        'trigger_price': data['entry'], 
                        'order_type': 'Market'
                    })
            
            self.sync_orders(sym, desired_orders)

        # B. New Entries (Fill Empty Slots)
        open_slots = 3 - len(current_positions)
        
        if open_slots > 0:
            print(f"   ✨ Looking for {open_slots} New Entries from Top 3...")
            for sym in top_3:
                if open_slots <= 0: break
                if sym in current_positions: continue
                
                data = market_data.get(sym)
                if not data: continue
                
                # Filters
                if not data['sma_ok']:
                    print(f"   ⚠️ Skipping {sym}: Below SMA 200 (${data['sma_200']:,.2f})")
                    self.sync_orders(sym, []) 
                    continue

                if not data['atr_rising']:
                    print(f"   ⚠️ Skipping {sym}: ATR not rising enough.")
                    self.sync_orders(sym, [])
                    continue

                if EDGE_FILTER_MODE in ("on", "soft"):
                    require_edge = (EDGE_FILTER_MODE == "on") or (ATR_RISING_MULTIPLIER < EDGE_REQUIRED_BELOW_MULTIPLIER)
                    if require_edge and not data.get('edge_ok', True):
                        print(f"   ⚠️ Skipping {sym}: Breakout edge too small.")
                        self.sync_orders(sym, [])
                        continue
                
                print(f"   🎯 Setting Up Entries for {sym} (Vol Validated ✅)")
                
                # AI Confirmation Check
                if ENABLE_AI_FILTER and self.ai_model:
                    print(f"      🧠 Checking AI Confirmation for {sym}...")
                    # Fetch 5m data for AI (need ~100 candles for indicators)
                    klines_df = self.get_kline_data(sym, interval="5", limit=200)
                    if klines_df is not None:
                        ai_result = self.get_ai_confirmation(sym, klines_df)
                        print(f"      🤖 AI Says: {ai_result['prediction']} (Conf: {ai_result['confidence']:.2%}) -> {ai_result['reason']}")
                        
                        if not ai_result['allowed']:
                            print(f"      ⛔ Trade Blocked by AI: {ai_result['reason']}")
                            self.sync_orders(sym, [])
                            continue
                    else:
                        print("      ⚠️ Could not fetch 5m data for AI. Proceeding with caution.")

                scout_usd = target_allocation * 0.30
                scout_qty = scout_usd / data['scout']
                boss_usd = target_allocation * 0.70
                boss_qty = boss_usd / data['entry']
                
                desired_orders = []
                
                # Case A: Full Breakout
                if data['price'] >= data['entry']:
                    print(f"      🚀 Price ({data['price']:,.4f}) >= Entry ({data['entry']:,.4f}). Buying FULL SIZE immediately.")
                    total_qty = scout_qty + boss_qty
                    if not DRY_RUN:
                         self.place_split_market_order(sym, "Buy", total_qty)
                         open_slots -= 1
                         continue

                # Case B: Partial Breakout
                elif data['price'] >= data['scout']:
                     print(f"      Price ({data['price']:,.4f}) > Scout ({data['scout']:,.4f}). Buying Scout Size NOW.")
                     if not DRY_RUN:
                         self.place_split_market_order(sym, "Buy", scout_qty)
                         open_slots -= 1 
                     
                     desired_orders.append({'side': 'Buy', 'qty': boss_qty, 'trigger_price': data['entry'], 'order_type': 'Market'})

                # Case C: Pre-Breakout
                else:
                    desired_orders.append({'side': 'Buy', 'qty': scout_qty, 'trigger_price': data['scout'], 'order_type': 'Market'})
                    desired_orders.append({'side': 'Buy', 'qty': boss_qty, 'trigger_price': data['entry'], 'order_type': 'Market'})
                    open_slots -= 1

                if desired_orders:
                    self.sync_orders(sym, desired_orders)
        else:
            print("   🔒 Portfolio Full (3/3). No new entries.")

        print("   ✅ Strategy Loop Complete. Sleeping...")

    def place_split_market_order(self, symbol: str, side: str, qty: float):
        """Place a Market order, splitting it if it exceeds maxOrderQty."""
        try:
            meta = self.get_instrument_meta(symbol)
            max_qty = meta.get("maxOrderQty", 1000000.0)
            remaining_qty = qty
            
            print(f"      🔄 Splitting Market Order: Total {qty} (Max/Order: {max_qty})")

            while remaining_qty > 0:
                chunk_qty = min(remaining_qty, max_qty)
                qty_str = self.format_qty(symbol, chunk_qty)
                
                if not qty_str or float(qty_str) <= 0:
                    break

                if DRY_RUN:
                    print(f"      🧪 DRY_RUN: would place {side} Market for {symbol} qty={qty_str}")
                else:
                    try:
                        self.session.place_order(
                            category="linear",
                            symbol=symbol,
                            side=side,
                            orderType="Market",
                            qty=qty_str
                        )
                        print(f"      ✅ Market {side} Placed for {symbol} (Qty: {qty_str})")
                    except Exception as e:
                        print(f"      ❌ Market {side} Failed for {symbol} (Qty: {qty_str}): {e}")
                        # If one chunk fails, we probably shouldn't continue blindly, but for now we try
                
                remaining_qty -= float(qty_str)
                if remaining_qty < (float(qty_str) * 0.001): 
                    break
                    
        except Exception as e:
            print(f"      Error placing split market order for {symbol}: {e}")

if __name__ == "__main__":
    bot = BybitAutoTrader()
    
    # Run immediately on startup
    bot.run_strategy()
    
    # Schedule to run every 15 minutes
    schedule.every(15).minutes.do(bot.run_strategy)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
