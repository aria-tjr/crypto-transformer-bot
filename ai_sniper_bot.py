import os
import time
import re
import pandas as pd
import numpy as np
import torch
import schedule
from datetime import datetime
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from typing import Dict, List, Optional, Tuple
from models.transformer import BTCTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load Environment Variables
load_dotenv()

# ============================================================
# 🎯 AI SNIPER CONFIGURATION
# ============================================================
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
IS_DEMO = os.getenv("BYBIT_DEMO", "True").lower() == "true"
if not API_KEY or not API_SECRET:
    raise SystemExit("Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables before running.")

# Model Settings
MODEL_PATH = "checkpoints/limit_transformer_best.pt"
CONFIDENCE_THRESHOLD = 0.85  # High confidence for Limit Orders (90% acc model)
MAX_OPEN_POSITIONS = 20      # Maximum simultaneous snipes
LEVERAGE = 5                 # Leverage for snipes
RISK_PER_TRADE = 0.02        # Risk 2% of equity per trade (Stop Loss based)
SYMBOLS_TO_TRADE = []        # Empty = Scan Top 100 Liquid Pairs (Universal Mode)

# Timeframes
TIMEFRAME = "5"              # 5-minute candles for AI input
CONTEXT_WINDOW = 72          # Model needs 72 candles (6 hours)

class AISniperBot:
    def __init__(self):
        print(f"🎯 Initializing Universal AI Sniper Bot (Demo: {IS_DEMO})")
        
        # 1. Connect to Bybit
        self.session = HTTP(
            demo=IS_DEMO,
            api_key=API_KEY,
            api_secret=API_SECRET,
        )
        
        # 2. Load the Brain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # 3. State
        self.open_positions = {} # Cache of open positions
        self.equity = 0.0
        self.position_timers = {} # Track entry time for time-based exits
        
    def _load_model(self):
        """Load the trained Transformer model"""
        print(f"🧠 Loading AI Model from {MODEL_PATH}...")
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            
            # Initialize Model Architecture (Must match training config)
            # 12 Features: log_ret, volatility, rsi, macd_hist, bb_width, bb_pos, atr_pct, vol_ratio, dist_ema50, mom_1h, mom_4h, returns
            model = BTCTransformer(input_dim=12).to(self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            print("✅ AI Model Loaded & Ready to Snipe")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            exit(1)

    def get_top_liquid_symbols(self, n=100) -> List[str]:
        """Get top N liquid symbols to scan"""
        # If specific symbols are configured, use those
        if SYMBOLS_TO_TRADE and len(SYMBOLS_TO_TRADE) > 0:
            print(f"🔒 Restricted to configured symbols: {SYMBOLS_TO_TRADE}")
            return SYMBOLS_TO_TRADE

        try:
            resp = self.session.get_tickers(category="linear")
            if resp['retCode'] != 0: return []
            
            items = resp['result']['list']
            scored = []
            for it in items:
                if not it['symbol'].endswith('USDT'): continue
                turnover = float(it.get('turnover24h', 0))
                scored.append((turnover, it['symbol']))
            
            scored.sort(reverse=True)
            return [s for _, s in scored[:n]]
        except Exception:
            return []

    def fetch_features(self, symbol: str) -> Tuple[Optional[torch.Tensor], float, float, float]:
        """Fetch live data and engineer features for the AI. Returns (Tensor, ATR, EMA, Price)"""
        try:
            # Fetch 1000 candles (Max allowed) to ensure robust normalization stats
            # 1000 candles * 5m = 5000m = ~3.5 days of data
            resp = self.session.get_kline(
                category="linear", symbol=symbol, interval=TIMEFRAME, limit=1000
            )
            if resp['retCode'] != 0: return None, 0.0, 0.0, 0.0
            
            # Parse Data
            klines = resp['result']['list']
            # Bybit returns [ts, open, high, low, close, vol, turnover]
            # Sort ascending (oldest first)
            klines.sort(key=lambda x: int(x[0]))
            
            df = pd.DataFrame(klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turn'])
            df = df.astype(float)
            
            # --- FEATURE ENGINEERING (Must match Training Logic exactly) ---
            # 1. Trend & Momentum
            df['returns'] = df['close'].pct_change()
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['mom_1h'] = df['close'].pct_change(12) # 1 Hour momentum (5m * 12)
            df['mom_4h'] = df['close'].pct_change(48) # 4 Hour momentum
            
            # 2. Volatility
            df['volatility'] = df['log_ret'].rolling(20).std()
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                             abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # 3. Oscillators
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            df['macd_hist'] = macd - signal
            
            # 4. Relative Position
            # Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['bb_width'] = (4 * std20) / sma20
            df['bb_pos'] = (df['close'] - (sma20 - 2*std20)) / (4*std20)
            
            # Distance from EMAs
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['dist_ema50'] = (df['close'] / df['ema_50']) - 1
            
            # 5. Volume
            vol_ma20 = df['vol'].rolling(20).mean()
            df['vol_ratio'] = df['vol'] / (vol_ma20 + 1e-8)
            
            # Drop NaNs
            df = df.dropna()
            
            if len(df) < CONTEXT_WINDOW: return None, 0.0, 0.0, 0.0
            
            # Get ATR for sizing
            current_atr = df['atr'].iloc[-1]
            current_ema = df['ema_50'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Select Features (Exact 12 features used in training)
            feature_cols = ['log_ret', 'volatility', 'rsi', 'macd_hist', 'bb_width', 'bb_pos', 
                            'atr_pct', 'vol_ratio', 'dist_ema50', 'mom_1h', 'mom_4h', 'returns']
            
            # Normalize (Z-Score) using the FULL HISTORY (1000 candles)
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
            
            # Get last sequence (Now normalized correctly)
            sequence = df[feature_cols].iloc[-CONTEXT_WINDOW:].copy()
            
            # Convert to Tensor
            return torch.FloatTensor(sequence.values).unsqueeze(0).to(self.device), current_atr, current_ema, current_price
            
        except Exception as e:
            # print(f"Error fetching {symbol}: {e}")
            return None, 0.0, 0.0, 0.0

    def get_ai_prediction(self, symbol: str) -> Dict:
        """Ask the AI for a prediction"""
        features, atr, ema, price = self.fetch_features(symbol)
        if features is None: return {'class': 1, 'prob': 0.0, 'atr': 0.0, 'ema': 0.0, 'price': 0.0} # Neutral default
        
        with torch.no_grad():
            outputs = self.model(features)
            # Handle both Dict (BTCTransformer) and Tensor (Raw) outputs
            if isinstance(outputs, dict):
                probs = outputs['direction_probs']
            else:
                probs = torch.softmax(outputs, dim=1)
                
            confidence, predicted_class = torch.max(probs, 1)
            
        return {
            'class': predicted_class.item(), # 0=Down, 1=Neutral, 2=Up
            'prob': confidence.item(),
            'probs': probs.cpu().numpy()[0],
            'atr': atr,
            'ema': ema,
            'price': price
        }

    def get_market_regime(self):
        """Check BTC and ETH to determine if we should be longing"""
        btc_pred = self.get_ai_prediction("BTCUSDT")
        eth_pred = self.get_ai_prediction("ETHUSDT")
        
        # If BTC or ETH is Bearish, Market is Unsafe
        if btc_pred['class'] == 0 or eth_pred['class'] == 0:
            return "BEARISH"
        
        # If both are Bullish, Market is Prime
        if btc_pred['class'] == 2 and eth_pred['class'] == 2:
            return "BULLISH"
            
        return "NEUTRAL"

    def execute_snipe(self, symbol: str, side: str, confidence: float, atr: float):
        """Execute a LIMIT Order with ATR-based Stops"""
        print(f"🔫 SNIPING {side.upper()} {symbol} (Conf: {confidence:.1%}, ATR: {atr:.4f})")
        
        # 1. Calculate Size (Risk-Based Sizing)
        self.update_equity()
        print(f"   💰 Equity: ${self.equity:,.2f}")
        
        # RISK MANAGEMENT:
        # We risk exactly 2.0% of equity per trade.
        # Position Size = (Risk Amount) / (Stop Loss Distance %)
        
        RISK_PER_TRADE = 0.02 # 2%
        risk_amount = self.equity * RISK_PER_TRADE
        
        # Get Price & Instrument Info
        ticker = self.session.get_tickers(category="linear", symbol=symbol)['result']['list'][0]
        current_price = float(ticker['lastPrice'])
        
        # LIMIT ORDER PRICING
        # DYNAMIC ENTRY: Use ATR to find the "Real Dip"
        # Instead of fixed 0.2%, we place limit at 2.0x ATR below price.
        # This ensures we only buy significant deviations (Value).
        
        entry_offset = atr * 2.0
        # Clamp offset (Min 0.2%, Max 2.0%)
        min_offset = current_price * 0.002
        max_offset = current_price * 0.02
        entry_offset = max(min_offset, min(entry_offset, max_offset))
        
        if side == "Buy":
            limit_price = current_price - entry_offset
        else:
            limit_price = current_price + entry_offset
            
        print(f"   🎯 Limit Price: {limit_price:.4f} (Current: {current_price:.4f}, Offset: {entry_offset/current_price:.2%})")
        
        # DYNAMIC STOPS (ATR Based)
        # Instead of fixed %, we use volatility (ATR) to set stops.
        # High Volatility = Wider Stops. Low Volatility = Tighter Stops.
        
        # Multipliers for 5m ATR:
        # SL = 6.0x ATR (Generous room for wicks)
        # TP = 10.0x ATR (Targeting the "Rip")
        
        sl_dist = atr * 6.0
        tp_dist = atr * 10.0
        
        # Safety Clamps (Min 0.5%, Max 5%)
        min_dist = limit_price * 0.005
        max_dist = limit_price * 0.05
        
        sl_dist = max(min_dist, min(sl_dist, max_dist))
        
        # Ensure Reward > Risk (1.5 ratio)
        if tp_dist < sl_dist * 1.5:
            tp_dist = sl_dist * 1.5
            
        print(f"   🛡️ Dynamic SL: {sl_dist/limit_price:.2%} | 🎯 Dynamic TP: {tp_dist/limit_price:.2%}")
        
        if side == "Buy":
            sl_price = limit_price - sl_dist
            tp_price = limit_price + tp_dist
        else: # Sell
            sl_price = limit_price + sl_dist
            tp_price = limit_price - tp_dist
        
        # Ensure SL is not negative
        if sl_price < 0: sl_price = limit_price * 0.5

        # CALCULATE POSITION SIZE
        # Size = Risk Amount / Stop Distance
        stop_pct = sl_dist / limit_price
        size_usd = risk_amount / stop_pct
        
        # Cap Max Leverage to 5x
        max_size = self.equity * 5.0
        if size_usd > max_size:
            size_usd = max_size
            
        print(f"   📏 Risk-Based Size: ${size_usd:,.2f} (Risk: ${risk_amount:.2f}, Stop: {stop_pct:.2%})")

        # Get Limits
        try:
            inst_info = self.session.get_instruments_info(category="linear", symbol=symbol)['result']['list'][0]
            lot_filter = inst_info['lotSizeFilter']
            price_filter = inst_info['priceFilter']
            min_qty = float(lot_filter['minOrderQty'])
            max_qty = float(lot_filter['maxOrderQty'])
            qty_step = float(lot_filter['qtyStep'])
            tick_size = float(price_filter['tickSize'])
        except:
            # Fallback defaults
            min_qty = 0.001
            max_qty = 1000000
            qty_step = 0.001
            tick_size = 0.01

        # Round Price to Tick Size
        precision_price = 0
        if tick_size < 1:
            temp_tick = tick_size
            while temp_tick < 1:
                temp_tick *= 10
                precision_price += 1
            limit_price = round(limit_price, precision_price)
        else:
            limit_price = round(limit_price / tick_size) * tick_size

        qty = size_usd / limit_price
        
        # Enforce Max Qty (Cap it to avoid error 10001)
        if qty > max_qty:
            print(f"   ⚠️ Qty {qty} > Max {max_qty}. Capping.")
            qty = max_qty
            
        # Enforce Min Qty
        if qty < min_qty:
            print(f"   ⚠️ Qty {qty} < Min {min_qty}. Skipping.")
            return

        # SPREAD CHECK
        # High spread kills scalping strategies.
        # If spread > 0.05%, skip.
        ticker = self.session.get_tickers(category="linear", symbol=symbol)['result']['list'][0]
        bid = float(ticker['bid1Price'])
        ask = float(ticker['ask1Price'])
        spread_pct = (ask - bid) / current_price
        
        if spread_pct > 0.0005: # 0.05%
            print(f"   ⚠️ Spread {spread_pct:.4%} too high (> 0.05%). Skipping {symbol}.")
            return

        # FUNDING RATE FILTER (Crowded Trade Protection)
        # If Funding is High Positive (> 0.02%), Longs are crowded/expensive.
        # If Funding is High Negative (< -0.02%), Shorts are crowded/expensive.
        funding_rate = float(ticker.get('fundingRate', 0))
        
        if side == "Buy" and funding_rate > 0.0002:
            print(f"   ⚠️ Funding High ({funding_rate:.4%}). Crowded Longs. Skipping.")
            return
        if side == "Sell" and funding_rate < -0.0002:
            print(f"   ⚠️ Funding Low ({funding_rate:.4%}). Crowded Shorts. Skipping.")
            return

        # VOLATILITY FLOOR (Dead Coin Protection)
        # If ATR is < 0.3% of price, the coin is not moving enough to profit.
        atr_pct = atr / current_price
        if atr_pct < 0.003:
            print(f"   💤 Volatility Low ({atr_pct:.2%}). Skipping dead coin.")
            return

        # Round to Step
        precision = 0
        if qty_step < 1:
            temp_step = qty_step
            while temp_step < 1:
                temp_step *= 10
                precision += 1
            qty = round(qty, precision)
        else:
            qty = int(qty // qty_step * qty_step)
        
        print(f"   🔢 Qty: {qty} @ {limit_price} (Max: {max_qty})")

        # TRAILING STOP (Lock in Profits)
        # If price moves 4x ATR in our favor, we activate a 2x ATR trailing stop.
        # This ensures we catch the "Rip" but don't give it all back.
        
        trailing_dist = str(round(atr * 2.0, 4))
        activation_price = 0
        
        if side == "Buy":
            activation_price = limit_price + (atr * 4.0)
        else:
            activation_price = limit_price - (atr * 4.0)
            
        activation_price = str(round(activation_price, 4))

        try:
            # Place LIMIT Order with Trailing Stop
            self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(limit_price),
                timeInForce="GTC", # Good Till Cancelled
                isLeverage=1, 
                stopLoss=str(round(sl_price, 4)),
                takeProfit=str(round(tp_price, 4)),
                slTriggerBy="LastPrice",
                tpTriggerBy="LastPrice",
                trailingStop=trailing_dist,
                activePrice=activation_price
            )
            print(f"✅ LIMIT ORDER PLACED: {side} {qty} {symbol} @ {limit_price}")
            
            # Track Entry Time
            self.position_timers[symbol] = time.time()
            
        except Exception as e:
            print(f"❌ Execution Failed: {e}")
            # Auto-Retry with Corrected Quantity if Limit Exceeded
            if "exceeds maximum limit allowed" in str(e):
                try:
                    # Extract max_qty from error message
                    match = re.search(r"max_qty:(\d+)", str(e))
                    if match:
                        raw_max = float(match.group(1))
                        real_max_qty = raw_max / 100000000.0
                        
                        print(f"   ⚠️ Limit exceeded. Max allowed: {real_max_qty}. Retrying with 95% of limit...")
                        new_qty = real_max_qty * 0.95
                        
                        if qty_step < 1:
                            new_qty = round(new_qty, precision)
                        else:
                            new_qty = int(new_qty // qty_step * qty_step)
                            
                        self.session.place_order(
                            category="linear",
                            symbol=symbol,
                            side=side,
                            orderType="Limit",
                            qty=str(new_qty),
                            price=str(limit_price),
                            timeInForce="GTC",
                            isLeverage=1, 
                            stopLoss=str(round(sl_price, 4)),
                            takeProfit=str(round(tp_price, 4)),
                            slTriggerBy="LastPrice",
                            tpTriggerBy="LastPrice"
                        )
                        print(f"✅ RETRY LIMIT PLACED: {side} {new_qty} {symbol}")
                        self.position_timers[symbol] = time.time()
                        
                except Exception as retry_e:
                    print(f"   ❌ Retry Failed: {retry_e}")

    def update_equity(self):
        try:
            resp = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            self.equity = float(resp['result']['list'][0]['totalEquity'])
        except:
            pass

    def manage_positions(self):
        """Smart Exit Logic & Order Management"""
        print("🔍 Managing Positions & Orders...")
        try:
            # 1. Manage Active Orders (Cancel Stale Limits)
            resp_orders = self.session.get_open_orders(category="linear", settleCoin="USDT")
            orders = resp_orders['result']['list']
            
            for order in orders:
                symbol = order['symbol']
                order_id = order['orderId']
                
                # Check age
                created_time = float(order['createdTime']) / 1000.0
                age = time.time() - created_time
                
                # If order is older than 60 mins, cancel it (The dip didn't happen)
                if age > 3600:
                    print(f"⏰ Cancelling Stale Limit Order for {symbol} (Age: {age/60:.1f}m)")
                    self.session.cancel_order(category="linear", symbol=symbol, orderId=order_id)

            # 2. Manage Open Positions
            resp = self.session.get_positions(category="linear", settleCoin="USDT")
            positions = resp['result']['list']
            
            for pos in positions:
                if float(pos['size']) == 0: continue
                
                symbol = pos['symbol']
                side = pos['side'] # "Buy" or "Sell"
                pred = self.get_ai_prediction(symbol)
                
                # TACTICAL RETREAT
                # Only exit if AI is confident in the reversal (> 60%)
                # This prevents "churning" on weak signals
                EXIT_CONFIDENCE = 0.60
                
                # TIME-BASED EXIT (Stale Prediction)
                # If trade is older than 60 mins (3600s), close it.
                # Use API createdTime if local timer is missing
                entry_time = self.position_timers.get(symbol, 0)
                if entry_time == 0:
                    try:
                        # createdTime is in ms string
                        entry_time = float(pos['createdTime']) / 1000.0
                        self.position_timers[symbol] = entry_time
                    except:
                        pass
                
                if entry_time > 0 and (time.time() - entry_time) > 3600:
                    print(f"⏰ TIME LIMIT: {symbol} held > 60 mins. Closing stale trade.")
                    self.session.place_order(category="linear", symbol=symbol, side="Sell" if side=="Buy" else "Buy", orderType="Market", qty=pos['size'], reduceOnly=True)
                    del self.position_timers[symbol]
                    continue

                # TRAILING STOP (Aggressive Scalping)
                # Handled by Bybit Native Trailing Stop now.
                # Just monitoring here.
                try:
                    entry_price = float(pos['avgPrice'])
                    mark_price = float(pos['markPrice'])
                    if side == "Buy":
                        pnl_pct = (mark_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - mark_price) / entry_price
                    
                    # print(f"   � {symbol} PnL: {pnl_pct:.2%}")
                except:
                    pass

                # If Long and AI says Down (0) -> Close
                if side == "Buy" and pred['class'] == 0 and pred['prob'] > EXIT_CONFIDENCE:
                    print(f"🚨 TACTICAL RETREAT: {symbol} Long -> AI Bearish ({pred['prob']:.1%}). Closing.")
                    self.session.place_order(category="linear", symbol=symbol, side="Sell", orderType="Market", qty=pos['size'], reduceOnly=True)
                
                # If Short and AI says Up (2) -> Close
                elif side == "Sell" and pred['class'] == 2 and pred['prob'] > EXIT_CONFIDENCE:
                    print(f"🚨 TACTICAL RETREAT: {symbol} Short -> AI Bullish ({pred['prob']:.1%}). Closing.")
                    self.session.place_order(category="linear", symbol=symbol, side="Buy", orderType="Market", qty=pos['size'], reduceOnly=True)

                
        except Exception as e:
            print(f"Error managing positions: {e}")

    def manage_stale_orders(self):
        """Cancel limit orders that haven't filled within 60 minutes"""
        try:
            resp = self.session.get_open_orders(category="linear", settleCoin="USDT")
            if resp['retCode'] != 0: return
            
            orders = resp['result']['list']
            now = time.time() * 1000 # ms
            
            for order in orders:
                # createdTime is string in ms
                created_time = float(order['createdTime'])
                age_minutes = (now - created_time) / 1000 / 60
                
                if age_minutes > 60:
                    print(f"♻️ Cancelling STALE Order: {order['symbol']} (Age: {age_minutes:.1f}m)")
                    self.session.cancel_order(category="linear", symbol=order['symbol'], orderId=order['orderId'])
                    
        except Exception as e:
            print(f"Error managing orders: {e}")

    def run(self):
        print("🚀 AI Sniper Bot Started (Enhanced: Multi-Threaded + Shorting + ATR Stops)")
        
        while True:
            try:
                print(f"\n⏰ Scan Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # 1. Check Market Regime
                regime = self.get_market_regime()
                print(f"🌍 Market Regime: {regime}")
                
                # 2. Scan Top 100 Symbols (Parallel)
                # BLACKLIST: Ignore stablecoins and problematic pairs
                BLACKLIST = ["USDCUSDT", "BUSDUSDT", "DAIUSDT", "USDEUSDT"]
                
                symbols = self.get_top_liquid_symbols(100)
                symbols = [s for s in symbols if s not in BLACKLIST]
                
                print(f"👀 Scanning {len(symbols)} symbols in parallel...")
                
                opportunities = []
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_symbol = {executor.submit(self.get_ai_prediction, sym): sym for sym in symbols}
                    for future in as_completed(future_to_symbol):
                        sym = future_to_symbol[future]
                        try:
                            pred = future.result()
                            
                            # Logic: Class 2 is UP, Class 0 is DOWN
                            # REGIME FILTER: Don't fight the trend.
                            # If Regime is BEARISH, ignore Longs.
                            # If Regime is BULLISH, ignore Shorts.
                            
                            if pred['class'] == 2 and pred['prob'] > CONFIDENCE_THRESHOLD:
                                if regime == "BEARISH":
                                    # print(f"   ⚠️ Skipping LONG {sym} (Market is Bearish)")
                                    pass
                                else:
                                    print(f"   ✨ LONG Signal: {sym} (Conf: {pred['prob']:.1%})")
                                    opportunities.append((pred['prob'], sym, "Buy", pred['atr']))
                                
                            elif pred['class'] == 0 and pred['prob'] > CONFIDENCE_THRESHOLD:
                                if regime == "BULLISH":
                                    # print(f"   ⚠️ Skipping SHORT {sym} (Market is Bullish)")
                                    pass
                                else:
                                    print(f"   ✨ SHORT Signal: {sym} (Conf: {pred['prob']:.1%})")
                                    opportunities.append((pred['prob'], sym, "Sell", pred['atr']))
                                
                        except Exception as exc:
                            print(f"   ❌ Error scanning {sym}: {exc}")
                
                # 3. Execute Best Snipes
                opportunities.sort(reverse=True) # Highest confidence first
                
                # Check current open positions AND active orders
                resp_pos = self.session.get_positions(category="linear", settleCoin="USDT")
                open_positions = [p for p in resp_pos['result']['list'] if float(p['size']) > 0]
                
                resp_ord = self.session.get_open_orders(category="linear", settleCoin="USDT")
                open_orders = resp_ord['result']['list']
                
                current_count = len(open_positions) + len(open_orders)
                held_symbols = {p['symbol'] for p in open_positions}
                ordered_symbols = {o['symbol'] for o in open_orders}
                
                # CORRELATION FILTER
                # Count Longs vs Shorts to prevent over-exposure to one side
                long_count = sum(1 for p in open_positions if p['side'] == 'Buy')
                short_count = sum(1 for p in open_positions if p['side'] == 'Sell')
                
                MAX_SIDE_EXPOSURE = 12 # Max 12 Longs or 12 Shorts
                
                slots_available = MAX_OPEN_POSITIONS - current_count
                
                if slots_available > 0 and opportunities:
                    print(f"⚡ Found {len(opportunities)} signals. Slots available: {slots_available}")
                    executed_count = 0
                    for conf, sym, side, atr in opportunities:
                        if executed_count >= slots_available: break
                        
                        if sym in held_symbols or sym in ordered_symbols:
                            # print(f"   ⚠️ Already active in {sym}. Skipping.")
                            continue
                            
                        # Check Correlation Limits
                        if side == "Buy" and long_count >= MAX_SIDE_EXPOSURE:
                            # print(f"   ⚠️ Max Longs ({long_count}) reached. Skipping {sym}.")
                            continue
                        if side == "Sell" and short_count >= MAX_SIDE_EXPOSURE:
                            # print(f"   ⚠️ Max Shorts ({short_count}) reached. Skipping {sym}.")
                            continue
                            
                        self.execute_snipe(sym, side, conf, atr)
                        executed_count += 1
                        held_symbols.add(sym) # Prevent double entry in same loop
                        if side == "Buy": long_count += 1
                        else: short_count += 1
                        
                elif slots_available <= 0:
                    print("🔒 Max positions/orders reached. No new snipes.")
                else:
                    print("💤 No high-confidence signals found.")
                
                # 4. Manage Exits
                self.manage_positions()
                self.manage_stale_orders() # <--- Add this line
                
                print("⏳ Sleeping 30 seconds...")
                time.sleep(30)
                
            except KeyboardInterrupt:
                print("👋 Stopping Bot")
                break
            except Exception as e:
                print(f"❌ Loop Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = AISniperBot()
    bot.run()
