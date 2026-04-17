"""Quick hyperparameter search script."""
import sys
sys.path.insert(0, '/Users/ariatajeri/trading_bot')

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from config.settings import settings
from training.hyperparam_search import HyperparameterSearch, SearchConfig, SearchSpace
import time

print("Starting hyperparameter search...")

# Fetch data
print("Fetching data from Bybit...")
session = HTTP(
    testnet=False,
    api_key=settings.bybit.api_key,
    api_secret=settings.bybit.api_secret
)

all_data = []
for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    print(f"Fetching {symbol}...")
    symbol_klines = []
    end_time = None
    target_candles = 25920  # 90 days

    while len(symbol_klines) < target_candles:
        params = {"category": "linear", "symbol": symbol, "interval": "5", "limit": 1000}
        if end_time:
            params["end"] = end_time
        response = session.get_kline(**params)
        if response['retCode'] != 0 or not response['result']['list']:
            break
        batch = response['result']['list']
        symbol_klines.extend(batch)
        oldest_ts = int(batch[-1][0])
        end_time = oldest_ts - 1
        time.sleep(0.1)

    if symbol_klines:
        df = pd.DataFrame(symbol_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

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
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_6'] = df['close'].pct_change(6)
        df['momentum_12'] = df['close'].pct_change(12)
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        df['price_sma10_ratio'] = df['close'] / (df['sma_10'] + 1e-8) - 1
        df['price_sma20_ratio'] = df['close'] / (df['sma_20'] + 1e-8) - 1
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-8)

        all_data.append(df.dropna())
        print(f"  {symbol}: {len(df.dropna())} candles")

# Combine and prepare features
combined = pd.concat(all_data, ignore_index=True)
feature_cols = ['close', 'returns', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_position', 'bb_width', 'atr_pct', 'momentum_3', 'momentum_6', 'momentum_12',
                'volume_ratio', 'price_sma10_ratio', 'price_sma20_ratio']

# Normalize
for col in feature_cols[1:]:
    if col in combined.columns:
        mean = combined[col].mean()
        std = combined[col].std()
        if std > 0:
            combined[col] = (combined[col] - mean) / std

features = combined[feature_cols].values
target_dim = 64
if features.shape[1] < target_dim:
    padding = np.zeros((len(features), target_dim - features.shape[1]))
    windows = [3, 6, 12, 24, 48]
    for i in range(target_dim - features.shape[1]):
        window = windows[i % len(windows)]
        ret = combined['close'].pct_change(window).fillna(0).values
        std = np.std(ret)
        if std > 0:
            ret = ret / std
        padding[:, i] = ret
    features = np.hstack([features, padding])

data = features.astype(np.float32)
print(f"Total data shape: {data.shape}")

# Prepare sequences
from training.innovative_trainer import DataPreparer
preparer = DataPreparer(sequence_length=48, prediction_horizon=3)
X, y = preparer.prepare_sequences(data)
print(f"Sequences: X={X.shape}, y={y.shape}")

# Quick search with limited trials
search_space = SearchSpace(
    model_type=['transformer_gru', 'tcn_attention'],
    d_model=[192, 256],
    n_layers=[3, 4],
    n_heads=[8],
    dropout=[0.15, 0.2],
    learning_rate=[3e-4, 5e-4],
    batch_size=[64],
    weight_decay=[1e-5],
    label_smoothing=[0.1]
)

config = SearchConfig(
    search_space=search_space,
    max_trials=12,  # 12 trials for quick search
    epochs_per_trial=25,  # 25 epochs per trial
    n_folds=2  # 2-fold CV
)

searcher = HyperparameterSearch(config, device='mps')
results = searcher.search(X, y, method='random')

print("\n=== SEARCH COMPLETE ===")
print(f"Best config: {results['best_config']}")
print(f"Best accuracy: {results['best_metric']:.4f}")
