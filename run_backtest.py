"""
Backtest with trained Transformer-GRU model.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time

from pybit.unified_trading import HTTP
from models.transformer_gru import TransformerGRU, TransformerGRUConfig
from training.backtest import BacktestEngine, BacktestConfig
from utils.metrics import format_metrics_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
logger.info(f"Device: {device}")

# Load trained model from trading_bot folder
checkpoint_path = Path("transformer_best.pt")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
logger.info(f"Loaded checkpoint - Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")

# Model config (must match training)
model_config = TransformerGRUConfig(
    input_dim=32,
    d_model=256,
    n_heads=8,
    n_encoder_layers=3,
    d_ff=1024,
    dropout=0.15,
    gru_hidden=256,
    output_dim=3
)

model = TransformerGRU(model_config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()
logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Fetch data
logger.info("Fetching data from Bybit...")
session = HTTP(testnet=False)

klines = []
end_time = None
target = 10000  # ~35 days for backtesting

while len(klines) < target:
    params = {'category': 'linear', 'symbol': 'BTCUSDT', 'interval': '5', 'limit': 1000}
    if end_time:
        params['end'] = end_time
    resp = session.get_kline(**params)
    if resp['retCode'] != 0 or not resp['result']['list']:
        break
    batch = resp['result']['list']
    klines.extend(batch)
    end_time = int(batch[-1][0]) - 1
    time.sleep(0.05)

df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)

# Features
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(20).std()
df['rsi'] = 100 - 100/(1 + df['close'].diff().clip(lower=0).rolling(14).mean() /
                      (df['close'].diff().clip(upper=0).abs().rolling(14).mean() + 1e-8))
df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_position'] = (df['close'] - (df['bb_mid'] - 2*df['bb_std'])) / (4*df['bb_std'] + 1e-8)
df['bb_width'] = 4*df['bb_std'] / (df['bb_mid'] + 1e-8)
df['atr'] = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(),
                      (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
df['atr_pct'] = df['atr'] / df['close']
df['momentum_3'] = df['close'].pct_change(3)
df['momentum_6'] = df['close'].pct_change(6)
df['momentum_12'] = df['close'].pct_change(12)
df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
df['price_sma10_ratio'] = df['close'] / df['close'].rolling(10).mean() - 1
df['price_sma20_ratio'] = df['close'] / df['close'].rolling(20).mean() - 1

btc_data = df.dropna().reset_index(drop=True)
logger.info(f"BTCUSDT: {len(btc_data)} bars ({btc_data['timestamp'].min()} to {btc_data['timestamp'].max()})")

# Prepare features
feature_cols = ['close', 'returns', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_position', 'bb_width', 'atr_pct', 'momentum_3', 'momentum_6', 'momentum_12',
                'volume_ratio', 'price_sma10_ratio', 'price_sma20_ratio']

# Store normalization params
feature_means = {col: btc_data[col].mean() for col in feature_cols[1:]}
feature_stds = {col: btc_data[col].std() for col in feature_cols[1:]}

def prepare_sequence(data, idx, seq_len=48):
    """Prepare a single sequence for model input."""
    if idx < seq_len:
        return None

    seq_data = data.iloc[idx-seq_len:idx][feature_cols].copy()

    # Normalize
    for col in feature_cols[1:]:
        if feature_stds[col] > 0:
            seq_data[col] = (seq_data[col] - feature_means[col]) / feature_stds[col]

    features = seq_data.values

    # Pad to 32 dims if needed
    if features.shape[1] < 32:
        pad = np.zeros((features.shape[0], 32 - features.shape[1]))
        features = np.hstack([features, pad])

    return features.astype(np.float32)

# Signal function using model
def model_signal_func(data: pd.DataFrame, bar: int) -> dict:
    """Generate signals from model predictions."""
    seq = prepare_sequence(data, bar)
    if seq is None:
        return {'BTCUSDT': 0}

    with torch.no_grad():
        x = torch.FloatTensor(seq).unsqueeze(0).to(device)
        output = model(x)
        probs = output['direction_probs'][0].cpu().numpy()

    # Direction: 0=down, 1=neutral, 2=up
    pred = np.argmax(probs)

    # Only trade if confident enough
    if probs[pred] < 0.45:
        return {'BTCUSDT': 0}

    if pred == 2:  # Up
        return {'BTCUSDT': 1}
    elif pred == 0:  # Down
        return {'BTCUSDT': -1}
    else:
        return {'BTCUSDT': 0}

# Backtest config
config = BacktestConfig(
    initial_capital=10000.0,
    maker_fee_bps=1.0,
    taker_fee_bps=6.0,
    slippage_bps=5.0,
    max_position_pct=0.2,
    max_positions=1,
    use_stop_loss=True,
    stop_loss_pct=0.015,
    use_take_profit=True,
    take_profit_pct=0.03,
)

# Run backtest
logger.info("Running backtest...")
engine = BacktestEngine(config)
result = engine.run(btc_data, model_signal_func)

# Print results
print("\n" + "="*60)
print("BACKTEST RESULTS - BTCUSDT")
print("="*60)
print(f"Period: {result.start_time} to {result.end_time}")
print(f"Bars: {result.n_bars} (5m candles)")
print(f"Initial Capital: ${config.initial_capital:,.0f}")
print(f"Final Equity: ${result.equity_curve[-1]:,.2f}")
print("="*60)
print(format_metrics_report(result.metrics))
print("="*60)

# Trade summary
if result.trades:
    wins = [t for t in result.trades if t.pnl > 0]
    losses = [t for t in result.trades if t.pnl <= 0]
    print(f"\nTrade Summary:")
    print(f"  Total Trades: {len(result.trades)}")
    print(f"  Winning: {len(wins)} ({len(wins)/len(result.trades)*100:.1f}%)")
    print(f"  Losing: {len(losses)} ({len(losses)/len(result.trades)*100:.1f}%)")
    if wins:
        print(f"  Avg Win: ${np.mean([t.pnl for t in wins]):.2f}")
    if losses:
        print(f"  Avg Loss: ${np.mean([t.pnl for t in losses]):.2f}")

# Save results
results_dict = {
    'total_return': float(result.metrics.total_return),
    'sharpe_ratio': float(result.metrics.sharpe_ratio),
    'sortino_ratio': float(result.metrics.sortino_ratio),
    'max_drawdown': float(result.metrics.max_drawdown),
    'win_rate': float(result.metrics.win_rate),
    'profit_factor': float(result.metrics.profit_factor),
    'num_trades': int(result.metrics.num_trades),
    'final_equity': float(result.equity_curve[-1]),
    'start_time': str(result.start_time),
    'end_time': str(result.end_time)
}

with open('backtest_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
logger.info("Results saved to backtest_results.json")

# Plot equity curve
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Equity curve
    axes[0].plot(result.equity_curve, label='Equity', color='blue')
    axes[0].axhline(y=config.initial_capital, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Equity Curve (Return: {result.metrics.total_return:.2%})')
    axes[0].set_ylabel('Equity ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    axes[1].fill_between(range(len(result.drawdown_curve)), -result.drawdown_curve * 100,
                         alpha=0.5, color='red')
    axes[1].set_title(f'Drawdown (Max: {result.metrics.max_drawdown:.2%})')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(True, alpha=0.3)

    # Trade returns
    if result.trades:
        trade_returns = [t.return_pct * 100 for t in result.trades]
        colors = ['green' if r > 0 else 'red' for r in trade_returns]
        axes[2].bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title(f'Trade Returns (Win Rate: {result.metrics.win_rate:.1%})')
        axes[2].set_xlabel('Trade #')
        axes[2].set_ylabel('Return (%)')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_chart.png', dpi=150)
    logger.info("Chart saved to backtest_chart.png")
    plt.show()
except Exception as e:
    logger.warning(f"Could not create chart: {e}")
