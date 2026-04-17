"""
Trading Bot Main Entry Point.

Modes:
- train: Train models on historical data
- backtest: Backtest strategy on historical data
- paper: Paper trading on live data (Bybit testnet)
- live: Live trading (requires explicit confirmation)
"""
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import signal
import logging

# Local imports
from config.settings import settings
from config.hyperparameters import hyperparams
from utils.logging import setup_logging, TradingLogger
from utils.device import device_manager
from data.collectors.bybit_ws import BybitWebSocketClient
from data.collectors.onchain import OnChainCollector
from data.collectors.sentiment import SentimentCollector
from data.processors.orderbook import OrderBookProcessor
from data.processors.features import FeatureProcessor
from data.storage.timeseries_db import TimeSeriesDB
from models.transformer_gru import TransformerGRU, TransformerGRUConfig
from agents.ppo_agent import PPOAgent, PPOConfig
from agents.environment import TradingEnvironment, TradingConfig
from execution.order_manager import BybitOrderManager, OrderManagerConfig
from execution.risk_manager import RiskManager, RiskConfig
from execution.position_sizer import PositionSizer, PositionSizerConfig
from training.innovative_trainer import InnovativeTrainer, TrainerConfig


class TradingBot:
    """
    Main trading bot orchestrator.

    Coordinates:
    - Data collection
    - Feature processing
    - Model inference
    - Trade execution
    - Risk management
    """

    def __init__(self, mode: str, logger: TradingLogger):
        self.mode = mode
        self.logger = logger
        self.log = logger.get_logger("main")

        self._running = False

        # Components
        self.ws_client: BybitWebSocketClient = None
        self.onchain_collector: OnChainCollector = None
        self.sentiment_collector: SentimentCollector = None
        self.db: TimeSeriesDB = None
        self.orderbook_processor: OrderBookProcessor = None
        self.feature_processor: FeatureProcessor = None
        self.model: TransformerGRU = None
        self.agent: PPOAgent = None
        self.order_manager: BybitOrderManager = None
        self.risk_manager: RiskManager = None
        self.position_sizer: PositionSizer = None

    async def initialize(self):
        """Initialize all components."""
        self.log.info(f"Initializing trading bot in {self.mode} mode...")
        self.log.info(f"Device: {device_manager.device}")

        # Initialize database
        self.db = TimeSeriesDB(settings.data.db_path)

        # Initialize processors
        self.orderbook_processor = OrderBookProcessor(
            depth=settings.trading.orderbook_depth
        )
        self.feature_processor = FeatureProcessor()

        # Initialize WebSocket client
        self.ws_client = BybitWebSocketClient(
            symbols=settings.trading.symbols,
            mode=settings.bybit.mode,
            orderbook_depth=settings.trading.orderbook_depth
        )

        # Initialize collectors
        self.onchain_collector = OnChainCollector(
            etherscan_api_key=settings.data.etherscan_api_key
        )
        self.sentiment_collector = SentimentCollector()

        # Initialize risk management
        self.risk_manager = RiskManager(
            initial_capital=hyperparams.backtest.initial_capital,
            config=RiskConfig(
                max_daily_drawdown_pct=settings.risk.max_daily_drawdown_pct,
                max_total_drawdown_pct=settings.risk.max_total_drawdown_pct,
                stop_loss_atr_multiplier=settings.risk.stop_loss_atr_multiplier
            )
        )

        self.position_sizer = PositionSizer(
            PositionSizerConfig(
                use_kelly=settings.risk.use_kelly,
                kelly_fraction=settings.risk.kelly_fraction,
                max_position_pct=settings.trading.max_position_pct
            )
        )

        # Initialize order manager for paper/live trading
        if self.mode in ["paper", "live"]:
            self.order_manager = BybitOrderManager(
                OrderManagerConfig(
                    api_key=settings.bybit.api_key,
                    api_secret=settings.bybit.api_secret,
                    mode=settings.bybit.mode
                )
            )

        self.log.info("Initialization complete")

    async def run_training(self):
        """Run model training."""
        self.log.info("Starting training pipeline...")

        trainer = InnovativeTrainer(
            TrainerConfig(
                checkpoint_dir=settings.data.data_dir / "checkpoints",
                log_dir=settings.logging.log_dir
            ),
            device=str(device_manager.device)
        )

        # Load historical data from Bybit
        self.log.info("Fetching historical data from Bybit...")

        import numpy as np
        import pandas as pd
        from pybit.unified_trading import HTTP

        # Initialize Bybit client
        session = HTTP(
            testnet=False,  # Use production API for historical data
            api_key=settings.bybit.api_key,
            api_secret=settings.bybit.api_secret
        )

        all_data = []
        for symbol in settings.trading.symbols:
            self.log.info(f"Fetching {symbol} klines...")
            try:
                # Paginate to get more data (90 days = ~25920 candles at 5m)
                # Bybit returns max 1000 per request, so we need multiple calls
                symbol_klines = []
                end_time = None  # Start from most recent
                target_candles = 25920  # ~90 days of 5m candles

                while len(symbol_klines) < target_candles:
                    params = {
                        "category": "linear",
                        "symbol": symbol,
                        "interval": "5",
                        "limit": 1000
                    }
                    if end_time:
                        params["end"] = end_time

                    response = session.get_kline(**params)

                    if response['retCode'] != 0 or not response['result']['list']:
                        break

                    batch = response['result']['list']
                    symbol_klines.extend(batch)

                    # Get the oldest timestamp for next pagination
                    oldest_ts = int(batch[-1][0])
                    end_time = oldest_ts - 1  # Go further back

                    # Rate limit protection
                    import time
                    time.sleep(0.1)

                if symbol_klines:
                    df = pd.DataFrame(symbol_klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                    ])
                    df = df.astype({
                        'open': float, 'high': float, 'low': float,
                        'close': float, 'volume': float
                    })
                    # Remove duplicates
                    df = df.drop_duplicates(subset=['timestamp'])

                    # Sort by timestamp (oldest first)
                    df = df.sort_values('timestamp').reset_index(drop=True)

                    # Calculate technical features for 5m timeframe
                    df['returns'] = df['close'].pct_change()
                    df['volatility'] = df['returns'].rolling(20).std()
                    df['sma_10'] = df['close'].rolling(10).mean()
                    df['sma_20'] = df['close'].rolling(20).mean()
                    df['sma_50'] = df['close'].rolling(50).mean()
                    df['rsi'] = self._calculate_rsi(df['close'], 14)
                    df['volume_ma'] = df['volume'].rolling(20).mean()

                    # NEW: ATR (Average True Range) for volatility
                    df['tr'] = np.maximum(
                        df['high'] - df['low'],
                        np.maximum(
                            abs(df['high'] - df['close'].shift(1)),
                            abs(df['low'] - df['close'].shift(1))
                        )
                    )
                    df['atr'] = df['tr'].rolling(14).mean()
                    df['atr_pct'] = df['atr'] / df['close']  # ATR as % of price

                    # NEW: Price momentum (rate of change)
                    df['momentum_3'] = df['close'].pct_change(3)   # 15 min
                    df['momentum_6'] = df['close'].pct_change(6)   # 30 min
                    df['momentum_12'] = df['close'].pct_change(12)  # 1 hour

                    # NEW: Volume ratio (current vs average)
                    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)

                    # NEW: Price position relative to SMAs
                    df['price_sma10_ratio'] = df['close'] / (df['sma_10'] + 1e-8) - 1
                    df['price_sma20_ratio'] = df['close'] / (df['sma_20'] + 1e-8) - 1

                    # MACD for 5m timeframe
                    df['ema_12'] = df['close'].ewm(span=12).mean()
                    df['ema_26'] = df['close'].ewm(span=26).mean()
                    df['macd'] = df['ema_12'] - df['ema_26']
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']  # NEW: MACD histogram

                    # Bollinger Bands
                    df['bb_mid'] = df['close'].rolling(20).mean()
                    df['bb_std'] = df['close'].rolling(20).std()
                    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
                    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
                    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-8)  # NEW: BB width

                    all_data.append(df.dropna())
                    self.log.info(f"  {symbol}: {len(df.dropna())} 5m candles fetched")
            except Exception as e:
                self.log.warning(f"Failed to fetch {symbol}: {e}")

        if not all_data:
            self.log.error("No data fetched, falling back to synthetic data")
            data = np.random.randn(5000, hyperparams.transformer.input_dim).astype(np.float32)
        else:
            # Combine and prepare features
            combined = pd.concat(all_data, ignore_index=True)

            # Feature columns for 5m scalping - EXPANDED feature set
            # First column is 'close' for RL environment price reference
            feature_cols = [
                'close',           # Price for RL
                'returns',         # Price momentum
                'volatility',      # Risk measure
                'rsi',             # Overbought/oversold
                'macd',            # Trend strength
                'macd_signal',     # MACD crossover signal
                'macd_hist',       # MACD histogram (momentum)
                'bb_position',     # Position within Bollinger Bands (0-1)
                'bb_width',        # BB width (volatility indicator)
                'atr_pct',         # ATR as % of price
                'momentum_3',      # 15 min momentum
                'momentum_6',      # 30 min momentum
                'momentum_12',     # 1 hour momentum
                'volume_ratio',    # Volume vs average
                'price_sma10_ratio',  # Price vs SMA10
                'price_sma20_ratio',  # Price vs SMA20
            ]

            # Normalize features (except close which RL needs raw)
            for col in feature_cols[1:]:  # Skip 'close'
                if col in combined.columns:
                    mean = combined[col].mean()
                    std = combined[col].std()
                    if std > 0:
                        combined[col] = (combined[col] - mean) / std

            # Build feature matrix
            features = combined[feature_cols].values
            n_features = features.shape[1]
            target_dim = hyperparams.transformer.input_dim

            if n_features < target_dim:
                # Add multi-timeframe returns as additional features
                padding = np.zeros((len(features), target_dim - n_features))
                windows = [3, 6, 12, 24, 48]  # 15m, 30m, 1h, 2h, 4h returns
                for i in range(target_dim - n_features):
                    window = windows[i % len(windows)]
                    ret = combined['close'].pct_change(window).fillna(0).values
                    # Normalize
                    std = np.std(ret)
                    if std > 0:
                        ret = ret / std
                    padding[:, i] = ret
                features = np.hstack([features, padding])

            data = features.astype(np.float32)
            self.log.info(f"Training data shape: {data.shape} (5m candles with {n_features} core features)")

        # Run training
        self.log.info("Starting model training...")
        results = trainer.train_full_pipeline(data)

        self.log.info("Training complete")
        self.log.info(f"Results: {results}")

        return results

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    async def run_backtest(self):
        """Run backtesting."""
        self.log.info("Starting backtest...")

        from training.backtest import BacktestEngine, BacktestConfig

        config = BacktestConfig(
            initial_capital=hyperparams.backtest.initial_capital,
            maker_fee_bps=hyperparams.backtest.maker_fee_bps,
            taker_fee_bps=hyperparams.backtest.taker_fee_bps,
            slippage_bps=hyperparams.backtest.slippage_bps
        )

        # Load data
        import pandas as pd
        import numpy as np

        # Dummy data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=10000, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(10000).cumsum() + 100,
            'high': np.random.randn(10000).cumsum() + 101,
            'low': np.random.randn(10000).cumsum() + 99,
            'close': np.random.randn(10000).cumsum() + 100,
            'volume': np.abs(np.random.randn(10000)) * 1000
        })

        # Simple signal function (replace with model inference)
        def signal_func(df, bar):
            if bar < 10:
                return {}
            returns = df['close'].iloc[bar-10:bar].pct_change().mean()
            if returns > 0.001:
                return {'BTCUSDT': 1}
            elif returns < -0.001:
                return {'BTCUSDT': -1}
            return {'BTCUSDT': 0}

        engine = BacktestEngine(config)
        result = engine.run(data, signal_func)

        self.log.info("Backtest complete")
        self.log.info(f"Total Return: {result.metrics.total_return:.2%}")
        self.log.info(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        self.log.info(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")

        return result

    async def run_paper_trading(self):
        """Run paper trading on testnet."""
        self.log.info("Starting paper trading...")

        # Connect to WebSocket
        ws_task = asyncio.create_task(self.ws_client.connect())

        # Register callbacks
        self.ws_client.on_orderbook(self._on_orderbook)
        self.ws_client.on_trade(self._on_trade)

        try:
            # Main trading loop
            while self._running:
                # Check risk status
                can_trade, reason = self.risk_manager.check_can_trade()
                if not can_trade:
                    self.log.warning(f"Trading paused: {reason}")
                    await asyncio.sleep(60)
                    continue

                # Collect external data periodically
                await self._collect_external_data()

                # Sleep between iterations
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.log.info("Paper trading cancelled")
        finally:
            await self.ws_client.disconnect()
            ws_task.cancel()

    def _on_orderbook(self, snapshot):
        """Handle orderbook update."""
        features = self.orderbook_processor.process(snapshot)

        # Store in database
        self.db.insert_orderbook_features(features)

        # Check for trading signals
        signal = self.orderbook_processor.get_imbalance_signal(
            snapshot.symbol,
            threshold=0.3
        )

        if signal != 0:
            self.log.debug(f"Imbalance signal: {snapshot.symbol} -> {signal}")

    def _on_trade(self, trade):
        """Handle trade update."""
        # Update feature processor
        self.feature_processor.update(
            symbol=trade.symbol,
            timestamp=trade.timestamp,
            close=trade.price,
            high=trade.price,
            low=trade.price,
            volume=trade.size
        )

    async def _collect_external_data(self):
        """Collect on-chain and sentiment data."""
        try:
            # On-chain data
            onchain = await self.onchain_collector.get_all_metrics(
                symbols=["BTC", "ETH", "SOL"]
            )

            for symbol, metrics in onchain.items():
                if metrics.fear_greed_value:
                    self.db.insert_onchain(
                        symbol, metrics.timestamp,
                        "fear_greed", metrics.fear_greed_value
                    )

            # Sentiment data
            sentiment = await self.sentiment_collector.get_aggregated_sentiment(
                symbols=["BTC", "ETH", "SOL"]
            )

            for symbol, data in sentiment.items():
                self.db.insert_sentiment(
                    "combined", data["timestamp"],
                    symbol, data["combined_score"],
                    data["reddit_volume"]
                )

        except Exception as e:
            self.log.error(f"Error collecting external data: {e}")

    async def start(self):
        """Start the bot."""
        self._running = True

        await self.initialize()

        if self.mode == "train":
            await self.run_training()
        elif self.mode == "backtest":
            await self.run_backtest()
        elif self.mode == "paper":
            await self.run_paper_trading()
        elif self.mode == "live":
            self.log.error("Live trading not yet implemented")
            # await self.run_live_trading()

    async def stop(self):
        """Stop the bot gracefully."""
        self.log.info("Stopping trading bot...")
        self._running = False

        # Cleanup
        if self.ws_client:
            await self.ws_client.disconnect()

        if self.onchain_collector:
            await self.onchain_collector.close()

        if self.sentiment_collector:
            await self.sentiment_collector.close()

        if self.order_manager:
            await self.order_manager.close()

        if self.db:
            self.db.close()

        self.log.info("Trading bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "mode",
        choices=["train", "backtest", "paper", "live"],
        help="Operating mode"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(
        log_dir=settings.logging.log_dir,
        log_level=args.log_level
    )

    logger.log_info(f"Starting trading bot in {args.mode} mode")
    logger.log_info(f"Device: {device_manager.device}")

    # Create bot
    bot = TradingBot(args.mode, logger)

    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        logger.log_info(f"Received signal {sig}, shutting down...")
        loop.create_task(bot.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.log_info("Interrupted by user")
    finally:
        loop.run_until_complete(bot.stop())
        loop.close()

    logger.log_info("Trading bot exited")


if __name__ == "__main__":
    main()
