"""
Global configuration settings for the trading bot.
"""
import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class BybitConfig:
    """Bybit API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BYBIT_API_SECRET", ""))

    # Mode: "demo", "testnet", or "live"
    # - demo: Uses production URLs with demo account credentials (Bybit demo trading)
    # - testnet: Uses testnet URLs (separate testnet environment)
    # - live: Uses production URLs with real account
    mode: str = "demo"

    # URLs are set in __post_init__ based on mode
    ws_url: str = ""
    rest_url: str = ""

    def __post_init__(self):
        if self.mode == "testnet":
            # Testnet has separate URLs
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
            self.rest_url = "https://api-testnet.bybit.com"
        else:
            # Demo and Live both use production URLs
            # Demo trading is distinguished by the account/credentials, not the URL
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"
            self.rest_url = "https://api.bybit.com"


@dataclass
class TradingConfig:
    """Trading parameters."""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    base_currency: str = "USDT"

    # Timeframe: 5m-15m scalping
    candle_interval: str = "5"  # 5-minute candles
    hold_time_candles: int = 3  # Target hold: 3 candles (15 min)

    # Position limits
    max_position_pct: float = 0.10  # 10% of capital per asset
    max_total_exposure_pct: float = 0.25  # 25% total exposure
    max_risk_per_trade_pct: float = 0.01  # 1% risk per trade

    # Trade frequency (adjusted for 5m timeframe)
    min_trade_interval_sec: float = 60.0  # Minimum 1 minute between trades
    max_trades_per_hour: int = 12  # ~1 trade per 5 min max

    # Order book depth
    orderbook_depth: int = 25  # Levels to capture
    orderbook_snapshot_interval_ms: int = 100  # 100ms snapshots


@dataclass
class RiskConfig:
    """Risk management parameters."""
    # Stop-loss
    stop_loss_atr_multiplier: float = 2.0  # 2x ATR for stops

    # Drawdown limits
    max_daily_drawdown_pct: float = 0.05  # 5% daily max loss
    max_total_drawdown_pct: float = 0.15  # 15% total max loss

    # Position sizing
    use_kelly: bool = True
    kelly_fraction: float = 0.5  # Half-Kelly

    # Performance targets
    min_sharpe_ratio: float = 2.0
    min_sortino_ratio: float = 2.0


@dataclass
class DataConfig:
    """Data collection and storage settings."""
    # Storage
    data_dir: Path = field(default_factory=lambda: Path.home() / "trading_bot_data")
    db_path: Path = field(default_factory=lambda: Path.home() / "trading_bot_data" / "timeseries.duckdb")

    # Historical data
    lookback_days: int = 180  # 6 months for training

    # Feature windows
    feature_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])

    # On-chain APIs (free tiers)
    etherscan_api_key: str = field(default_factory=lambda: os.getenv("ETHERSCAN_API_KEY", ""))

    # Rate limits (requests per second)
    etherscan_rate_limit: float = 5.0
    solscan_rate_limit: float = 10.0
    coingecko_rate_limit: float = 0.5  # 30 per minute

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture settings."""
    # Sequence length for time series
    sequence_length: int = 100

    # Transformer settings
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    dropout: float = 0.1

    # GRU settings
    gru_hidden_size: int = 128
    gru_num_layers: int = 2

    # GNN settings
    gnn_hidden_channels: int = 64
    gnn_num_layers: int = 3

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: Path = field(default_factory=lambda: Path.home() / "trading_bot_data" / "logs")
    log_level: str = "INFO"
    log_to_file: bool = True
    log_trades: bool = True
    tensorboard_dir: Path = field(default_factory=lambda: Path.home() / "trading_bot_data" / "tensorboard")

    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Main settings container."""
    bybit: BybitConfig = field(default_factory=BybitConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Global settings instance
settings = Settings()
