"""
Structured logging and trade journal.
"""
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from logging.handlers import RotatingFileHandler
import threading


@dataclass
class TradeLogEntry:
    """Trade journal entry."""
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, CLOSE
    direction: str  # LONG, SHORT
    price: float
    size: float
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    reason: Optional[str] = None
    model_confidence: Optional[float] = None
    regime: Optional[str] = None


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Truncate long messages
        message = record.getMessage()
        if len(message) > 200:
            message = message[:197] + "..."

        return f"{color}[{timestamp}] {record.levelname:<8}{self.RESET} {message}"


class TradeJournal:
    """
    Trade journal for logging and analyzing trades.

    Provides:
    - Persistent trade logging
    - Trade analysis
    - Performance tracking
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.entries: List[TradeLogEntry] = []
        self._lock = threading.Lock()

        # Load existing entries
        self._load()

    def _load(self):
        """Load existing journal entries."""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    data = json.load(f)
                    self.entries = [TradeLogEntry(**e) for e in data]
            except (json.JSONDecodeError, KeyError):
                self.entries = []

    def _save(self):
        """Save journal to disk."""
        with open(self.log_path, 'w') as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def log_trade(
        self,
        symbol: str,
        action: str,
        direction: str,
        price: float,
        size: float,
        pnl: float = None,
        return_pct: float = None,
        reason: str = None,
        confidence: float = None,
        regime: str = None
    ):
        """Log a trade entry."""
        entry = TradeLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            action=action,
            direction=direction,
            price=price,
            size=size,
            pnl=pnl,
            return_pct=return_pct,
            reason=reason,
            model_confidence=confidence,
            regime=regime
        )

        with self._lock:
            self.entries.append(entry)
            self._save()

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[TradeLogEntry]:
        """Get filtered trade entries."""
        result = self.entries

        if symbol:
            result = [e for e in result if e.symbol == symbol]

        if start_date:
            result = [e for e in result if e.timestamp >= start_date]

        if end_date:
            result = [e for e in result if e.timestamp <= end_date]

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get trade summary statistics."""
        if not self.entries:
            return {}

        closed_trades = [e for e in self.entries if e.pnl is not None]

        if not closed_trades:
            return {'total_trades': len(self.entries), 'closed_trades': 0}

        pnls = [e.pnl for e in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            'total_trades': len(self.entries),
            'closed_trades': len(closed_trades),
            'total_pnl': sum(pnls),
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': len(wins) / len(closed_trades) if closed_trades else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'largest_win': max(wins) if wins else 0,
            'largest_loss': min(losses) if losses else 0
        }


class TradingLogger:
    """
    Main logging facade for the trading bot.

    Provides:
    - Structured JSON file logging
    - Console logging
    - Trade journal
    - Performance metrics
    """

    def __init__(
        self,
        log_dir: Path,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_level = getattr(logging, log_level.upper())

        # Root logger
        self.root_logger = logging.getLogger("trading_bot")
        self.root_logger.setLevel(self.log_level)
        self.root_logger.handlers = []  # Clear existing handlers

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            console_handler.setLevel(self.log_level)
            self.root_logger.addHandler(console_handler)

        # File handler (JSON)
        if log_to_file:
            log_file = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            file_handler.setFormatter(JsonFormatter())
            file_handler.setLevel(logging.DEBUG)
            self.root_logger.addHandler(file_handler)

        # Trade journal
        self.journal = TradeJournal(self.log_dir / "trade_journal.json")

        # Performance logger
        self._setup_performance_logger()

    def _setup_performance_logger(self):
        """Setup separate performance metrics logger."""
        perf_logger = logging.getLogger("trading_bot.performance")
        perf_file = self.log_dir / "performance.log"

        handler = RotatingFileHandler(
            perf_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        handler.setFormatter(JsonFormatter())
        perf_logger.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a named logger."""
        return logging.getLogger(f"trading_bot.{name}")

    def log_trade(self, **kwargs):
        """Log a trade to the journal."""
        self.journal.log_trade(**kwargs)

        # Also log to main logger
        self.root_logger.info(
            f"Trade: {kwargs.get('action')} {kwargs.get('symbol')} "
            f"@ {kwargs.get('price')}"
        )

    def log_performance(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        perf_logger = logging.getLogger("trading_bot.performance")

        record = logging.LogRecord(
            name="trading_bot.performance",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Performance metrics",
            args=(),
            exc_info=None
        )
        record.extra_data = {"metrics": metrics}
        perf_logger.handle(record)

    def log_error(self, message: str, exc_info: bool = False):
        """Log an error."""
        self.root_logger.error(message, exc_info=exc_info)

    def log_warning(self, message: str):
        """Log a warning."""
        self.root_logger.warning(message)

    def log_info(self, message: str):
        """Log info."""
        self.root_logger.info(message)

    def log_debug(self, message: str):
        """Log debug."""
        self.root_logger.debug(message)


def setup_logging(
    log_dir: Path = None,
    log_level: str = "INFO"
) -> TradingLogger:
    """
    Setup logging for the trading bot.

    Args:
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        Configured TradingLogger
    """
    if log_dir is None:
        log_dir = Path.home() / "trading_bot_data" / "logs"

    return TradingLogger(log_dir, log_level)
