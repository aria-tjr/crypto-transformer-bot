"""
Risk management with stop-losses, drawdown limits, and circuit breakers.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskStatus(Enum):
    """Risk status levels."""
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    HALTED = 4


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Stop-loss settings
    stop_loss_atr_multiplier: float = 2.0
    trailing_stop_atr_multiplier: float = 1.5
    use_trailing_stop: bool = True

    # Drawdown limits
    max_daily_drawdown_pct: float = 0.05  # 5%
    max_total_drawdown_pct: float = 0.15  # 15%
    drawdown_warning_pct: float = 0.03  # 3% triggers warning

    # Circuit breakers
    max_consecutive_losses: int = 5
    cooldown_minutes: int = 30
    max_trades_per_hour: int = 20

    # Position limits
    max_position_pct: float = 0.10
    max_total_exposure_pct: float = 0.25

    # Volatility-based adjustments
    volatility_scale_factor: float = 1.0
    high_volatility_threshold: float = 0.03  # 3% daily vol

    # Recovery settings
    recovery_reduction_pct: float = 0.5  # Reduce size by 50% after drawdown


@dataclass
class StopLoss:
    """Stop-loss order."""
    symbol: str
    entry_price: float
    stop_price: float
    is_trailing: bool = False
    highest_price: float = 0.0  # For trailing stops
    created_at: datetime = field(default_factory=datetime.now)

    def update_trailing(self, current_price: float, atr: float, multiplier: float) -> bool:
        """
        Update trailing stop.

        Returns True if stop was updated.
        """
        if not self.is_trailing:
            return False

        if current_price > self.highest_price:
            self.highest_price = current_price
            new_stop = current_price - atr * multiplier
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                return True

        return False

    def is_triggered(self, current_price: float) -> bool:
        """Check if stop-loss is triggered."""
        return current_price <= self.stop_price


@dataclass
class RiskState:
    """Current risk state."""
    status: RiskStatus = RiskStatus.NORMAL
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    consecutive_losses: int = 0
    trades_this_hour: int = 0
    last_trade_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    position_size_multiplier: float = 1.0


class RiskManager:
    """
    Manages trading risk with multiple safety mechanisms.
    """

    def __init__(
        self,
        initial_capital: float,
        config: RiskConfig = None
    ):
        self.initial_capital = initial_capital
        self.config = config or RiskConfig()

        # State
        self.state = RiskState(peak_equity=initial_capital)
        self.current_equity = initial_capital

        # Stop-losses
        self.stop_losses: Dict[str, StopLoss] = {}

        # History
        self.equity_history: deque = deque(maxlen=1000)
        self.trade_history: List[Dict] = []
        self.daily_start_equity = initial_capital
        self.last_daily_reset = datetime.now().date()

        # ATR cache
        self.atr_cache: Dict[str, float] = {}

    def check_can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if halted
        if self.state.status == RiskStatus.HALTED:
            return False, "Trading halted due to risk limits"

        # Check cooldown
        if self.state.cooldown_until:
            if datetime.now() < self.state.cooldown_until:
                remaining = (self.state.cooldown_until - datetime.now()).seconds // 60
                return False, f"In cooldown period ({remaining} minutes remaining)"
            else:
                self.state.cooldown_until = None

        # Check hourly trade limit
        if self.state.trades_this_hour >= self.config.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({self.config.max_trades_per_hour})"

        # Check drawdown
        if self.state.current_drawdown >= self.config.max_total_drawdown_pct:
            self._trigger_halt("Maximum drawdown exceeded")
            return False, "Maximum drawdown exceeded"

        if self.state.daily_pnl <= -self.initial_capital * self.config.max_daily_drawdown_pct:
            self._trigger_halt("Daily loss limit exceeded")
            return False, "Daily loss limit exceeded"

        # Check consecutive losses
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger_cooldown()
            return False, f"Consecutive loss limit reached ({self.config.max_consecutive_losses})"

        return True, "OK"

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on risk state.

        Returns value between 0 and 1.
        """
        multiplier = 1.0

        # Reduce after drawdown
        if self.state.current_drawdown > self.config.drawdown_warning_pct:
            drawdown_factor = self.state.current_drawdown / self.config.max_total_drawdown_pct
            multiplier *= (1 - drawdown_factor * self.config.recovery_reduction_pct)

        # Reduce after consecutive losses
        if self.state.consecutive_losses > 2:
            loss_factor = min(1.0, self.state.consecutive_losses / self.config.max_consecutive_losses)
            multiplier *= (1 - loss_factor * 0.5)

        # Status-based reduction
        if self.state.status == RiskStatus.WARNING:
            multiplier *= 0.5
        elif self.state.status == RiskStatus.CRITICAL:
            multiplier *= 0.25

        return max(0.1, multiplier)

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        is_long: bool,
        atr: Optional[float] = None
    ) -> float:
        """
        Calculate stop-loss price.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            is_long: True for long position
            atr: Average True Range (optional)

        Returns:
            Stop-loss price
        """
        if atr is None:
            atr = self.atr_cache.get(symbol, entry_price * 0.02)

        stop_distance = atr * self.config.stop_loss_atr_multiplier

        if is_long:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        return stop_price

    def set_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        is_trailing: bool = None
    ):
        """Set stop-loss for a position."""
        if is_trailing is None:
            is_trailing = self.config.use_trailing_stop

        self.stop_losses[symbol] = StopLoss(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            is_trailing=is_trailing,
            highest_price=entry_price
        )

    def check_stop_losses(
        self,
        prices: Dict[str, float]
    ) -> List[str]:
        """
        Check all stop-losses against current prices.

        Args:
            prices: Dict mapping symbol to current price

        Returns:
            List of symbols where stop was triggered
        """
        triggered = []

        for symbol, stop in list(self.stop_losses.items()):
            if symbol in prices:
                current_price = prices[symbol]

                # Update trailing stop
                if stop.is_trailing:
                    atr = self.atr_cache.get(symbol, stop.entry_price * 0.02)
                    stop.update_trailing(
                        current_price,
                        atr,
                        self.config.trailing_stop_atr_multiplier
                    )

                # Check if triggered
                if stop.is_triggered(current_price):
                    triggered.append(symbol)
                    logger.warning(f"Stop-loss triggered for {symbol} at {current_price}")

        return triggered

    def remove_stop_loss(self, symbol: str):
        """Remove stop-loss for a symbol."""
        if symbol in self.stop_losses:
            del self.stop_losses[symbol]

    def update_atr(self, symbol: str, atr: float):
        """Update ATR cache for symbol."""
        self.atr_cache[symbol] = atr

    def update_equity(self, equity: float):
        """Update current equity and risk metrics."""
        self.current_equity = equity
        self.equity_history.append(equity)

        # Update peak and drawdown
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        self.state.current_drawdown = (self.state.peak_equity - equity) / self.state.peak_equity

        # Reset daily tracking if needed
        today = datetime.now().date()
        if today > self.last_daily_reset:
            self.daily_start_equity = equity
            self.last_daily_reset = today
            self.state.daily_pnl = 0
            self.state.trades_this_hour = 0

        # Update daily P&L
        self.state.daily_pnl = equity - self.daily_start_equity
        self.state.total_pnl = equity - self.initial_capital

        # Update risk status
        self._update_risk_status()

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        return_pct: float
    ):
        """Record a completed trade."""
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'return_pct': return_pct,
            'timestamp': datetime.now()
        })

        # Update consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Update hourly trade count
        self.state.trades_this_hour += 1
        self.state.last_trade_time = datetime.now()

        # Check if cooldown needed
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger_cooldown()

    def _update_risk_status(self):
        """Update risk status based on current metrics."""
        if self.state.current_drawdown >= self.config.max_total_drawdown_pct:
            self.state.status = RiskStatus.HALTED
        elif self.state.current_drawdown >= self.config.max_total_drawdown_pct * 0.8:
            self.state.status = RiskStatus.CRITICAL
        elif self.state.current_drawdown >= self.config.drawdown_warning_pct:
            self.state.status = RiskStatus.WARNING
        elif self.state.consecutive_losses >= 3:
            self.state.status = RiskStatus.CAUTION
        else:
            self.state.status = RiskStatus.NORMAL

    def _trigger_cooldown(self):
        """Trigger cooldown period."""
        self.state.cooldown_until = datetime.now() + timedelta(
            minutes=self.config.cooldown_minutes
        )
        logger.warning(f"Cooldown triggered until {self.state.cooldown_until}")

    def _trigger_halt(self, reason: str):
        """Halt all trading."""
        self.state.status = RiskStatus.HALTED
        logger.error(f"Trading HALTED: {reason}")

    def reset_halt(self):
        """Reset halt status (requires manual intervention)."""
        self.state.status = RiskStatus.CAUTION
        self.state.consecutive_losses = 0
        logger.info("Trading halt reset - status set to CAUTION")

    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report."""
        return {
            'status': self.state.status.name,
            'current_equity': self.current_equity,
            'initial_capital': self.initial_capital,
            'total_pnl': self.state.total_pnl,
            'total_return_pct': self.state.total_pnl / self.initial_capital * 100,
            'daily_pnl': self.state.daily_pnl,
            'daily_return_pct': self.state.daily_pnl / self.daily_start_equity * 100,
            'peak_equity': self.state.peak_equity,
            'current_drawdown_pct': self.state.current_drawdown * 100,
            'max_allowed_drawdown_pct': self.config.max_total_drawdown_pct * 100,
            'consecutive_losses': self.state.consecutive_losses,
            'trades_this_hour': self.state.trades_this_hour,
            'active_stop_losses': len(self.stop_losses),
            'position_size_multiplier': self.get_position_size_multiplier(),
            'in_cooldown': self.state.cooldown_until is not None
        }


class PortfolioRiskManager(RiskManager):
    """
    Extended risk manager for multi-asset portfolios.
    """

    def __init__(
        self,
        initial_capital: float,
        symbols: List[str],
        config: RiskConfig = None
    ):
        super().__init__(initial_capital, config)
        self.symbols = symbols

        # Per-asset tracking
        self.asset_pnl: Dict[str, float] = {s: 0 for s in symbols}
        self.asset_exposure: Dict[str, float] = {s: 0 for s in symbols}

    def update_asset_exposure(self, symbol: str, exposure: float):
        """Update exposure for an asset."""
        self.asset_exposure[symbol] = exposure

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure."""
        return sum(abs(e) for e in self.asset_exposure.values())

    def can_increase_exposure(self, symbol: str, additional: float) -> Tuple[bool, str]:
        """Check if additional exposure is allowed."""
        current_total = self.get_total_exposure()
        current_asset = abs(self.asset_exposure.get(symbol, 0))

        # Check asset limit
        if current_asset + additional > self.current_equity * self.config.max_position_pct:
            return False, f"Asset exposure limit exceeded for {symbol}"

        # Check total limit
        if current_total + additional > self.current_equity * self.config.max_total_exposure_pct:
            return False, "Total portfolio exposure limit exceeded"

        return True, "OK"

    def get_diversification_score(self) -> float:
        """
        Calculate portfolio diversification score.

        Returns value from 0 (concentrated) to 1 (diversified).
        """
        exposures = [abs(e) for e in self.asset_exposure.values() if e != 0]

        if not exposures:
            return 1.0

        total = sum(exposures)
        if total == 0:
            return 1.0

        # Herfindahl-Hirschman Index
        weights = [e / total for e in exposures]
        hhi = sum(w ** 2 for w in weights)

        # Normalize: 1 asset = HHI 1.0, N assets evenly = HHI 1/N
        # Score = 1 - HHI (higher is more diversified)
        return 1 - hhi
