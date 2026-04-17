"""
Backtesting engine for strategy evaluation.
Event-driven simulation with realistic market conditions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from utils.metrics import calculate_all_metrics, PerformanceMetrics, format_metrics_report

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Completed trade record."""
    id: int
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    fees: float
    slippage: float
    hold_duration: int  # In bars


@dataclass
class Position:
    """Open position."""
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Capital
    initial_capital: float = 10000.0

    # Costs
    maker_fee_bps: float = 1.0  # 0.01%
    taker_fee_bps: float = 6.0  # 0.06%
    slippage_bps: float = 5.0  # 0.05%

    # Execution
    use_limit_orders: bool = False
    fill_probability: float = 0.8  # For limit orders

    # Latency simulation
    latency_bars: int = 0  # Execution delay in bars

    # Position limits
    max_position_pct: float = 0.1
    max_positions: int = 3

    # Risk
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2%
    use_take_profit: bool = False
    take_profit_pct: float = 0.04  # 4%


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: np.ndarray
    returns: np.ndarray
    drawdown_curve: np.ndarray
    positions_over_time: np.ndarray
    start_time: datetime
    end_time: datetime
    n_bars: int


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates trading with realistic:
    - Transaction costs
    - Slippage
    - Execution latency
    - Position management
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Dict] = []
        self.trades: List[Trade] = []
        self.trade_counter = 0

        # Time series tracking
        self.equity_curve: List[float] = []
        self.position_sizes: List[float] = []

        # Current bar
        self.current_bar = 0
        self.current_time: Optional[datetime] = None

    def reset(self):
        """Reset engine state."""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.pending_orders = []
        self.trades = []
        self.trade_counter = 0
        self.equity_curve = [self.config.initial_capital]
        self.position_sizes = [0.0]
        self.current_bar = 0
        self.current_time = None

    def run(
        self,
        data: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame, int], Dict[str, int]],
        price_col: str = 'close',
        time_col: str = 'timestamp'
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            data: OHLCV DataFrame
            signal_func: Function that returns signals {symbol: direction}
            price_col: Column name for execution price
            time_col: Column name for timestamps

        Returns:
            BacktestResult with all metrics and trades
        """
        self.reset()

        start_time = None
        end_time = None

        for bar in range(len(data)):
            self.current_bar = bar
            row = data.iloc[bar]

            if time_col in data.columns:
                self.current_time = pd.to_datetime(row[time_col])
                if start_time is None:
                    start_time = self.current_time
                end_time = self.current_time
            else:
                self.current_time = datetime.now()

            price = row[price_col]

            # Process pending orders
            self._process_pending_orders(price)

            # Check stop-loss and take-profit
            self._check_exits(price)

            # Get signals
            signals = signal_func(data, bar)

            # Generate orders from signals
            for symbol, direction in signals.items():
                if direction != 0:
                    self._generate_order(symbol, direction, price)
                elif symbol in self.positions:
                    # Close position on neutral signal
                    self._close_position(symbol, price)

            # Update equity
            equity = self._calculate_equity(price)
            self.equity_curve.append(equity)

            # Track position size
            total_exposure = sum(
                abs(pos.size * price) for pos in self.positions.values()
            )
            self.position_sizes.append(total_exposure / equity if equity > 0 else 0)

        # Close remaining positions
        final_price = data.iloc[-1][price_col]
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, final_price)

        # Calculate results
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak

        pnl = np.array([t.pnl for t in self.trades]) if self.trades else np.array([0.0])

        metrics = calculate_all_metrics(
            pnl,
            equity_curve,
            np.array([t.hold_duration for t in self.trades]) if self.trades else None
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            equity_curve=equity_curve,
            returns=returns,
            drawdown_curve=drawdown,
            positions_over_time=np.array(self.position_sizes),
            start_time=start_time or datetime.now(),
            end_time=end_time or datetime.now(),
            n_bars=len(data)
        )

    def _generate_order(self, symbol: str, direction: int, price: float):
        """Generate order from signal."""
        # Check if already positioned
        if symbol in self.positions:
            current_pos = self.positions[symbol]
            if current_pos.direction.value == direction:
                return  # Already in same direction

            # Close existing position first
            self._close_position(symbol, price)

        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            return

        # Calculate position size
        equity = self._calculate_equity(price)
        position_value = equity * self.config.max_position_pct
        size = position_value / price

        order = {
            'symbol': symbol,
            'direction': TradeDirection.LONG if direction > 0 else TradeDirection.SHORT,
            'size': size,
            'price': price,
            'bar': self.current_bar
        }

        if self.config.latency_bars > 0:
            self.pending_orders.append(order)
        else:
            self._execute_order(order)

    def _process_pending_orders(self, current_price: float):
        """Process delayed orders."""
        executed = []

        for i, order in enumerate(self.pending_orders):
            if self.current_bar >= order['bar'] + self.config.latency_bars:
                order['price'] = current_price  # Execute at current price
                self._execute_order(order)
                executed.append(i)

        # Remove executed orders
        for i in reversed(executed):
            self.pending_orders.pop(i)

    def _execute_order(self, order: Dict):
        """Execute an order."""
        symbol = order['symbol']
        direction = order['direction']
        size = order['size']
        price = order['price']

        # Apply slippage
        slippage = price * self.config.slippage_bps / 10000
        if direction == TradeDirection.LONG:
            entry_price = price + slippage
        else:
            entry_price = price - slippage

        # Calculate fees
        fee = abs(size * entry_price) * self.config.taker_fee_bps / 10000

        # Deduct from capital
        self.capital -= fee

        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_time=self.current_time,
            entry_price=entry_price,
            size=size
        )

        # Set stop-loss
        if self.config.use_stop_loss:
            if direction == TradeDirection.LONG:
                position.stop_loss = entry_price * (1 - self.config.stop_loss_pct)
            else:
                position.stop_loss = entry_price * (1 + self.config.stop_loss_pct)

        # Set take-profit
        if self.config.use_take_profit:
            if direction == TradeDirection.LONG:
                position.take_profit = entry_price * (1 + self.config.take_profit_pct)
            else:
                position.take_profit = entry_price * (1 - self.config.take_profit_pct)

        self.positions[symbol] = position

    def _check_exits(self, price: float):
        """Check stop-loss and take-profit conditions."""
        to_close = []

        for symbol, pos in self.positions.items():
            should_close = False

            if pos.direction == TradeDirection.LONG:
                if pos.stop_loss and price <= pos.stop_loss:
                    should_close = True
                if pos.take_profit and price >= pos.take_profit:
                    should_close = True
            else:
                if pos.stop_loss and price >= pos.stop_loss:
                    should_close = True
                if pos.take_profit and price <= pos.take_profit:
                    should_close = True

            if should_close:
                to_close.append(symbol)

        for symbol in to_close:
            self._close_position(symbol, price)

    def _close_position(self, symbol: str, price: float):
        """Close a position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Apply slippage
        slippage = price * self.config.slippage_bps / 10000
        if pos.direction == TradeDirection.LONG:
            exit_price = price - slippage
        else:
            exit_price = price + slippage

        # Calculate P&L
        if pos.direction == TradeDirection.LONG:
            pnl = pos.size * (exit_price - pos.entry_price)
        else:
            pnl = pos.size * (pos.entry_price - exit_price)

        # Fees
        fee = abs(pos.size * exit_price) * self.config.taker_fee_bps / 10000
        pnl -= fee

        # Update capital
        self.capital += pnl

        # Record trade
        self.trade_counter += 1
        trade = Trade(
            id=self.trade_counter,
            symbol=symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=self.current_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            return_pct=pnl / (pos.size * pos.entry_price),
            fees=fee * 2,  # Entry + exit
            slippage=slippage * 2,
            hold_duration=self.current_bar  # Approximate
        )
        self.trades.append(trade)

        # Remove position
        del self.positions[symbol]

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized P&L."""
        equity = self.capital

        for pos in self.positions.values():
            if pos.direction == TradeDirection.LONG:
                unrealized = pos.size * (current_price - pos.entry_price)
            else:
                unrealized = pos.size * (pos.entry_price - current_price)

            equity += unrealized

        return equity


def run_backtest(
    data: pd.DataFrame,
    signal_func: Callable,
    config: BacktestConfig = None
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        data: OHLCV DataFrame
        signal_func: Signal generation function
        config: Backtest configuration

    Returns:
        BacktestResult
    """
    engine = BacktestEngine(config)
    return engine.run(data, signal_func)


def compare_strategies(
    data: pd.DataFrame,
    strategies: Dict[str, Callable],
    config: BacktestConfig = None
) -> Dict[str, BacktestResult]:
    """
    Compare multiple strategies on the same data.

    Args:
        data: OHLCV DataFrame
        strategies: Dict mapping strategy name to signal function
        config: Backtest configuration

    Returns:
        Dict mapping strategy name to BacktestResult
    """
    results = {}

    for name, signal_func in strategies.items():
        logger.info(f"Backtesting strategy: {name}")
        engine = BacktestEngine(config)
        results[name] = engine.run(data, signal_func)

    return results


def generate_comparison_report(results: Dict[str, BacktestResult]) -> str:
    """Generate comparison report for multiple strategies."""
    lines = []
    lines.append("=" * 80)
    lines.append("STRATEGY COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Header
    header = f"{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}"
    lines.append(header)
    lines.append("-" * 80)

    # Rows
    for name, result in results.items():
        m = result.metrics
        row = (
            f"{name:<20} "
            f"{m.total_return:>9.1%} "
            f"{m.sharpe_ratio:>8.2f} "
            f"{m.sortino_ratio:>8.2f} "
            f"{m.max_drawdown:>7.1%} "
            f"{m.win_rate:>7.1%} "
            f"{m.num_trades:>8}"
        )
        lines.append(row)

    lines.append("-" * 80)

    # Best performers
    lines.append("")
    lines.append("RANKINGS")
    lines.append("-" * 40)

    # Best Sharpe
    best_sharpe = max(results.items(), key=lambda x: x[1].metrics.sharpe_ratio)
    lines.append(f"Best Sharpe:   {best_sharpe[0]} ({best_sharpe[1].metrics.sharpe_ratio:.2f})")

    # Best Return
    best_return = max(results.items(), key=lambda x: x[1].metrics.total_return)
    lines.append(f"Best Return:   {best_return[0]} ({best_return[1].metrics.total_return:.1%})")

    # Lowest Drawdown
    best_dd = min(results.items(), key=lambda x: x[1].metrics.max_drawdown)
    lines.append(f"Lowest DD:     {best_dd[0]} ({best_dd[1].metrics.max_drawdown:.1%})")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
