"""
Performance metrics for trading evaluation.
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    num_trades: int
    avg_trade_duration: float


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate simple returns from price series."""
    return np.diff(prices) / prices[:-1]


def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate log returns from price series."""
    return np.diff(np.log(prices))


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 60  # Minutes per year
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0:
        return 0.0

    return mean_excess / std_excess * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 60
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation).

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year

    mean_excess = np.mean(excess_returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    return mean_excess / downside_std * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        equity_curve: Array of portfolio values over time

    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx, recovery_idx)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0, 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max

    max_dd_idx = np.argmin(drawdowns)
    max_dd = abs(drawdowns[max_dd_idx])

    # Find peak before max drawdown
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1])

    # Find recovery after max drawdown
    recovery_idx = max_dd_idx
    for i in range(max_dd_idx, len(equity_curve)):
        if equity_curve[i] >= equity_curve[peak_idx]:
            recovery_idx = i
            break

    duration = recovery_idx - peak_idx

    return max_dd, peak_idx, max_dd_idx, duration


def calculate_calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252 * 24 * 60
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Array of period returns
        equity_curve: Array of portfolio values
        periods_per_year: Number of periods in a year

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    annualized_return = np.mean(returns) * periods_per_year
    max_dd, _, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return float('inf') if annualized_return > 0 else 0.0

    return annualized_return / max_dd


def calculate_win_rate(pnl: np.ndarray) -> float:
    """Calculate win rate from P&L array."""
    if len(pnl) == 0:
        return 0.0

    wins = np.sum(pnl > 0)
    return wins / len(pnl)


def calculate_profit_factor(pnl: np.ndarray) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_average_win_loss(pnl: np.ndarray) -> Tuple[float, float]:
    """Calculate average winning and losing trade."""
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    return avg_win, avg_loss


def calculate_rolling_sharpe(
    returns: np.ndarray,
    window: int = 20,
    periods_per_year: int = 252 * 24 * 60
) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Array of period returns
        window: Rolling window size
        periods_per_year: Number of periods in a year

    Returns:
        Array of rolling Sharpe ratios
    """
    if len(returns) < window:
        return np.zeros(len(returns))

    rolling_sharpe = np.zeros(len(returns))

    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        rolling_sharpe[i] = calculate_sharpe_ratio(
            window_returns,
            periods_per_year=periods_per_year
        )

    return rolling_sharpe


def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252 * 24 * 60
) -> float:
    """Calculate Information Ratio against a benchmark."""
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0

    active_returns = returns - benchmark_returns

    mean_active = np.mean(active_returns)
    std_active = np.std(active_returns, ddof=1)

    if std_active == 0:
        return 0.0

    return mean_active / std_active * np.sqrt(periods_per_year)


def calculate_all_metrics(
    pnl: np.ndarray,
    equity_curve: np.ndarray,
    trade_durations: Optional[np.ndarray] = None,
    periods_per_year: int = 252 * 24 * 60
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.

    Args:
        pnl: Array of trade P&L values
        equity_curve: Array of portfolio values over time
        trade_durations: Optional array of trade durations
        periods_per_year: Number of periods in a year

    Returns:
        PerformanceMetrics dataclass with all metrics
    """
    returns = calculate_returns(equity_curve)

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if len(equity_curve) > 0 else 0.0
    annualized_return = np.mean(returns) * periods_per_year if len(returns) > 0 else 0.0

    sharpe = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(returns, periods_per_year=periods_per_year)
    max_dd, _, _, dd_duration = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(returns, equity_curve, periods_per_year=periods_per_year)

    win_rate = calculate_win_rate(pnl)
    profit_factor = calculate_profit_factor(pnl)
    avg_win, avg_loss = calculate_average_win_loss(pnl)

    avg_duration = np.mean(trade_durations) if trade_durations is not None and len(trade_durations) > 0 else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        num_trades=len(pnl),
        avg_trade_duration=avg_duration
    )


def format_metrics_report(metrics: PerformanceMetrics) -> str:
    """Format metrics as a human-readable report."""
    return f"""
Performance Report
==================
Total Return:        {metrics.total_return:.2%}
Annualized Return:   {metrics.annualized_return:.2%}
Sharpe Ratio:        {metrics.sharpe_ratio:.2f}
Sortino Ratio:       {metrics.sortino_ratio:.2f}
Calmar Ratio:        {metrics.calmar_ratio:.2f}
Max Drawdown:        {metrics.max_drawdown:.2%}
Drawdown Duration:   {metrics.max_drawdown_duration} periods

Trade Statistics
----------------
Number of Trades:    {metrics.num_trades}
Win Rate:            {metrics.win_rate:.2%}
Profit Factor:       {metrics.profit_factor:.2f}
Average Win:         {metrics.avg_win:.4f}
Average Loss:        {metrics.avg_loss:.4f}
Avg Trade Duration:  {metrics.avg_trade_duration:.1f} periods
"""
