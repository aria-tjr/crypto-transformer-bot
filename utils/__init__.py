"""Utilities module."""
from .device import device_manager, get_device, get_dtype, to_device
from .metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_all_metrics,
    PerformanceMetrics
)

__all__ = [
    "device_manager",
    "get_device",
    "get_dtype",
    "to_device",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "calculate_all_metrics",
    "PerformanceMetrics"
]
