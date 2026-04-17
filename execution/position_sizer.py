"""
Position sizing using Kelly Criterion and volatility adjustments.
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    size: float  # Position size in base currency
    size_pct: float  # Position size as % of capital
    kelly_fraction: float  # Raw Kelly fraction
    adjusted_fraction: float  # Adjusted fraction used
    volatility_adjustment: float  # Volatility scaling factor
    confidence_adjustment: float  # Confidence scaling factor
    max_size: float  # Maximum allowed size
    reasoning: str  # Explanation of sizing decision


@dataclass
class PositionSizerConfig:
    """Configuration for position sizing."""
    # Kelly settings
    use_kelly: bool = True
    kelly_fraction: float = 0.5  # Half-Kelly
    min_kelly_fraction: float = 0.1
    max_kelly_fraction: float = 0.25

    # Position limits
    max_position_pct: float = 0.10  # 10% max per position
    max_total_exposure_pct: float = 0.25  # 25% total exposure
    min_position_size: float = 10.0  # Minimum $10 position

    # Volatility adjustment
    target_volatility: float = 0.02  # 2% daily volatility target
    volatility_lookback: int = 20  # Days for volatility calculation
    volatility_floor: float = 0.5  # Min volatility multiplier
    volatility_ceiling: float = 2.0  # Max volatility multiplier

    # Risk per trade
    max_risk_per_trade_pct: float = 0.01  # 1% risk per trade


class PositionSizer:
    """
    Calculates optimal position sizes using Kelly Criterion
    with volatility and confidence adjustments.
    """

    def __init__(self, config: PositionSizerConfig = None):
        self.config = config or PositionSizerConfig()

        # Performance tracking for Kelly calculation
        self.trade_returns: deque = deque(maxlen=100)
        self.win_count = 0
        self.loss_count = 0

        # Volatility tracking per symbol
        self.price_history: Dict[str, deque] = {}
        self.volatility_cache: Dict[str, float] = {}

    def calculate_size(
        self,
        capital: float,
        price: float,
        symbol: str,
        confidence: float = 1.0,
        current_exposure: float = 0.0,
        stop_loss_pct: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.

        Args:
            capital: Available capital
            price: Current asset price
            symbol: Trading symbol
            confidence: Model confidence (0-1)
            current_exposure: Current total exposure as fraction
            stop_loss_pct: Optional stop-loss percentage for risk-based sizing

        Returns:
            PositionSizeResult with sizing details
        """
        # Start with max allowed by config
        max_size_by_position = capital * self.config.max_position_pct
        max_size_by_exposure = capital * (self.config.max_total_exposure_pct - current_exposure)
        max_size = min(max_size_by_position, max_size_by_exposure)

        if max_size < self.config.min_position_size:
            return PositionSizeResult(
                size=0,
                size_pct=0,
                kelly_fraction=0,
                adjusted_fraction=0,
                volatility_adjustment=1.0,
                confidence_adjustment=confidence,
                max_size=max_size,
                reasoning="Insufficient capital for minimum position size"
            )

        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction()

        # Volatility adjustment
        volatility = self._get_volatility(symbol)
        volatility_adjustment = self._calculate_volatility_adjustment(volatility)

        # Confidence adjustment (scale down for low confidence)
        confidence_adjustment = self._calculate_confidence_adjustment(confidence)

        # Combined adjustment
        adjusted_fraction = kelly_fraction * volatility_adjustment * confidence_adjustment

        # Clamp to config limits
        adjusted_fraction = np.clip(
            adjusted_fraction,
            self.config.min_kelly_fraction,
            self.config.max_kelly_fraction
        )

        # Calculate size
        if stop_loss_pct and stop_loss_pct > 0:
            # Risk-based sizing: size such that loss at stop = max_risk_per_trade
            max_loss = capital * self.config.max_risk_per_trade_pct
            size_by_risk = max_loss / stop_loss_pct
            size = min(capital * adjusted_fraction, size_by_risk, max_size)
            reasoning = f"Risk-based sizing with {stop_loss_pct:.1%} stop"
        else:
            size = min(capital * adjusted_fraction, max_size)
            reasoning = f"Kelly-based sizing with {adjusted_fraction:.1%} fraction"

        # Ensure minimum size
        if size < self.config.min_position_size:
            size = 0
            reasoning = "Position size below minimum threshold"

        size_pct = size / capital if capital > 0 else 0

        return PositionSizeResult(
            size=size,
            size_pct=size_pct,
            kelly_fraction=kelly_fraction,
            adjusted_fraction=adjusted_fraction,
            volatility_adjustment=volatility_adjustment,
            confidence_adjustment=confidence_adjustment,
            max_size=max_size,
            reasoning=reasoning
        )

    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly fraction from trade history.

        Kelly = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        """
        if not self.config.use_kelly:
            return self.config.max_position_pct

        total_trades = self.win_count + self.loss_count

        if total_trades < 20:
            # Not enough data, use conservative estimate
            return self.config.kelly_fraction * 0.5

        win_rate = self.win_count / total_trades

        # Calculate average win/loss ratio
        if self.trade_returns:
            returns = list(self.trade_returns)
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]

            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 1

            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        else:
            win_loss_ratio = 1

        # Kelly formula
        if win_loss_ratio == 0:
            kelly = 0
        else:
            kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Apply fractional Kelly
        kelly = kelly * self.config.kelly_fraction

        # Clamp to reasonable range
        kelly = max(0, min(kelly, self.config.max_kelly_fraction))

        return kelly

    def _get_volatility(self, symbol: str) -> float:
        """Get volatility for symbol."""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]

        if symbol not in self.price_history:
            return self.config.target_volatility

        prices = list(self.price_history[symbol])
        if len(prices) < 5:
            return self.config.target_volatility

        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]

        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        self.volatility_cache[symbol] = volatility
        return volatility

    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """
        Calculate volatility-based position size adjustment.

        Higher volatility = smaller position (inverse volatility sizing)
        """
        if volatility <= 0:
            return 1.0

        # Target volatility / actual volatility
        adjustment = self.config.target_volatility / volatility

        # Clamp to floor/ceiling
        adjustment = np.clip(
            adjustment,
            self.config.volatility_floor,
            self.config.volatility_ceiling
        )

        return adjustment

    def _calculate_confidence_adjustment(self, confidence: float) -> float:
        """
        Adjust position size based on prediction confidence.

        Scale position linearly with confidence.
        """
        # Minimum confidence threshold
        if confidence < 0.5:
            return 0.5

        # Linear scaling from 0.5 to 1.0
        return 0.5 + 0.5 * confidence

    def update_price(self, symbol: str, price: float):
        """Update price history for volatility calculation."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.config.volatility_lookback)

        self.price_history[symbol].append(price)

        # Invalidate cache
        if symbol in self.volatility_cache:
            del self.volatility_cache[symbol]

    def record_trade(self, pnl: float, return_pct: float):
        """Record trade result for Kelly calculation."""
        self.trade_returns.append(return_pct)

        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

    def get_stats(self) -> Dict:
        """Get position sizing statistics."""
        total_trades = self.win_count + self.loss_count

        if total_trades == 0:
            return {
                'win_rate': 0,
                'kelly_fraction': self.config.kelly_fraction * 0.5,
                'avg_return': 0,
                'total_trades': 0
            }

        returns = list(self.trade_returns) if self.trade_returns else [0]

        return {
            'win_rate': self.win_count / total_trades,
            'kelly_fraction': self._calculate_kelly_fraction(),
            'avg_return': np.mean(returns),
            'total_trades': total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count
        }


class CorrelationAdjustedSizer(PositionSizer):
    """
    Position sizer that accounts for correlations between assets.

    Reduces position sizes when assets are highly correlated
    to avoid concentrated risk.
    """

    def __init__(
        self,
        config: PositionSizerConfig = None,
        correlation_lookback: int = 50
    ):
        super().__init__(config)
        self.correlation_lookback = correlation_lookback
        self.returns_history: Dict[str, deque] = {}

    def update_return(self, symbol: str, return_pct: float):
        """Update returns for correlation calculation."""
        if symbol not in self.returns_history:
            self.returns_history[symbol] = deque(maxlen=self.correlation_lookback)
        self.returns_history[symbol].append(return_pct)

    def get_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix between assets."""
        symbols = list(self.returns_history.keys())
        n_assets = len(symbols)

        if n_assets < 2:
            return np.eye(n_assets)

        # Build returns matrix
        min_len = min(len(self.returns_history[s]) for s in symbols)

        if min_len < 10:
            return np.eye(n_assets)

        returns_matrix = np.array([
            list(self.returns_history[s])[-min_len:]
            for s in symbols
        ])

        # Calculate correlation
        corr = np.corrcoef(returns_matrix)

        return corr

    def calculate_correlation_adjustment(
        self,
        symbol: str,
        current_positions: Dict[str, float]
    ) -> float:
        """
        Calculate adjustment based on correlation with existing positions.

        Returns multiplier between 0.5 and 1.0
        """
        if not current_positions:
            return 1.0

        symbols = list(self.returns_history.keys())
        if symbol not in symbols:
            return 1.0

        corr_matrix = self.get_correlation_matrix()
        symbol_idx = symbols.index(symbol)

        # Calculate weighted average correlation with existing positions
        total_weight = 0
        weighted_corr = 0

        for other_symbol, position_size in current_positions.items():
            if other_symbol in symbols and other_symbol != symbol:
                other_idx = symbols.index(other_symbol)
                corr = abs(corr_matrix[symbol_idx, other_idx])
                weighted_corr += corr * position_size
                total_weight += position_size

        if total_weight == 0:
            return 1.0

        avg_corr = weighted_corr / total_weight

        # High correlation = lower position size
        # Corr 0 -> multiplier 1.0
        # Corr 1 -> multiplier 0.5
        adjustment = 1.0 - 0.5 * avg_corr

        return max(0.5, adjustment)
