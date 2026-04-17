"""
Trading environment for reinforcement learning.
Implements a Gymnasium-compatible environment with realistic market simulation.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces


class Action(IntEnum):
    """Trading actions."""
    SELL = 0
    HOLD = 1
    BUY = 2


@dataclass
class Position:
    """Trading position."""
    size: float = 0.0  # Positive = long, negative = short
    entry_price: float = 0.0
    entry_time: int = 0
    unrealized_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return abs(self.size) > 1e-8

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float


@dataclass
class TradingConfig:
    """Configuration for trading environment."""
    initial_capital: float = 10000.0
    max_position_pct: float = 0.1  # Max 10% per position
    transaction_cost_bps: float = 6.0  # 6 bps (0.06%)
    slippage_bps: float = 5.0  # 5 bps slippage
    max_trades_per_episode: int = 50  # Reduced to prevent overtrading
    episode_length: int = 500  # Shorter episodes for faster learning
    reward_scaling: float = 10.0  # Reduced from 100 for stability
    use_sharpe_reward: bool = True
    sharpe_window: int = 50  # Increased window for stability
    reward_clip: float = 5.0  # Clip extreme rewards

    # Advanced reward shaping
    use_sortino: bool = True  # Use Sortino ratio (downside risk only)
    use_profit_factor: bool = True  # Reward for profit/loss ratio
    drawdown_penalty: float = 2.0  # Penalty multiplier for drawdowns
    win_rate_bonus: float = 0.5  # Bonus for maintaining high win rate
    hold_penalty: float = 0.01  # Small penalty for holding to encourage action


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment.

    Observation space:
    - Market features (OHLCV, technical indicators, etc.)
    - Position information
    - Account state

    Action space:
    - Discrete: Sell, Hold, Buy
    - Can extend to continuous for position sizing
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data: np.ndarray,
        config: TradingConfig = None,
        feature_names: List[str] = None
    ):
        """
        Initialize environment.

        Args:
            data: Market data array (timesteps, features)
            config: Trading configuration
            feature_names: Names of features for debugging
        """
        super().__init__()

        self.data = data
        self.config = config or TradingConfig()
        self.feature_names = feature_names

        self.n_timesteps, self.n_features = data.shape

        # Define spaces
        # Observation: features + position info + account state
        obs_dim = self.n_features + 5  # +5 for position, pnl, etc.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)  # Sell, Hold, Buy

        # Initialize state
        self._reset_state()

    def _reset_state(self):
        """Reset internal state."""
        self.current_step = 0
        self.capital = self.config.initial_capital
        self.position = Position()
        self.trades: List[TradeRecord] = []
        self.returns: List[float] = []
        self.equity_curve: List[float] = [self.config.initial_capital]
        self.n_trades = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self._reset_state()

        # Random start position within data
        if options and "start_idx" in options:
            self.current_step = options["start_idx"]
        else:
            max_start = max(0, self.n_timesteps - self.config.episode_length - 1)
            self.current_step = self.np_random.integers(0, max_start + 1)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.

        Args:
            action: Trading action (0=sell, 1=hold, 2=buy)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current and next price
        current_price = self._get_price()

        # Execute action
        self._execute_action(action, current_price)

        # Move to next step
        self.current_step += 1

        # Update position P&L
        if self.position.is_open:
            new_price = self._get_price()
            self._update_position_pnl(new_price)

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        # Market features
        market_features = self.data[self.current_step]

        # Position features
        position_features = np.array([
            self.position.size / self.config.max_position_pct,  # Normalized position
            float(self.position.is_long),
            float(self.position.is_short),
            self.position.unrealized_pnl / self.config.initial_capital,  # Normalized P&L
            (self.capital - self.config.initial_capital) / self.config.initial_capital  # Return
        ], dtype=np.float32)

        return np.concatenate([market_features, position_features]).astype(np.float32)

    def _get_price(self, step: int = None) -> float:
        """Get price at current or specified step."""
        if step is None:
            step = self.current_step

        # Assume first feature is close price or use mid price
        # This should be adapted based on actual data format
        return float(self.data[step, 0])

    def _execute_action(self, action: int, price: float):
        """Execute trading action."""
        if action == Action.BUY:
            if not self.position.is_long:
                self._open_position(1.0, price)
        elif action == Action.SELL:
            if not self.position.is_short:
                self._open_position(-1.0, price)
        # HOLD: do nothing

    def _open_position(self, direction: float, price: float):
        """Open or flip position."""
        # Close existing position if any
        if self.position.is_open:
            self._close_position(price)

        # Apply slippage
        slippage = price * self.config.slippage_bps / 10000
        if direction > 0:
            entry_price = price + slippage
        else:
            entry_price = price - slippage

        # Calculate position size (with safety check)
        position_value = self.capital * self.config.max_position_pct
        if abs(entry_price) < 1e-6:
            return  # Skip if price is invalid
        size = direction * position_value / entry_price

        # Apply transaction cost
        cost = abs(position_value) * self.config.transaction_cost_bps / 10000
        self.capital -= cost

        self.position = Position(
            size=size,
            entry_price=entry_price,
            entry_time=self.current_step
        )

        self.n_trades += 1

    def _close_position(self, price: float):
        """Close current position."""
        if not self.position.is_open:
            return

        # Apply slippage
        slippage = price * self.config.slippage_bps / 10000
        if self.position.is_long:
            exit_price = price - slippage
        else:
            exit_price = price + slippage

        # Calculate P&L
        if self.position.is_long:
            pnl = self.position.size * (exit_price - self.position.entry_price)
        else:
            pnl = self.position.size * (exit_price - self.position.entry_price)

        # Apply transaction cost
        position_value = abs(self.position.size * exit_price)
        cost = position_value * self.config.transaction_cost_bps / 10000

        realized_pnl = pnl - cost
        self.capital += realized_pnl

        # Record trade (with safety check for division)
        denominator = abs(self.position.size) * self.position.entry_price
        return_pct = realized_pnl / denominator if abs(denominator) > 1e-8 else 0.0

        self.trades.append(TradeRecord(
            entry_time=self.position.entry_time,
            exit_time=self.current_step,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            pnl=realized_pnl,
            return_pct=return_pct
        ))

        # Reset position
        self.position = Position()

    def _update_position_pnl(self, price: float):
        """Update unrealized P&L."""
        if not self.position.is_open:
            return

        if self.position.is_long:
            self.position.unrealized_pnl = self.position.size * (price - self.position.entry_price)
        else:
            self.position.unrealized_pnl = self.position.size * (price - self.position.entry_price)

    def _calculate_reward(self) -> float:
        """Calculate step reward with advanced shaping."""
        # Current equity
        equity = self.capital + self.position.unrealized_pnl
        self.equity_curve.append(equity)

        # Calculate return
        if len(self.equity_curve) > 1:
            step_return = (equity - self.equity_curve[-2]) / self.equity_curve[-2]
        else:
            step_return = 0.0

        self.returns.append(step_return)

        reward = 0.0

        # Base reward: risk-adjusted returns
        if len(self.returns) >= self.config.sharpe_window:
            recent_returns = np.array(self.returns[-self.config.sharpe_window:])
            mean_return = np.mean(recent_returns)

            if self.config.use_sortino:
                # Sortino ratio: only penalize downside volatility
                downside_returns = recent_returns[recent_returns < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns) + 1e-8
                else:
                    downside_std = 1e-8
                reward = mean_return / downside_std
            elif self.config.use_sharpe_reward:
                # Standard Sharpe ratio
                std_return = np.std(recent_returns) + 1e-8
                reward = mean_return / std_return
            else:
                reward = mean_return
        else:
            reward = step_return

        # Profit factor bonus
        if self.config.use_profit_factor and len(self.trades) >= 5:
            wins = sum(t.pnl for t in self.trades if t.pnl > 0)
            losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
            if losses > 0:
                profit_factor = wins / losses
                # Bonus for profit factor > 1 (profitable)
                if profit_factor > 1.0:
                    reward += self.config.win_rate_bonus * (profit_factor - 1.0)
                else:
                    reward -= self.config.win_rate_bonus * (1.0 - profit_factor)

        # Drawdown penalty
        if len(self.equity_curve) > 10:
            peak = max(self.equity_curve)
            current_drawdown = (peak - equity) / peak
            if current_drawdown > 0.05:  # More than 5% drawdown
                reward -= self.config.drawdown_penalty * current_drawdown

        # Win rate bonus
        if len(self.trades) >= 5:
            winning_trades = sum(1 for t in self.trades if t.pnl > 0)
            win_rate = winning_trades / len(self.trades)
            if win_rate > 0.5:
                reward += self.config.win_rate_bonus * (win_rate - 0.5)
            else:
                reward -= self.config.win_rate_bonus * (0.5 - win_rate) * 0.5

        # Small penalty for holding (encourage active trading)
        if not self.position.is_open:
            reward -= self.config.hold_penalty

        # Scale reward
        reward *= self.config.reward_scaling

        # Penalty for overtrading
        if self.n_trades > self.config.max_trades_per_episode:
            reward -= 0.5

        # Clip extreme rewards for stability
        reward = np.clip(reward, -self.config.reward_clip, self.config.reward_clip)

        return float(reward)

    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Bankrupt (lost 90% of capital)
        equity = self.capital + self.position.unrealized_pnl
        if equity <= self.config.initial_capital * 0.1:
            return True

        # Large drawdown (30% instead of 20% for more learning time)
        if len(self.equity_curve) > 20:
            max_equity = max(self.equity_curve)
            drawdown = (max_equity - equity) / max_equity
            if drawdown > 0.3:  # 30% drawdown limit
                return True

        return False

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated."""
        # End of data
        if self.current_step >= self.n_timesteps - 1:
            return True

        # Episode length limit
        steps_taken = self.current_step - (self.equity_curve[0] if len(self.equity_curve) > 0 else 0)
        if steps_taken >= self.config.episode_length:
            return True

        return False

    def _get_info(self) -> Dict:
        """Get step info."""
        equity = self.capital + self.position.unrealized_pnl

        return {
            "step": self.current_step,
            "capital": self.capital,
            "equity": equity,
            "position": self.position.size,
            "unrealized_pnl": self.position.unrealized_pnl,
            "n_trades": self.n_trades,
            "total_return": (equity - self.config.initial_capital) / self.config.initial_capital
        }

    def render(self):
        """Render environment state."""
        info = self._get_info()
        print(f"Step: {info['step']}, Equity: ${info['equity']:.2f}, "
              f"Position: {info['position']:.4f}, Trades: {info['n_trades']}")

    def get_performance_summary(self) -> Dict:
        """Get overall performance metrics."""
        equity_curve = np.array(self.equity_curve)
        returns = np.array(self.returns) if self.returns else np.array([0.0])

        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if len(equity_curve) > 0 else 0

        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
        else:
            sharpe = 0

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Trade stats
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            win_rate = len(wins) / len(self.trades)
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            losses = [t for t in self.trades if t.pnl < 0]
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "n_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "final_equity": equity_curve[-1] if len(equity_curve) > 0 else self.config.initial_capital
        }


class MultiAssetEnvironment(TradingEnvironment):
    """
    Multi-asset trading environment for BTC/ETH/SOL.

    Extends base environment to handle multiple assets simultaneously.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        config: TradingConfig = None,
        symbols: List[str] = None
    ):
        """
        Initialize multi-asset environment.

        Args:
            data: Dict mapping symbol to data array
            config: Trading configuration
            symbols: List of symbols to trade
        """
        self.symbols = symbols or list(data.keys())
        self.multi_data = data

        # Concatenate all data for observation
        combined_data = np.concatenate([data[s] for s in self.symbols], axis=1)

        super().__init__(combined_data, config)

        # Per-asset positions
        self.positions: Dict[str, Position] = {s: Position() for s in self.symbols}
        self.asset_trades: Dict[str, List[TradeRecord]] = {s: [] for s in self.symbols}

        # Expand action space: 3 actions per asset
        self.action_space = spaces.MultiDiscrete([3] * len(self.symbols))

        # Update observation space
        obs_dim = sum(d.shape[1] for d in data.values()) + 5 * len(self.symbols)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _reset_state(self):
        """Reset internal state for multi-asset."""
        super()._reset_state()
        self.positions = {s: Position() for s in self.symbols}
        self.asset_trades = {s: [] for s in self.symbols}

    def _get_observation(self) -> np.ndarray:
        """Build observation for all assets."""
        features = []

        for symbol in self.symbols:
            # Market features for this asset
            asset_data = self.multi_data[symbol][self.current_step]
            features.append(asset_data)

            # Position features for this asset
            pos = self.positions[symbol]
            pos_features = np.array([
                pos.size / self.config.max_position_pct,
                float(pos.is_long),
                float(pos.is_short),
                pos.unrealized_pnl / self.config.initial_capital,
                (self.capital - self.config.initial_capital) / self.config.initial_capital
            ])
            features.append(pos_features)

        return np.concatenate(features).astype(np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute actions for all assets."""
        # Execute action for each asset
        for i, symbol in enumerate(self.symbols):
            action = actions[i]
            price = float(self.multi_data[symbol][self.current_step, 0])
            self._execute_asset_action(symbol, action, price)

        self.current_step += 1

        # Update all positions
        for symbol in self.symbols:
            if self.positions[symbol].is_open:
                price = float(self.multi_data[symbol][self.current_step, 0])
                self._update_asset_pnl(symbol, price)

        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_asset_action(self, symbol: str, action: int, price: float):
        """Execute action for a specific asset."""
        pos = self.positions[symbol]

        if action == Action.BUY and not pos.is_long:
            self._open_asset_position(symbol, 1.0, price)
        elif action == Action.SELL and not pos.is_short:
            self._open_asset_position(symbol, -1.0, price)

    def _open_asset_position(self, symbol: str, direction: float, price: float):
        """Open position for specific asset."""
        if self.positions[symbol].is_open:
            self._close_asset_position(symbol, price)

        # Per-asset position sizing (divide by number of assets)
        position_pct = self.config.max_position_pct / len(self.symbols)
        position_value = self.capital * position_pct

        slippage = price * self.config.slippage_bps / 10000
        entry_price = price + slippage * direction

        size = direction * position_value / entry_price
        cost = abs(position_value) * self.config.transaction_cost_bps / 10000
        self.capital -= cost

        self.positions[symbol] = Position(
            size=size,
            entry_price=entry_price,
            entry_time=self.current_step
        )

        self.n_trades += 1

    def _close_asset_position(self, symbol: str, price: float):
        """Close position for specific asset."""
        pos = self.positions[symbol]
        if not pos.is_open:
            return

        slippage = price * self.config.slippage_bps / 10000
        exit_price = price - slippage * np.sign(pos.size)

        pnl = pos.size * (exit_price - pos.entry_price)
        cost = abs(pos.size * exit_price) * self.config.transaction_cost_bps / 10000

        self.capital += pnl - cost

        self.asset_trades[symbol].append(TradeRecord(
            entry_time=pos.entry_time,
            exit_time=self.current_step,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl - cost,
            return_pct=(pnl - cost) / abs(pos.size * pos.entry_price)
        ))

        self.positions[symbol] = Position()

    def _update_asset_pnl(self, symbol: str, price: float):
        """Update unrealized P&L for asset."""
        pos = self.positions[symbol]
        if pos.is_open:
            pos.unrealized_pnl = pos.size * (price - pos.entry_price)


def create_env(
    data: np.ndarray,
    config: TradingConfig = None
) -> TradingEnvironment:
    """Factory function to create trading environment."""
    return TradingEnvironment(data, config)
