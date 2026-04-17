"""
Meta-learning for rapid strategy adaptation.
Implements MAML-style learning with regime detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from copy import deepcopy
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    VOLATILE = 3


@dataclass
class MetaConfig:
    """Meta-learning configuration."""
    # Inner loop (task-specific)
    inner_lr: float = 0.01
    inner_steps: int = 5

    # Outer loop (meta)
    meta_lr: float = 1e-3
    meta_batch_size: int = 4

    # Regime detection
    regime_window: int = 50
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.01

    # Adaptation
    adaptation_threshold: float = 0.3
    adaptation_samples: int = 20


class RegimeDetector(nn.Module):
    """
    Detects current market regime from recent price data.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(MarketRegime))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get regime probabilities."""
        return F.softmax(self.network(x), dim=-1)

    def detect(self, features: torch.Tensor) -> Tuple[MarketRegime, float]:
        """
        Detect regime with confidence.

        Returns:
            Tuple of (regime, confidence)
        """
        with torch.no_grad():
            probs = self.forward(features)
            confidence, regime_idx = probs.max(dim=-1)

        return MarketRegime(regime_idx.item()), confidence.item()


class RuleBasedRegimeDetector:
    """
    Simple rule-based regime detection using price statistics.
    """

    def __init__(self, config: MetaConfig):
        self.config = config
        self.price_history: deque = deque(maxlen=config.regime_window)

    def update(self, price: float):
        """Update with new price."""
        self.price_history.append(price)

    def detect(self) -> Tuple[MarketRegime, float]:
        """
        Detect current regime.

        Returns:
            Tuple of (regime, confidence)
        """
        if len(self.price_history) < 20:
            return MarketRegime.SIDEWAYS, 0.5

        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]

        # Calculate metrics
        volatility = np.std(returns)
        trend = (prices[-1] - prices[0]) / prices[0]
        mean_return = np.mean(returns)

        # Classify regime
        if volatility > self.config.volatility_threshold:
            regime = MarketRegime.VOLATILE
            confidence = min(1.0, volatility / self.config.volatility_threshold)
        elif trend > self.config.trend_threshold:
            regime = MarketRegime.BULL
            confidence = min(1.0, trend / self.config.trend_threshold)
        elif trend < -self.config.trend_threshold:
            regime = MarketRegime.BEAR
            confidence = min(1.0, abs(trend) / self.config.trend_threshold)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 1.0 - abs(trend) / self.config.trend_threshold

        return regime, confidence


class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning trainer.

    Learns to quickly adapt to new market conditions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MetaConfig,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.meta_lr
        )

        # Store adapted models per regime
        self.regime_models: Dict[MarketRegime, nn.Module] = {}

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        model: nn.Module = None
    ) -> nn.Module:
        """
        Inner loop adaptation on support set.

        Returns:
            Adapted model
        """
        if model is None:
            model = deepcopy(self.model)

        model.train()

        for _ in range(self.config.inner_steps):
            # Forward pass
            outputs = model(support_x)

            # Compute loss (adjust based on model type)
            if isinstance(outputs, dict):
                if 'direction' in outputs:
                    loss = F.cross_entropy(outputs['direction'], support_y)
                else:
                    loss = outputs.get('loss', F.mse_loss(outputs['pred'], support_y))
            else:
                loss = F.mse_loss(outputs, support_y)

            # Manual gradient step
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            with torch.no_grad():
                for param, grad in zip(model.parameters(), grads):
                    param.data = param.data - self.config.inner_lr * grad

        return model

    def meta_update(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Outer loop meta-update.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Meta loss
        """
        self.meta_optimizer.zero_grad()

        total_loss = 0

        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to task
            adapted_model = self.inner_loop(support_x, support_y)

            # Evaluate on query set
            outputs = adapted_model(query_x)

            if isinstance(outputs, dict):
                if 'direction' in outputs:
                    loss = F.cross_entropy(outputs['direction'], query_y)
                else:
                    loss = outputs.get('loss', F.mse_loss(outputs['pred'], query_y))
            else:
                loss = F.mse_loss(outputs, query_y)

            total_loss += loss

        # Average loss
        meta_loss = total_loss / len(tasks)

        # Meta-update
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_to_regime(
        self,
        regime: MarketRegime,
        data: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        Adapt model to specific regime.

        Args:
            regime: Target market regime
            data: Tuple of (features, targets) for adaptation
        """
        features, targets = data
        adapted = self.inner_loop(features, targets)
        self.regime_models[regime] = adapted

    def get_regime_model(self, regime: MarketRegime) -> nn.Module:
        """Get model adapted for specific regime."""
        if regime in self.regime_models:
            return self.regime_models[regime]
        return self.model


class MetaLearningAgent:
    """
    Agent that uses meta-learning for rapid adaptation.
    """

    def __init__(
        self,
        base_agent,
        config: MetaConfig = None,
        device: str = "cpu"
    ):
        self.base_agent = base_agent
        self.config = config or MetaConfig()
        self.device = device

        # Regime detection
        self.regime_detector = RuleBasedRegimeDetector(config)
        self.learned_detector = RegimeDetector().to(device)

        # Meta-trainer for base agent's network
        self.maml = MAMLTrainer(
            base_agent.network,
            config,
            device
        )

        # Current state
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.adaptation_buffer: List[Tuple] = []

        # Performance tracking per regime
        self.regime_performance: Dict[MarketRegime, deque] = {
            regime: deque(maxlen=100) for regime in MarketRegime
        }

    def update_regime(self, price: float, features: torch.Tensor = None):
        """
        Update regime detection with new data.

        Args:
            price: Current price
            features: Optional feature vector for learned detection
        """
        self.regime_detector.update(price)

        # Get regime from rule-based detector
        rule_regime, rule_conf = self.regime_detector.detect()

        # If we have features, also use learned detector
        if features is not None:
            learned_regime, learned_conf = self.learned_detector.detect(features)

            # Combine (prefer learned if confident)
            if learned_conf > 0.7:
                self.current_regime = learned_regime
                self.regime_confidence = learned_conf
            else:
                self.current_regime = rule_regime
                self.regime_confidence = rule_conf
        else:
            self.current_regime = rule_regime
            self.regime_confidence = rule_conf

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using regime-appropriate model.

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Get regime-adapted model if available
        adapted_network = self.maml.get_regime_model(self.current_regime)

        # Temporarily swap network
        original_network = self.base_agent.network
        self.base_agent.network = adapted_network

        # Get action
        action, log_prob, value = self.base_agent.select_action(obs, deterministic)

        # Restore original network
        self.base_agent.network = original_network

        return action, log_prob, value

    def add_adaptation_sample(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        target: int
    ):
        """Add sample for online adaptation."""
        self.adaptation_buffer.append((obs, action, reward, target))

        # Adapt when buffer is full
        if len(self.adaptation_buffer) >= self.config.adaptation_samples:
            self._adapt()
            self.adaptation_buffer = []

    def _adapt(self):
        """Perform online adaptation to current regime."""
        if not self.adaptation_buffer:
            return

        # Prepare data
        obs = torch.FloatTensor([s[0] for s in self.adaptation_buffer]).to(self.device)
        targets = torch.LongTensor([s[3] for s in self.adaptation_buffer]).to(self.device)

        # Adapt model for current regime
        self.maml.adapt_to_regime(
            self.current_regime,
            (obs, targets)
        )

    def record_performance(self, reward: float):
        """Record performance for current regime."""
        self.regime_performance[self.current_regime].append(reward)

    def get_regime_stats(self) -> Dict[str, Dict]:
        """Get performance statistics per regime."""
        stats = {}

        for regime in MarketRegime:
            perf = list(self.regime_performance[regime])
            if perf:
                stats[regime.name] = {
                    'mean_reward': np.mean(perf),
                    'std_reward': np.std(perf),
                    'n_samples': len(perf)
                }
            else:
                stats[regime.name] = {
                    'mean_reward': 0.0,
                    'std_reward': 0.0,
                    'n_samples': 0
                }

        return stats

    def should_adapt(self) -> bool:
        """Check if adaptation is needed based on recent performance."""
        recent_perf = list(self.regime_performance[self.current_regime])[-20:]

        if len(recent_perf) < 10:
            return False

        # Adapt if performance is poor
        mean_perf = np.mean(recent_perf)
        return mean_perf < self.config.adaptation_threshold

    def save(self, path: str):
        """Save meta-learner state."""
        torch.save({
            'base_model': self.base_agent.network.state_dict(),
            'regime_models': {
                regime.value: model.state_dict()
                for regime, model in self.maml.regime_models.items()
            },
            'learned_detector': self.learned_detector.state_dict(),
            'regime_performance': {
                regime.value: list(perf)
                for regime, perf in self.regime_performance.items()
            }
        }, path)

    def load(self, path: str):
        """Load meta-learner state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.base_agent.network.load_state_dict(checkpoint['base_model'])

        for regime_val, state_dict in checkpoint['regime_models'].items():
            regime = MarketRegime(regime_val)
            model = deepcopy(self.base_agent.network)
            model.load_state_dict(state_dict)
            self.maml.regime_models[regime] = model

        self.learned_detector.load_state_dict(checkpoint['learned_detector'])


def create_meta_agent(
    base_agent,
    config: MetaConfig = None,
    device: str = "cpu"
) -> MetaLearningAgent:
    """Factory function to create meta-learning agent."""
    return MetaLearningAgent(base_agent, config, device)
