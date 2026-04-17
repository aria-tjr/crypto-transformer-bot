"""
Proximal Policy Optimization (PPO) agent for trading.
Implements actor-critic with GAE and Sharpe-based rewards.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"

    # PPO specific
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # Exploration
    initial_entropy_coef: float = 0.01
    final_entropy_coef: float = 0.001
    entropy_decay_steps: int = 100000


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared feature extractor.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu"
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
            ])
            in_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Small weights for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (action_logits, value)
        """
        features = self.shared(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value.squeeze(-1)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            Tuple of (action, log_prob, value)
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    """

    def __init__(self, buffer_size: int, obs_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device

        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add experience to buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = float(done)

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float
    ):
        """Compute GAE advantages and returns."""
        last_gae = 0

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Generate batches for training."""
        indices = np.random.permutation(self.buffer_size)

        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                'observations': torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                'actions': torch.LongTensor(self.actions[batch_indices]).to(self.device),
                'old_log_probs': torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                'advantages': torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                'returns': torch.FloatTensor(self.returns[batch_indices]).to(self.device)
            }

    def reset(self):
        """Reset buffer."""
        self.pos = 0
        self.full = False


class PPOAgent:
    """
    PPO agent for trading.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig = None,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()
        self.device = device

        # Network
        self.network = ActorCriticNetwork(
            obs_dim,
            action_dim,
            self.config.hidden_dims,
            self.config.activation
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )

        # Buffer
        self.buffer = RolloutBuffer(
            self.config.n_steps,
            obs_dim,
            device
        )

        # Training state
        self.total_steps = 0
        self.episodes_completed = 0

        # Logging
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient with decay."""
        progress = min(1.0, self.total_steps / self.config.entropy_decay_steps)
        return self.config.initial_entropy_coef + \
               (self.config.final_entropy_coef - self.config.initial_entropy_coef) * progress

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given observation.

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(obs_tensor, deterministic)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in buffer."""
        self.buffer.add(obs, action, reward, value, log_prob, done)
        self.total_steps += 1

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """
        Perform PPO update.

        Returns:
            Dict of training metrics
        """
        # Get last value for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, last_value = self.network(obs_tensor)
            last_value = last_value.item()

        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda
        )

        # Normalize advantages
        self.buffer.advantages = (self.buffer.advantages - self.buffer.advantages.mean()) / \
                                  (self.buffer.advantages.std() + 1e-8)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # PPO epochs
        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch['observations'],
                    batch['actions']
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch['old_log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch['returns'])

                # Entropy bonus
                entropy_coef = self.get_entropy_coef()
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Reset buffer
        self.buffer.reset()

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'entropy_coef': self.get_entropy_coef()
        }

    def train_episode(self, env) -> Dict[str, float]:
        """
        Train for one episode.

        Returns:
            Episode metrics
        """
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select action
            action, log_prob, value = self.select_action(obs)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            self.store_transition(obs, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            # Update if buffer full
            if self.buffer.full:
                self.update(obs)

            if done:
                break

        self.episodes_completed += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths)
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.episodes_completed = checkpoint['episodes_completed']


def create_agent(
    obs_dim: int,
    action_dim: int,
    config: PPOConfig = None,
    device: str = "cpu"
) -> PPOAgent:
    """Factory function to create PPO agent."""
    return PPOAgent(obs_dim, action_dim, config, device)
