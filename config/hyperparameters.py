"""
Model and training hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TransformerHyperparams:
    """Transformer+GRU hybrid model hyperparameters."""
    # Architecture
    input_dim: int = 64  # Number of input features
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    d_ff: int = 512  # Feed-forward dimension
    dropout: float = 0.1

    # GRU component
    gru_hidden: int = 128
    gru_layers: int = 2
    bidirectional: bool = True

    # Output
    output_dim: int = 3  # [direction, magnitude, confidence]

    # Positional encoding
    max_seq_len: int = 500


@dataclass
class GNNHyperparams:
    """Graph Neural Network hyperparameters."""
    # Node features
    node_input_dim: int = 32
    hidden_channels: int = 64
    num_layers: int = 3

    # Graph structure
    num_assets: int = 3  # BTC, ETH, SOL
    num_auxiliary_nodes: int = 2  # Sentiment, macro

    # Attention
    heads: int = 4
    dropout: float = 0.1

    # Temporal
    temporal_window: int = 20

    # Output
    output_dim: int = 3  # Per-asset predictions


@dataclass
class ContrastiveHyperparams:
    """Contrastive pre-training hyperparameters (TF-C style)."""
    # Encoder
    encoder_dim: int = 128
    projection_dim: int = 64

    # Augmentation
    jitter_ratio: float = 0.1
    scaling_ratio: float = 0.1
    permutation_max_segments: int = 5

    # Loss
    temperature: float = 0.07

    # Training
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-3


@dataclass
class PPOHyperparams:
    """PPO agent hyperparameters."""
    # Actor-Critic
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

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
    initial_exploration: float = 1.0
    final_exploration: float = 0.1
    exploration_decay: int = 100000


@dataclass
class RewardHyperparams:
    """Reward function hyperparameters."""
    # Sharpe-based reward
    sharpe_window: int = 20  # Rolling window for Sharpe calculation
    sharpe_weight: float = 1.0

    # Transaction costs
    transaction_cost_weight: float = 0.1
    slippage_estimate_bps: float = 5.0  # 5 basis points

    # Risk penalties
    drawdown_penalty_weight: float = 0.5
    overtrading_penalty_weight: float = 0.2

    # Progressive negative reward for losing positions
    holding_loss_decay: float = 0.95


@dataclass
class MetaLearningHyperparams:
    """MAML-style meta-learning hyperparameters."""
    # Inner loop (task-specific)
    inner_lr: float = 0.01
    inner_steps: int = 5

    # Outer loop (meta)
    meta_lr: float = 1e-3
    meta_batch_size: int = 4  # Number of tasks per update

    # Regime detection
    num_regimes: int = 4  # Bull, bear, sideways, crash
    regime_window: int = 50

    # Adaptation
    adaptation_threshold: float = 0.3  # Confidence for regime switch


@dataclass
class TrainingHyperparams:
    """General training hyperparameters."""
    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Sequence
    sequence_length: int = 100
    prediction_horizon: int = 10  # Predict 10 steps ahead

    # Training
    max_epochs: int = 200
    patience: int = 20  # Early stopping
    min_delta: float = 1e-4

    # Batch
    batch_size: int = 64
    num_workers: int = 4

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Walk-forward validation
    walk_forward_window: int = 14  # 14 days
    retrain_frequency: int = 7  # Retrain every 7 days


@dataclass
class BacktestHyperparams:
    """Backtesting hyperparameters."""
    # Simulation
    initial_capital: float = 10000.0

    # Costs
    maker_fee_bps: float = 1.0  # 0.01%
    taker_fee_bps: float = 6.0  # 0.06%
    slippage_bps: float = 5.0

    # Latency simulation
    latency_ms: Tuple[float, float] = (10.0, 50.0)  # Min, max latency

    # Reporting
    report_frequency: str = "daily"


@dataclass
class Hyperparameters:
    """Container for all hyperparameters."""
    transformer: TransformerHyperparams = field(default_factory=TransformerHyperparams)
    gnn: GNNHyperparams = field(default_factory=GNNHyperparams)
    contrastive: ContrastiveHyperparams = field(default_factory=ContrastiveHyperparams)
    ppo: PPOHyperparams = field(default_factory=PPOHyperparams)
    reward: RewardHyperparams = field(default_factory=RewardHyperparams)
    meta: MetaLearningHyperparams = field(default_factory=MetaLearningHyperparams)
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    backtest: BacktestHyperparams = field(default_factory=BacktestHyperparams)


# Global hyperparameters instance
hyperparams = Hyperparameters()
