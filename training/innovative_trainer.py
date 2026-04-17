"""
Multi-stage training orchestrator.
Implements curriculum learning, contrastive pre-training, and RL training.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
import json

from torch.utils.data import DataLoader, TensorDataset

# TensorBoard disabled due to TensorFlow/NumPy 2.x compatibility issues
# To re-enable: pip install numpy<2 and uncomment the import
HAS_TENSORBOARD = False
SummaryWriter = None

from models.transformer_gru import TransformerGRU, TransformerGRUConfig
from models.gnn_cross_asset import CrossAssetGNN, GNNConfig
from models.contrastive import TFCModel, ContrastiveConfig, ContrastivePretrainer
from models.ensemble import WeightedEnsemble, EnsembleConfig
from agents.ppo_agent import PPOAgent, PPOConfig
from agents.environment import TradingEnvironment, TradingConfig
from utils.device import get_device, get_dtype
from utils.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for training pipeline."""
    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # Data - optimized for 1H horizon (Medium Future) to minimize drawdown
    sequence_length: int = 72  # 6 hours of 5m candles (increased from 48)
    prediction_horizon: int = 12  # 60 min ahead (12 x 5m candles)
    train_split: float = 0.7
    val_split: float = 0.15

    # Label thresholds for 1H swing
    # 0.6% move in 1 hour is significant and covers fees easily
    up_threshold: float = 0.006  # 0.6% up
    down_threshold: float = -0.006  # 0.6% down
    use_adaptive_threshold: bool = True  # Adapt thresholds to volatility

    # Pre-training (reduced for faster iteration - increase for production)
    pretrain_epochs: int = 20
    pretrain_batch_size: int = 64
    pretrain_lr: float = 1e-3

    # Supervised training - INCREASED for larger dataset
    supervised_epochs: int = 80  # Increased from 50
    supervised_batch_size: int = 64
    supervised_lr: float = 3e-4  # Slightly higher LR
    early_stopping_patience: int = 20  # More patience
    label_smoothing: float = 0.1  # Prevent overconfidence

    # Model architecture selection
    model_type: str = 'tcn_attention'  # Testing TCN with attention
    d_model: int = 256  # Reset to 256
    n_layers: int = 4  # Reset to 4
    n_heads: int = 8
    dropout: float = 0.2  # Reset to 0.2

    # RL training
    rl_episodes: int = 250  # Increased from 150
    rl_update_frequency: int = 256

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 3

    # Checkpointing
    save_frequency: int = 10
    keep_best_n: int = 5

    # Advanced training options
    use_hyperparam_search: bool = False  # Run hyperparameter search first
    use_walk_forward: bool = False  # Use walk-forward validation


class DataPreparer:
    """Prepares data for different training stages."""

    def __init__(
        self,
        sequence_length: int = 48,
        prediction_horizon: int = 3,
        up_threshold: float = 0.0015,
        down_threshold: float = -0.0015,
        use_adaptive_threshold: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.use_adaptive_threshold = use_adaptive_threshold

    def prepare_sequences(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.

        Args:
            data: Raw data (timesteps, features)
            labels: Optional labels

        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        class_counts = {0: 0, 1: 0, 2: 0}  # Track class distribution

        # Pre-calculate rolling returns for adaptive threshold
        prices = data[:, 0]  # First column is close price
        returns = np.zeros(len(prices))
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        # Rolling volatility (std of returns) for adaptive threshold
        vol_window = 20
        rolling_vol = np.zeros(len(returns))
        for i in range(vol_window, len(returns)):
            rolling_vol[i] = np.std(returns[i-vol_window:i])

        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            X.append(data[i:i + self.sequence_length])

            if labels is not None:
                y.append(labels[i + self.sequence_length + self.prediction_horizon - 1])
            else:
                # Use future return as label (first column is close price)
                current = data[i + self.sequence_length - 1, 0]
                future = data[i + self.sequence_length + self.prediction_horizon - 1, 0]

                # Calculate percentage return
                if abs(current) > 1e-8:
                    ret = (future - current) / current
                else:
                    ret = 0.0

                # Use adaptive or fixed thresholds
                if self.use_adaptive_threshold:
                    # Threshold scales mildly with volatility (0.2 multiplier instead of 0.5)
                    # This keeps ~30% up, ~30% neutral, ~30% down distribution
                    vol_idx = i + self.sequence_length - 1
                    local_vol = rolling_vol[vol_idx] if vol_idx < len(rolling_vol) else 0.001
                    # Clamp volatility to prevent extreme thresholds
                    clamped_vol = min(local_vol, 0.003)  # Max 0.3% volatility impact
                    adaptive_up = self.up_threshold + 0.2 * clamped_vol
                    adaptive_down = self.down_threshold - 0.2 * clamped_vol
                else:
                    adaptive_up = self.up_threshold
                    adaptive_down = self.down_threshold

                # Classify: 0=down, 1=neutral, 2=up
                if np.isnan(ret) or np.isinf(ret):
                    label = 1  # Neutral for invalid values
                elif ret < adaptive_down:
                    label = 0  # Down
                elif ret > adaptive_up:
                    label = 2  # Up
                else:
                    label = 1  # Neutral

                y.append(label)
                class_counts[label] += 1

        # Log class distribution
        total = sum(class_counts.values())
        if total > 0:
            logger.info(f"Class distribution: Down={class_counts[0]/total:.1%}, "
                       f"Neutral={class_counts[1]/total:.1%}, Up={class_counts[2]/total:.1%}")

        return np.array(X), np.array(y)

    def create_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        train_split: float = 0.7,
        val_split: float = 0.15
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders."""
        n = len(X)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))

        # Convert to tensors
        X_train = torch.FloatTensor(X[:train_end])
        y_train = torch.LongTensor(y[:train_end])
        X_val = torch.FloatTensor(X[train_end:val_end])
        y_val = torch.LongTensor(y[train_end:val_end])
        X_test = torch.FloatTensor(X[val_end:])
        y_test = torch.LongTensor(y[val_end:])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=batch_size
        )

        return train_loader, val_loader, test_loader


class InnovativeTrainer:
    """
    Multi-stage training orchestrator.

    Training stages:
    1. Contrastive pre-training
    2. Supervised fine-tuning
    3. RL policy training
    4. Ensemble creation
    """

    def __init__(
        self,
        config: TrainerConfig = None,
        device: str = None
    ):
        self.config = config or TrainerConfig()
        self.device = device or str(get_device())

        # Create directories
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard (optional)
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.config.log_dir / "tensorboard")
        else:
            self.writer = None
            logger.warning("TensorBoard not available, logging disabled")

        # Models
        self.contrastive_model: Optional[TFCModel] = None
        self.transformer_model: Optional[TransformerGRU] = None
        self.gnn_model: Optional[CrossAssetGNN] = None
        self.ensemble: Optional[WeightedEnsemble] = None
        self.rl_agent: Optional[PPOAgent] = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history: List[Dict] = []

    def _create_model(self, input_dim: int) -> nn.Module:
        """Create model based on config.model_type."""
        from models.tcn import TCN, TCNConfig, TCNAttention

        if self.config.model_type == 'transformer_gru':
            transformer_config = TransformerGRUConfig(
                input_dim=input_dim,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_encoder_layers=self.config.n_layers,
                d_ff=self.config.d_model * 4,
                dropout=self.config.dropout,
                gru_hidden=self.config.d_model,
                output_dim=3
            )
            return TransformerGRU(transformer_config).to(self.device)

        elif self.config.model_type == 'tcn':
            channels = [self.config.d_model] * self.config.n_layers
            tcn_config = TCNConfig(
                input_dim=input_dim,
                num_channels=channels,
                kernel_size=3,
                dropout=self.config.dropout,
                output_dim=3
            )
            return TCN(tcn_config).to(self.device)

        elif self.config.model_type == 'tcn_attention':
            channels = [self.config.d_model] * self.config.n_layers
            tcn_config = TCNConfig(
                input_dim=input_dim,
                num_channels=channels,
                kernel_size=3,
                dropout=self.config.dropout,
                output_dim=3
            )
            return TCNAttention(tcn_config).to(self.device)

        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def pretrain_contrastive(
        self,
        data: np.ndarray,
        epochs: int = None
    ) -> Dict[str, float]:
        """
        Stage 1: Contrastive pre-training.

        Learns representations without labels.
        """
        logger.info("Stage 1: Contrastive Pre-training")

        epochs = epochs or self.config.pretrain_epochs

        # Prepare data (use config thresholds for consistent labeling)
        preparer = DataPreparer(
            self.config.sequence_length,
            self.config.prediction_horizon,
            self.config.up_threshold,
            self.config.down_threshold,
            self.config.use_adaptive_threshold
        )
        X, _ = preparer.prepare_sequences(data)

        # Create model
        contrastive_config = ContrastiveConfig(
            input_dim=X.shape[-1],
            encoder_dim=128,
            projection_dim=64
        )
        self.contrastive_model = TFCModel(contrastive_config).to(self.device)

        # Create pretrainer
        pretrainer = ContrastivePretrainer(
            self.contrastive_model,
            learning_rate=self.config.pretrain_lr,
            device=self.device
        )

        # Create dataloader
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.config.pretrain_batch_size, shuffle=True)

        # Training loop
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                output = self.contrastive_model(x, return_loss=True)
                loss = output['loss']

                pretrainer.optimizer.zero_grad()
                loss.backward()
                pretrainer.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if self.writer:
                self.writer.add_scalar("pretrain/loss", avg_loss, epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint("contrastive_best.pt", {
                    'model': self.contrastive_model.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss
                })

            if epoch % 10 == 0:
                logger.info(f"Pretrain Epoch {epoch}: Loss = {avg_loss:.4f}")

        logger.info(f"Contrastive pre-training complete. Best loss: {best_loss:.4f}")

        return {'final_loss': avg_loss, 'best_loss': best_loss}

    def train_supervised(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        epochs: int = None
    ) -> Dict[str, float]:
        """
        Stage 2: Supervised training.

        Fine-tunes on labeled data.
        """
        logger.info("Stage 2: Supervised Training")

        epochs = epochs or self.config.supervised_epochs

        # Prepare data (use config thresholds for balanced labels)
        preparer = DataPreparer(
            self.config.sequence_length,
            self.config.prediction_horizon,
            self.config.up_threshold,
            self.config.down_threshold,
            self.config.use_adaptive_threshold
        )
        X, y = preparer.prepare_sequences(data, labels)

        train_loader, val_loader, test_loader = preparer.create_dataloaders(
            X, y,
            self.config.supervised_batch_size,
            self.config.train_split,
            self.config.val_split
        )

        # Create model based on config.model_type
        self.transformer_model = self._create_model(X.shape[-1])
        logger.info(f"Model type: {self.config.model_type}")
        logger.info(f"Model params: {sum(p.numel() for p in self.transformer_model.parameters()):,}")

        # Transfer weights from contrastive model if available
        if self.contrastive_model:
            self._transfer_contrastive_weights()

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.transformer_model.parameters(),
            lr=self.config.supervised_lr,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y, minlength=3)
        total = len(y)
        # Inverse frequency weighting
        class_weights = total / (3 * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 3  # Normalize
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        logger.info(f"Class distribution: Down={class_counts[0]/total:.1%}, Neutral={class_counts[1]/total:.1%}, Up={class_counts[2]/total:.1%}")
        logger.info(f"Class weights: Down={class_weights[0]:.2f}, Neutral={class_weights[1]:.2f}, Up={class_weights[2]:.2f}")

        # Loss with class weighting AND label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.transformer_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                output = self.transformer_model(batch_x)
                loss = criterion(output['direction'], batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                pred = output['direction'].argmax(dim=-1)
                train_correct += (pred == batch_y).sum().item()
                train_total += batch_y.size(0)

            scheduler.step()

            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)

            # Logging
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            if self.writer:
                self.writer.add_scalar("supervised/train_loss", avg_train_loss, epoch)
                self.writer.add_scalar("supervised/train_acc", train_acc, epoch)
                self.writer.add_scalar("supervised/val_loss", val_loss, epoch)
                self.writer.add_scalar("supervised/val_acc", val_acc, epoch)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("transformer_best.pt", {
                    'model': self.transformer_model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                    f"Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Test evaluation
        test_loss, test_acc = self._validate(test_loader, criterion)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        return {
            'final_train_loss': avg_train_loss,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_acc': test_acc
        }

    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate model on dataloader."""
        self.transformer_model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.transformer_model(batch_x)
                loss = criterion(output['direction'], batch_y)

                total_loss += loss.item()
                pred = output['direction'].argmax(dim=-1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        return total_loss / len(loader), correct / total

    def train_rl(
        self,
        env_data: np.ndarray,
        episodes: int = None
    ) -> Dict[str, float]:
        """
        Stage 3: RL policy training.

        Trains PPO agent in trading environment.
        """
        logger.info("Stage 3: RL Training")

        episodes = episodes or self.config.rl_episodes

        # Create environment
        trading_config = TradingConfig(
            initial_capital=10000,
            max_position_pct=0.1,
            use_sharpe_reward=True
        )
        env = TradingEnvironment(env_data, trading_config)

        # Create agent
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        ppo_config = PPOConfig(
            hidden_dims=[256, 128, 64],
            learning_rate=3e-4,
            n_steps=self.config.rl_update_frequency
        )
        self.rl_agent = PPOAgent(obs_dim, action_dim, ppo_config, self.device)

        # Training loop
        best_reward = float('-inf')
        episode_rewards = []

        for episode in range(episodes):
            metrics = self.rl_agent.train_episode(env)
            episode_rewards.append(metrics['episode_reward'])

            # Logging
            if self.writer:
                self.writer.add_scalar("rl/episode_reward", metrics['episode_reward'], episode)
                self.writer.add_scalar("rl/avg_reward", metrics['avg_reward'], episode)

            if metrics['episode_reward'] > best_reward:
                best_reward = metrics['episode_reward']
                self.rl_agent.save(str(self.config.checkpoint_dir / "ppo_best.pt"))

            if episode % 100 == 0:
                logger.info(
                    f"Episode {episode}: Reward={metrics['episode_reward']:.2f}, "
                    f"Avg={metrics['avg_reward']:.2f}"
                )

        logger.info(f"RL training complete. Best reward: {best_reward:.2f}")

        return {
            'best_reward': best_reward,
            'final_avg_reward': np.mean(episode_rewards[-100:])
        }

    def _transfer_contrastive_weights(self):
        """Transfer learned representations from contrastive model."""
        if self.contrastive_model is None:
            return

        # Transfer time encoder weights to transformer input projection
        try:
            with torch.no_grad():
                contrastive_state = self.contrastive_model.time_encoder.state_dict()
                transformer_state = self.transformer_model.input_proj.state_dict()

                # Match compatible layers
                for key in transformer_state:
                    if key in contrastive_state:
                        if transformer_state[key].shape == contrastive_state[key].shape:
                            transformer_state[key] = contrastive_state[key]

                self.transformer_model.input_proj.load_state_dict(transformer_state)
                logger.info("Transferred contrastive weights to transformer")
        except Exception as e:
            logger.warning(f"Could not transfer weights: {e}")

    def _save_checkpoint(self, name: str, state: Dict):
        """Save a checkpoint."""
        path = self.config.checkpoint_dir / name
        torch.save(state, path)
        logger.debug(f"Saved checkpoint: {path}")

    def train_full_pipeline(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Returns results from all stages.
        """
        logger.info("Starting full training pipeline")
        start_time = datetime.now()

        results = {}

        # Stage 1: Contrastive pre-training
        try:
            results['contrastive'] = self.pretrain_contrastive(data)
        except Exception as e:
            logger.error(f"Contrastive pre-training failed: {e}")
            results['contrastive'] = {'error': str(e)}

        # Stage 2: Supervised training
        try:
            results['supervised'] = self.train_supervised(data, labels)
        except Exception as e:
            logger.error(f"Supervised training failed: {e}")
            results['supervised'] = {'error': str(e)}

        # Stage 3: RL training
        try:
            results['rl'] = self.train_rl(data)
        except Exception as e:
            logger.error(f"RL training failed: {e}")
            results['rl'] = {'error': str(e)}

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        results['elapsed_seconds'] = elapsed
        results['timestamp'] = datetime.now().isoformat()

        # Save results
        with open(self.config.log_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Training complete in {elapsed:.1f} seconds")

        if self.writer:
            self.writer.close()

        return results
