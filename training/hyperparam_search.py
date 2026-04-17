"""
Hyperparameter search for model optimization.
Supports grid search and random search strategies.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
import json
import itertools
from concurrent.futures import ProcessPoolExecutor
import copy

from torch.utils.data import DataLoader, TensorDataset

from models.transformer_gru import TransformerGRU, TransformerGRUConfig
from models.tcn import TCN, TCNConfig, TCNAttention
from utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Defines the hyperparameter search space."""
    # Model architecture
    model_type: List[str] = field(default_factory=lambda: ['transformer_gru', 'tcn', 'tcn_attention'])
    d_model: List[int] = field(default_factory=lambda: [128, 256, 384])
    n_layers: List[int] = field(default_factory=lambda: [3, 4, 5])
    n_heads: List[int] = field(default_factory=lambda: [4, 8])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    # Training
    learning_rate: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128])
    weight_decay: List[float] = field(default_factory=lambda: [1e-5, 1e-4])

    # Label smoothing
    label_smoothing: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.15])


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""
    search_space: SearchSpace = field(default_factory=SearchSpace)
    max_trials: int = 50  # Max combinations to try
    epochs_per_trial: int = 30  # Fewer epochs for quick evaluation
    early_stopping_patience: int = 10
    n_folds: int = 3  # Cross-validation folds
    metric: str = 'val_acc'  # Metric to optimize
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    save_dir: Path = field(default_factory=lambda: Path("checkpoints/search"))
    random_seed: int = 42


class HyperparameterSearch:
    """
    Hyperparameter search engine.

    Supports:
    - Grid search (exhaustive)
    - Random search (sampled)
    - Cross-validation for robust evaluation
    """

    def __init__(
        self,
        config: SearchConfig = None,
        device: str = None
    ):
        self.config = config or SearchConfig()
        self.device = device or str(get_device())
        self.results: List[Dict] = []

        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

    def _generate_grid_configs(self) -> List[Dict]:
        """Generate all combinations for grid search."""
        space = self.config.search_space

        # All parameter combinations
        params = {
            'model_type': space.model_type,
            'd_model': space.d_model,
            'n_layers': space.n_layers,
            'n_heads': space.n_heads,
            'dropout': space.dropout,
            'learning_rate': space.learning_rate,
            'batch_size': space.batch_size,
            'weight_decay': space.weight_decay,
            'label_smoothing': space.label_smoothing
        }

        # Generate all combinations
        keys = list(params.keys())
        values = list(params.values())
        configs = []

        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))

            # Convert to native Python types for PyTorch compatibility
            config['d_model'] = int(config['d_model'])
            config['n_layers'] = int(config['n_layers'])
            config['n_heads'] = int(config['n_heads'])
            config['batch_size'] = int(config['batch_size'])
            config['dropout'] = float(config['dropout'])
            config['learning_rate'] = float(config['learning_rate'])
            config['weight_decay'] = float(config['weight_decay'])
            config['label_smoothing'] = float(config['label_smoothing'])

            # Skip invalid combinations
            if config['model_type'] == 'transformer_gru':
                # d_model must be divisible by n_heads
                if config['d_model'] % config['n_heads'] != 0:
                    continue

            configs.append(config)

        logger.info(f"Grid search: {len(configs)} total combinations")
        return configs

    def _generate_random_configs(self, n_trials: int) -> List[Dict]:
        """Generate random configurations for random search."""
        space = self.config.search_space
        configs = []

        for _ in range(n_trials):
            # Convert numpy types to native Python types for PyTorch compatibility
            config = {
                'model_type': str(np.random.choice(space.model_type)),
                'd_model': int(np.random.choice(space.d_model)),
                'n_layers': int(np.random.choice(space.n_layers)),
                'n_heads': int(np.random.choice(space.n_heads)),
                'dropout': float(np.random.choice(space.dropout)),
                'learning_rate': float(np.random.choice(space.learning_rate)),
                'batch_size': int(np.random.choice(space.batch_size)),
                'weight_decay': float(np.random.choice(space.weight_decay)),
                'label_smoothing': float(np.random.choice(space.label_smoothing))
            }

            # Ensure valid config for transformer
            if config['model_type'] == 'transformer_gru':
                while config['d_model'] % config['n_heads'] != 0:
                    config['n_heads'] = int(np.random.choice(space.n_heads))

            configs.append(config)

        logger.info(f"Random search: {n_trials} configurations")
        return configs

    def _create_model(self, params: Dict, input_dim: int) -> nn.Module:
        """Create model from hyperparameters."""
        model_type = params['model_type']

        if model_type == 'transformer_gru':
            config = TransformerGRUConfig(
                input_dim=input_dim,
                d_model=params['d_model'],
                n_heads=params['n_heads'],
                n_encoder_layers=params['n_layers'],
                d_ff=params['d_model'] * 4,
                dropout=params['dropout'],
                gru_hidden=params['d_model'],
                output_dim=3
            )
            return TransformerGRU(config)

        elif model_type == 'tcn':
            # TCN channels based on d_model
            channels = [params['d_model']] * params['n_layers']
            config = TCNConfig(
                input_dim=input_dim,
                num_channels=channels,
                kernel_size=3,
                dropout=params['dropout'],
                output_dim=3
            )
            return TCN(config)

        elif model_type == 'tcn_attention':
            channels = [params['d_model']] * params['n_layers']
            config = TCNConfig(
                input_dim=input_dim,
                num_channels=channels,
                kernel_size=3,
                dropout=params['dropout'],
                output_dim=3
            )
            return TCNAttention(config)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _train_fold(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        params: Dict
    ) -> Dict[str, float]:
        """Train one fold and return metrics."""
        model = model.to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.epochs_per_trial
        )

        criterion = nn.CrossEntropyLoss(
            label_smoothing=params['label_smoothing']
        )

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(self.config.epochs_per_trial):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output['direction'], batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                pred = output['direction'].argmax(dim=-1)
                train_correct += (pred == batch_y).sum().item()
                train_total += batch_y.size(0)

            scheduler.step()

            # Validate
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    output = model(batch_x)
                    loss = criterion(output['direction'], batch_y)

                    val_loss += loss.item()
                    pred = output['direction'].argmax(dim=-1)
                    val_correct += (pred == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_acc = val_correct / val_total

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

        return {
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_correct / train_total,
            'val_loss': val_loss / len(val_loader),
            'val_acc': best_val_acc
        }

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict
    ) -> Dict[str, float]:
        """Run k-fold cross-validation."""
        n = len(X)
        fold_size = n // self.config.n_folds
        fold_results = []

        for fold in range(self.config.n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size

            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
            y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)

            # Create dataloaders
            train_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.LongTensor(y_train)
                ),
                batch_size=params['batch_size'],
                shuffle=True
            )

            val_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.LongTensor(y_val)
                ),
                batch_size=params['batch_size']
            )

            # Create and train model
            model = self._create_model(params, X.shape[-1])
            metrics = self._train_fold(model, train_loader, val_loader, params)
            fold_results.append(metrics)

            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Average across folds
        avg_results = {}
        for key in fold_results[0]:
            values = [r[key] for r in fold_results]
            avg_results[key] = np.mean(values)
            avg_results[f'{key}_std'] = np.std(values)

        return avg_results

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'random'
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search.

        Args:
            X: Training data (samples, seq_len, features)
            y: Labels
            method: 'grid' or 'random'

        Returns:
            Best configuration and results
        """
        logger.info(f"Starting {method} search")
        start_time = datetime.now()

        # Generate configurations
        if method == 'grid':
            configs = self._generate_grid_configs()
            # Limit to max_trials
            if len(configs) > self.config.max_trials:
                np.random.shuffle(configs)
                configs = configs[:self.config.max_trials]
        else:
            configs = self._generate_random_configs(self.config.max_trials)

        # Run trials
        best_metric = float('-inf') if self.config.direction == 'maximize' else float('inf')
        best_config = None
        best_results = None

        for i, params in enumerate(configs):
            logger.info(f"Trial {i+1}/{len(configs)}: {params}")

            try:
                results = self._cross_validate(X, y, params)
                results['params'] = params
                self.results.append(results)

                # Check if best
                metric_value = results[self.config.metric]
                is_better = (
                    (self.config.direction == 'maximize' and metric_value > best_metric) or
                    (self.config.direction == 'minimize' and metric_value < best_metric)
                )

                if is_better:
                    best_metric = metric_value
                    best_config = params
                    best_results = results
                    logger.info(f"  New best! {self.config.metric}={metric_value:.4f}")

            except Exception as e:
                logger.error(f"  Trial failed: {e}")
                continue

        # Save results
        elapsed = (datetime.now() - start_time).total_seconds()

        final_results = {
            'best_config': best_config,
            'best_results': best_results,
            'best_metric': best_metric,
            'all_results': self.results,
            'elapsed_seconds': elapsed,
            'n_trials': len(self.results),
            'timestamp': datetime.now().isoformat()
        }

        # Save to file
        with open(self.config.save_dir / "search_results.json", "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"Search complete in {elapsed:.1f}s")
        logger.info(f"Best config: {best_config}")
        logger.info(f"Best {self.config.metric}: {best_metric:.4f}")

        return final_results

    def get_best_model(self, input_dim: int) -> nn.Module:
        """Get the best model from search results."""
        if not self.results:
            raise ValueError("No search results available")

        # Find best
        metric = self.config.metric
        if self.config.direction == 'maximize':
            best = max(self.results, key=lambda x: x[metric])
        else:
            best = min(self.results, key=lambda x: x[metric])

        return self._create_model(best['params'], input_dim)


def quick_search(
    X: np.ndarray,
    y: np.ndarray,
    max_trials: int = 20,
    device: str = None
) -> Dict[str, Any]:
    """
    Quick hyperparameter search with reduced search space.

    Good for rapid iteration.
    """
    # Reduced search space for quick search
    space = SearchSpace(
        model_type=['transformer_gru', 'tcn_attention'],
        d_model=[128, 256],
        n_layers=[3, 4],
        n_heads=[8],
        dropout=[0.1, 0.2],
        learning_rate=[3e-4, 1e-3],
        batch_size=[64],
        weight_decay=[1e-5],
        label_smoothing=[0.1]
    )

    config = SearchConfig(
        search_space=space,
        max_trials=max_trials,
        epochs_per_trial=20,
        n_folds=2
    )

    searcher = HyperparameterSearch(config, device)
    return searcher.search(X, y, method='random')
