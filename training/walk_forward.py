"""
Walk-forward validation for non-stationary time series.
Implements rolling train/test windows with regime stratification.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import calculate_all_metrics, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Window sizes (in samples)
    train_window: int = 5000  # ~7 days at 1-min data
    test_window: int = 1000  # ~1.5 days
    step_size: int = 500  # Advance by ~12 hours

    # Minimum requirements
    min_train_samples: int = 1000
    min_test_samples: int = 100

    # Regime stratification
    stratify_by_regime: bool = True
    regime_column: Optional[str] = None

    # Performance thresholds
    min_sharpe_threshold: float = 1.0
    max_drawdown_threshold: float = 0.15


@dataclass
class WalkForwardFold:
    """Single walk-forward fold."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_samples: int
    test_samples: int
    regime: Optional[str] = None


@dataclass
class FoldResult:
    """Results from a single fold."""
    fold: WalkForwardFold
    metrics: PerformanceMetrics
    predictions: np.ndarray
    actuals: np.ndarray
    passed_threshold: bool


class WalkForwardValidator:
    """
    Implements walk-forward validation for trading strategies.

    Features:
    - Rolling train/test windows
    - Regime-stratified evaluation
    - Performance tracking across folds
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.folds: List[WalkForwardFold] = []
        self.results: List[FoldResult] = []

    def generate_folds(
        self,
        data_length: int,
        regime_labels: Optional[np.ndarray] = None
    ) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds.

        Args:
            data_length: Total number of samples
            regime_labels: Optional regime labels for each sample

        Returns:
            List of WalkForwardFold objects
        """
        self.folds = []

        start = 0
        fold_id = 0

        while start + self.config.train_window + self.config.test_window <= data_length:
            train_start = start
            train_end = start + self.config.train_window
            test_start = train_end
            test_end = min(test_start + self.config.test_window, data_length)

            # Check minimum samples
            if (train_end - train_start) < self.config.min_train_samples:
                start += self.config.step_size
                continue

            if (test_end - test_start) < self.config.min_test_samples:
                break

            # Determine regime
            regime = None
            if regime_labels is not None and self.config.stratify_by_regime:
                test_regimes = regime_labels[test_start:test_end]
                regime = self._get_dominant_regime(test_regimes)

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_samples=train_end - train_start,
                test_samples=test_end - test_start,
                regime=regime
            )

            self.folds.append(fold)
            fold_id += 1
            start += self.config.step_size

        logger.info(f"Generated {len(self.folds)} walk-forward folds")

        return self.folds

    def _get_dominant_regime(self, regimes: np.ndarray) -> str:
        """Get most common regime in window."""
        unique, counts = np.unique(regimes, return_counts=True)
        return str(unique[np.argmax(counts)])

    def iterate_folds(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[WalkForwardFold, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Iterate through folds, yielding train/test splits.

        Yields:
            Tuple of (fold, X_train, X_test, y_train, y_test)
        """
        for fold in self.folds:
            X_train = X[fold.train_start:fold.train_end]
            X_test = X[fold.test_start:fold.test_end]

            if y is not None:
                y_train = y[fold.train_start:fold.train_end]
                y_test = y[fold.test_start:fold.test_end]
            else:
                y_train = None
                y_test = None

            yield fold, X_train, X_test, y_train, y_test

    def record_result(
        self,
        fold: WalkForwardFold,
        predictions: np.ndarray,
        actuals: np.ndarray,
        equity_curve: np.ndarray,
        pnl: np.ndarray
    ):
        """
        Record results from a fold evaluation.

        Args:
            fold: The evaluated fold
            predictions: Model predictions
            actuals: Actual values/labels
            equity_curve: Portfolio equity over time
            pnl: Trade P&L values
        """
        # Calculate metrics
        metrics = calculate_all_metrics(pnl, equity_curve)

        # Check if passed thresholds
        passed = (
            metrics.sharpe_ratio >= self.config.min_sharpe_threshold and
            metrics.max_drawdown <= self.config.max_drawdown_threshold
        )

        result = FoldResult(
            fold=fold,
            metrics=metrics,
            predictions=predictions,
            actuals=actuals,
            passed_threshold=passed
        )

        self.results.append(result)

        logger.info(
            f"Fold {fold.fold_id}: Sharpe={metrics.sharpe_ratio:.2f}, "
            f"Drawdown={metrics.max_drawdown:.2%}, Passed={passed}"
        )

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics across all folds."""
        if not self.results:
            return {}

        sharpe_ratios = [r.metrics.sharpe_ratio for r in self.results]
        sortino_ratios = [r.metrics.sortino_ratio for r in self.results]
        drawdowns = [r.metrics.max_drawdown for r in self.results]
        returns = [r.metrics.total_return for r in self.results]
        win_rates = [r.metrics.win_rate for r in self.results]

        return {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            'mean_sortino': np.mean(sortino_ratios),
            'mean_max_drawdown': np.mean(drawdowns),
            'worst_drawdown': np.max(drawdowns),
            'mean_return': np.mean(returns),
            'total_return': np.sum(returns),
            'mean_win_rate': np.mean(win_rates),
            'folds_passed': sum(1 for r in self.results if r.passed_threshold),
            'total_folds': len(self.results),
            'pass_rate': sum(1 for r in self.results if r.passed_threshold) / len(self.results)
        }

    def get_regime_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics grouped by market regime."""
        if not self.results:
            return {}

        regime_results: Dict[str, List[FoldResult]] = {}

        for result in self.results:
            regime = result.fold.regime or 'unknown'
            if regime not in regime_results:
                regime_results[regime] = []
            regime_results[regime].append(result)

        regime_metrics = {}
        for regime, results in regime_results.items():
            sharpes = [r.metrics.sharpe_ratio for r in results]
            returns = [r.metrics.total_return for r in results]

            regime_metrics[regime] = {
                'count': len(results),
                'mean_sharpe': np.mean(sharpes),
                'mean_return': np.mean(returns),
                'pass_rate': sum(1 for r in results if r.passed_threshold) / len(results)
            }

        return regime_metrics

    def generate_report(self) -> str:
        """Generate text report of validation results."""
        if not self.results:
            return "No results to report."

        agg = self.get_aggregate_metrics()
        regime = self.get_regime_metrics()

        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("AGGREGATE METRICS")
        report.append("-" * 40)
        report.append(f"Total Folds:        {agg['total_folds']}")
        report.append(f"Passed Folds:       {agg['folds_passed']} ({agg['pass_rate']:.1%})")
        report.append(f"Mean Sharpe:        {agg['mean_sharpe']:.2f} (+/- {agg['std_sharpe']:.2f})")
        report.append(f"Mean Sortino:       {agg['mean_sortino']:.2f}")
        report.append(f"Mean Return:        {agg['mean_return']:.2%}")
        report.append(f"Mean Max Drawdown:  {agg['mean_max_drawdown']:.2%}")
        report.append(f"Worst Drawdown:     {agg['worst_drawdown']:.2%}")
        report.append(f"Mean Win Rate:      {agg['mean_win_rate']:.1%}")
        report.append("")

        if regime:
            report.append("REGIME BREAKDOWN")
            report.append("-" * 40)
            for regime_name, metrics in regime.items():
                report.append(f"{regime_name.upper()}:")
                report.append(f"  Count:      {metrics['count']}")
                report.append(f"  Sharpe:     {metrics['mean_sharpe']:.2f}")
                report.append(f"  Return:     {metrics['mean_return']:.2%}")
                report.append(f"  Pass Rate:  {metrics['pass_rate']:.1%}")
            report.append("")

        report.append("FOLD DETAILS")
        report.append("-" * 40)
        for result in self.results[-10:]:  # Last 10 folds
            f = result.fold
            m = result.metrics
            status = "PASS" if result.passed_threshold else "FAIL"
            report.append(
                f"Fold {f.fold_id}: Sharpe={m.sharpe_ratio:.2f}, "
                f"DD={m.max_drawdown:.1%}, Win={m.win_rate:.1%} [{status}]"
            )

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


class ExpandingWindowValidator(WalkForwardValidator):
    """
    Expanding window validation (growing training set).

    Training window starts at min_train and grows each fold.
    """

    def generate_folds(
        self,
        data_length: int,
        regime_labels: Optional[np.ndarray] = None
    ) -> List[WalkForwardFold]:
        """Generate expanding window folds."""
        self.folds = []

        fold_id = 0
        train_start = 0
        train_end = self.config.train_window

        while train_end + self.config.test_window <= data_length:
            test_start = train_end
            test_end = min(test_start + self.config.test_window, data_length)

            if (test_end - test_start) < self.config.min_test_samples:
                break

            regime = None
            if regime_labels is not None and self.config.stratify_by_regime:
                test_regimes = regime_labels[test_start:test_end]
                regime = self._get_dominant_regime(test_regimes)

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,  # Always starts at 0
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_samples=train_end - train_start,
                test_samples=test_end - test_start,
                regime=regime
            )

            self.folds.append(fold)
            fold_id += 1
            train_end += self.config.step_size

        logger.info(f"Generated {len(self.folds)} expanding window folds")

        return self.folds


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation for time series.

    Implements purging and embargo to prevent data leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 10,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test indices.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            test_indices = indices[test_start:test_end]

            # Training indices with purge and embargo
            train_mask = np.ones(n_samples, dtype=bool)

            # Purge: remove samples near test set
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)
            train_mask[purge_start:purge_end] = False

            # Embargo: remove samples after test end
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices


@dataclass
class ModelWalkForwardConfig:
    """Configuration for model-based walk-forward validation."""
    # Window sizes
    train_window: int = 15000  # Training window (~52 days of 5m)
    val_window: int = 2000  # Validation window (~7 days)
    test_window: int = 3000  # Test window (~10 days)
    step_size: int = 2000  # Step forward (~7 days)

    # Model type
    model_type: str = 'transformer_gru'  # or 'tcn', 'tcn_attention'
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.2

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    early_stopping_patience: int = 15
    label_smoothing: float = 0.1

    # Data
    sequence_length: int = 48
    prediction_horizon: int = 3
    up_threshold: float = 0.0015
    down_threshold: float = -0.0015

    # Output
    save_dir: Path = field(default_factory=lambda: Path("checkpoints/walk_forward"))


class ModelWalkForwardValidator:
    """
    Walk-forward validation with model training.

    Trains a fresh model on each window to simulate real trading.
    """

    def __init__(
        self,
        config: ModelWalkForwardConfig = None,
        device: str = None
    ):
        self.config = config or ModelWalkForwardConfig()
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.window_results: List[Dict] = []

        self.config.save_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences with adaptive threshold labeling."""
        X, y = [], []

        prices = data[:, 0]
        returns = np.zeros(len(prices))
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        vol_window = 20
        rolling_vol = np.zeros(len(returns))
        for i in range(vol_window, len(returns)):
            rolling_vol[i] = np.std(returns[i-vol_window:i])

        seq_len = self.config.sequence_length
        horizon = self.config.prediction_horizon

        for i in range(len(data) - seq_len - horizon):
            X.append(data[i:i + seq_len])

            current = data[i + seq_len - 1, 0]
            future = data[i + seq_len + horizon - 1, 0]

            ret = (future - current) / current if abs(current) > 1e-8 else 0.0

            vol_idx = i + seq_len - 1
            local_vol = rolling_vol[vol_idx] if vol_idx < len(rolling_vol) else 0.001
            clamped_vol = min(local_vol, 0.003)
            adaptive_up = self.config.up_threshold + 0.2 * clamped_vol
            adaptive_down = self.config.down_threshold - 0.2 * clamped_vol

            if np.isnan(ret) or np.isinf(ret):
                label = 1
            elif ret < adaptive_down:
                label = 0
            elif ret > adaptive_up:
                label = 2
            else:
                label = 1

            y.append(label)

        return np.array(X), np.array(y)

    def _create_model(self, input_dim: int) -> nn.Module:
        """Create model based on config."""
        from models.transformer_gru import TransformerGRU, TransformerGRUConfig
        from models.tcn import TCN, TCNConfig, TCNAttention

        if self.config.model_type == 'transformer_gru':
            model_config = TransformerGRUConfig(
                input_dim=input_dim,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_encoder_layers=self.config.n_layers,
                d_ff=self.config.d_model * 4,
                dropout=self.config.dropout,
                gru_hidden=self.config.d_model,
                output_dim=3
            )
            return TransformerGRU(model_config)

        elif self.config.model_type in ('tcn', 'tcn_attention'):
            channels = [self.config.d_model] * self.config.n_layers
            model_config = TCNConfig(
                input_dim=input_dim,
                num_channels=channels,
                kernel_size=3,
                dropout=self.config.dropout,
                output_dim=3
            )
            if self.config.model_type == 'tcn_attention':
                return TCNAttention(model_config)
            return TCN(model_config)

        raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _train_window(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[nn.Module, Dict]:
        """Train model on a single window."""
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.config.batch_size
        )

        model = self._create_model(X_train.shape[-1]).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)

        class_counts = np.bincount(y_train, minlength=3)
        class_weights = len(y_train) / (3 * class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 3
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.config.label_smoothing)

        best_val_acc = 0
        best_state = None
        patience = 0

        for epoch in range(self.config.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output['direction'], batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output = model(batch_x)
                    pred = output['direction'].argmax(dim=-1)
                    val_correct += (pred == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_acc = val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if patience >= self.config.early_stopping_patience:
                break

        if best_state:
            model.load_state_dict(best_state)

        return model, {'best_val_acc': best_val_acc, 'epochs_trained': epoch + 1}

    def _evaluate(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        model.eval()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=self.config.batch_size
        )

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                output = model(batch_x)
                preds = output['direction'].argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # Direction accuracy (non-neutral)
        dir_mask = all_labels != 1
        dir_acc = (all_preds[dir_mask] == all_labels[dir_mask]).mean() if dir_mask.sum() > 0 else 0.0

        # Per-class accuracy
        class_acc = {}
        for c in range(3):
            mask = all_labels == c
            class_acc[c] = (all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0

        return {
            'accuracy': float(accuracy),
            'direction_acc': float(dir_acc),
            'down_acc': float(class_acc[0]),
            'neutral_acc': float(class_acc[1]),
            'up_acc': float(class_acc[2]),
            'n_samples': len(all_labels)
        }

    def run(self, data: np.ndarray) -> Dict[str, Any]:
        """Run walk-forward validation."""
        logger.info("Starting model-based walk-forward validation")
        start_time = datetime.now()

        X, y = self._prepare_sequences(data)
        n = len(X)
        logger.info(f"Total samples: {n}")

        total_window = self.config.train_window + self.config.val_window + self.config.test_window
        n_windows = max(1, (n - total_window) // self.config.step_size + 1)
        logger.info(f"Walk-forward windows: {n_windows}")

        for w in range(n_windows):
            start = w * self.config.step_size
            train_end = start + self.config.train_window
            val_end = train_end + self.config.val_window
            test_end = val_end + self.config.test_window

            if test_end > n:
                break

            logger.info(f"Window {w+1}/{n_windows}: train[{start}:{train_end}] val[{train_end}:{val_end}] test[{val_end}:{test_end}]")

            model, train_metrics = self._train_window(
                X[start:train_end], y[start:train_end],
                X[train_end:val_end], y[train_end:val_end]
            )
            logger.info(f"  Val acc: {train_metrics['best_val_acc']:.4f}")

            test_metrics = self._evaluate(model, X[val_end:test_end], y[val_end:test_end])
            logger.info(f"  Test acc: {test_metrics['accuracy']:.4f}, Direction acc: {test_metrics['direction_acc']:.4f}")

            self.window_results.append({
                'window': w,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            })

            torch.save(model.state_dict(), self.config.save_dir / f"model_window_{w}.pt")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = (datetime.now() - start_time).total_seconds()
        test_accs = [r['test_metrics']['accuracy'] for r in self.window_results]
        dir_accs = [r['test_metrics']['direction_acc'] for r in self.window_results]

        summary = {
            'n_windows': len(self.window_results),
            'mean_test_acc': float(np.mean(test_accs)),
            'std_test_acc': float(np.std(test_accs)),
            'min_test_acc': float(np.min(test_accs)),
            'max_test_acc': float(np.max(test_accs)),
            'mean_direction_acc': float(np.mean(dir_accs)),
            'window_results': self.window_results,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.config.save_dir / "walk_forward_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\n=== Summary ===")
        logger.info(f"Mean accuracy: {summary['mean_test_acc']:.4f} +/- {summary['std_test_acc']:.4f}")
        logger.info(f"Mean direction acc: {summary['mean_direction_acc']:.4f}")
        logger.info(f"Elapsed: {elapsed:.1f}s")

        return summary


def run_model_walk_forward(data: np.ndarray, config: ModelWalkForwardConfig = None, device: str = None) -> Dict[str, Any]:
    """Convenience function for model walk-forward validation."""
    validator = ModelWalkForwardValidator(config, device)
    return validator.run(data)
