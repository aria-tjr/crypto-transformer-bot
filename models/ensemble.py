"""
Model ensemble for combining predictions from multiple models.
Implements confidence-weighted averaging and disagreement detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EnsembleConfig:
    """Configuration for ensemble."""
    num_models: int = 3
    output_dim: int = 3
    temperature: float = 1.0
    disagreement_threshold: float = 0.3
    use_learned_weights: bool = True


class ModelWrapper:
    """Wrapper for individual models in ensemble."""

    def __init__(
        self,
        model: nn.Module,
        name: str,
        weight: float = 1.0
    ):
        self.model = model
        self.name = name
        self.weight = weight
        self.performance_history: List[float] = []

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get prediction from model."""
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def update_weight(self, performance: float, decay: float = 0.95):
        """Update weight based on recent performance."""
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        # Exponential moving average of performance
        if self.performance_history:
            weights = [decay ** i for i in range(len(self.performance_history) - 1, -1, -1)]
            self.weight = sum(p * w for p, w in zip(self.performance_history, weights)) / sum(weights)
            self.weight = max(0.1, min(1.0, self.weight))


class WeightedEnsemble(nn.Module):
    """
    Ensemble that combines multiple model predictions.

    Features:
    - Confidence-weighted averaging
    - Disagreement detection
    - Dynamic weight adjustment
    """

    def __init__(self, config: EnsembleConfig):
        super().__init__()
        self.config = config

        self.models: List[ModelWrapper] = []

        # Learned combination weights
        if config.use_learned_weights:
            self.weight_net = nn.Sequential(
                nn.Linear(config.num_models * config.output_dim, 64),
                nn.ReLU(),
                nn.Linear(64, config.num_models),
                nn.Softmax(dim=-1)
            )
        else:
            self.weight_net = None

        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)

    def add_model(self, model: nn.Module, name: str, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(ModelWrapper(model, name, weight))

    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get ensemble prediction.

        Args:
            x: Input tensor
            return_individual: Whether to return individual model predictions

        Returns:
            Dict with ensemble prediction and metadata
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # Collect predictions from all models
        predictions = []
        confidences = []

        for wrapper in self.models:
            pred = wrapper.predict(x)

            # Handle different model output formats
            if 'direction_probs' in pred:
                probs = pred['direction_probs']
            elif 'direction' in pred:
                probs = F.softmax(pred['direction'] / self.temperature, dim=-1)
            else:
                probs = F.softmax(pred['logits'] / self.temperature, dim=-1)

            predictions.append(probs)

            # Get confidence
            if 'confidence' in pred:
                conf = pred['confidence']
            else:
                conf = probs.max(dim=-1)[0]

            confidences.append(conf * wrapper.weight)

        predictions = torch.stack(predictions, dim=1)  # (batch, num_models, output_dim)
        confidences = torch.stack(confidences, dim=1)  # (batch, num_models)

        # Compute ensemble weights
        if self.weight_net is not None:
            flat_preds = predictions.view(predictions.size(0), -1)
            weights = self.weight_net(flat_preds)  # (batch, num_models)
        else:
            # Normalize confidences as weights
            weights = F.softmax(confidences, dim=-1)

        # Weighted average of predictions
        weights_expanded = weights.unsqueeze(-1)  # (batch, num_models, 1)
        ensemble_probs = (predictions * weights_expanded).sum(dim=1)  # (batch, output_dim)

        # Calculate disagreement
        disagreement = self._calculate_disagreement(predictions)

        # Ensemble confidence
        base_confidence = ensemble_probs.max(dim=-1)[0]
        ensemble_confidence = base_confidence * (1 - disagreement)

        result = {
            'probs': ensemble_probs,
            'direction': torch.argmax(ensemble_probs, dim=-1) - 1,  # Map to [-1, 0, 1]
            'confidence': ensemble_confidence,
            'disagreement': disagreement,
            'weights': weights
        }

        if return_individual:
            result['individual_predictions'] = predictions
            result['individual_confidences'] = confidences

        return result

    def _calculate_disagreement(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate disagreement between models.

        High disagreement indicates uncertainty.
        """
        # Variance across model predictions
        variance = predictions.var(dim=1).mean(dim=-1)  # (batch,)

        # Normalize to [0, 1]
        disagreement = torch.tanh(variance * 10)

        return disagreement

    def get_prediction_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[int, float, float, bool]:
        """
        Get prediction with uncertainty estimate.

        Returns:
            Tuple of (direction, confidence, uncertainty, is_reliable)
        """
        output = self.forward(x)

        direction = output['direction'][0].item()
        confidence = output['confidence'][0].item()
        disagreement = output['disagreement'][0].item()

        is_reliable = disagreement < self.config.disagreement_threshold

        return direction, confidence, disagreement, is_reliable

    def update_model_weights(self, model_performances: Dict[str, float]):
        """Update individual model weights based on performance."""
        for wrapper in self.models:
            if wrapper.name in model_performances:
                wrapper.update_weight(model_performances[wrapper.name])


class RegimeAwareEnsemble(WeightedEnsemble):
    """
    Ensemble that adjusts weights based on market regime.

    Different models may excel in different regimes.
    """

    REGIMES = ['bull', 'bear', 'sideways', 'volatile']

    def __init__(self, config: EnsembleConfig):
        super().__init__(config)

        # Per-regime model weights
        self.regime_weights = nn.ParameterDict({
            regime: nn.Parameter(torch.ones(config.num_models) / config.num_models)
            for regime in self.REGIMES
        })

        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.REGIMES)),
            nn.Softmax(dim=-1)
        )

    def detect_regime(self, market_features: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """
        Detect current market regime.

        Args:
            market_features: Recent market features

        Returns:
            Tuple of (regime_name, regime_probabilities)
        """
        regime_probs = self.regime_classifier(market_features)
        regime_idx = torch.argmax(regime_probs, dim=-1)
        regime_name = self.REGIMES[regime_idx.item()]

        return regime_name, regime_probs

    def forward(
        self,
        x: torch.Tensor,
        market_features: Optional[torch.Tensor] = None,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get regime-aware ensemble prediction.

        Args:
            x: Model input
            market_features: Features for regime detection
            return_individual: Return individual predictions

        Returns:
            Ensemble prediction with regime info
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # Detect regime
        if market_features is not None:
            regime_name, regime_probs = self.detect_regime(market_features)
            regime_weights = F.softmax(self.regime_weights[regime_name], dim=-1)
        else:
            regime_name = 'unknown'
            regime_probs = None
            regime_weights = torch.ones(len(self.models)) / len(self.models)

        # Collect predictions
        predictions = []
        confidences = []

        for i, wrapper in enumerate(self.models):
            pred = wrapper.predict(x)

            if 'direction_probs' in pred:
                probs = pred['direction_probs']
            elif 'direction' in pred:
                probs = F.softmax(pred['direction'] / self.temperature, dim=-1)
            else:
                probs = F.softmax(pred['logits'] / self.temperature, dim=-1)

            predictions.append(probs)

            if 'confidence' in pred:
                conf = pred['confidence']
            else:
                conf = probs.max(dim=-1)[0]

            # Apply regime-specific weight
            confidences.append(conf * regime_weights[i])

        predictions = torch.stack(predictions, dim=1)
        confidences = torch.stack(confidences, dim=1)

        # Weighted average
        weights = F.softmax(confidences, dim=-1)
        weights_expanded = weights.unsqueeze(-1)
        ensemble_probs = (predictions * weights_expanded).sum(dim=1)

        disagreement = self._calculate_disagreement(predictions)
        ensemble_confidence = ensemble_probs.max(dim=-1)[0] * (1 - disagreement)

        result = {
            'probs': ensemble_probs,
            'direction': torch.argmax(ensemble_probs, dim=-1) - 1,
            'confidence': ensemble_confidence,
            'disagreement': disagreement,
            'weights': weights,
            'regime': regime_name
        }

        if regime_probs is not None:
            result['regime_probs'] = regime_probs

        if return_individual:
            result['individual_predictions'] = predictions

        return result


class AdaptiveEnsemble(WeightedEnsemble):
    """
    Ensemble that adapts weights online based on recent performance.
    """

    def __init__(
        self,
        config: EnsembleConfig,
        adaptation_rate: float = 0.1,
        performance_window: int = 50
    ):
        super().__init__(config)
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window

        # Track recent performance
        self.recent_predictions: List[Dict] = []
        self.recent_targets: List[int] = []

    def record_prediction(self, prediction: Dict, target: int):
        """Record a prediction and its target for performance tracking."""
        self.recent_predictions.append(prediction)
        self.recent_targets.append(target)

        if len(self.recent_predictions) > self.performance_window:
            self.recent_predictions.pop(0)
            self.recent_targets.pop(0)

        # Update weights periodically
        if len(self.recent_predictions) >= 10 and len(self.recent_predictions) % 10 == 0:
            self._update_weights()

    def _update_weights(self):
        """Update model weights based on recent performance."""
        if not self.recent_predictions or 'individual_predictions' not in self.recent_predictions[0]:
            return

        num_models = len(self.models)
        accuracies = [0.0] * num_models

        for pred, target in zip(self.recent_predictions, self.recent_targets):
            individual = pred['individual_predictions']

            for i in range(num_models):
                model_pred = torch.argmax(individual[:, i], dim=-1) - 1
                if model_pred.item() == target:
                    accuracies[i] += 1

        # Normalize
        accuracies = [acc / len(self.recent_predictions) for acc in accuracies]

        # Update weights
        for i, wrapper in enumerate(self.models):
            wrapper.weight = (1 - self.adaptation_rate) * wrapper.weight + self.adaptation_rate * accuracies[i]
            wrapper.weight = max(0.1, wrapper.weight)


def create_ensemble(
    models: List[nn.Module],
    model_names: List[str] = None,
    config: EnsembleConfig = None
) -> WeightedEnsemble:
    """
    Create ensemble from list of models.

    Args:
        models: List of trained models
        model_names: Optional names for models
        config: Ensemble configuration

    Returns:
        Configured ensemble
    """
    if config is None:
        config = EnsembleConfig(num_models=len(models))

    ensemble = WeightedEnsemble(config)

    for i, model in enumerate(models):
        name = model_names[i] if model_names else f"model_{i}"
        ensemble.add_model(model, name)

    return ensemble
