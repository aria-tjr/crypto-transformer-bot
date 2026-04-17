"""
Temporal Convolutional Network (TCN) for time series forecasting.
Efficient alternative to RNNs with parallel computation and stable gradients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TCNConfig:
    """Configuration for TCN model."""
    input_dim: int = 64
    num_channels: list = None  # Channel sizes for each layer
    kernel_size: int = 3
    dropout: float = 0.2
    output_dim: int = 3  # [down, neutral, up]

    def __post_init__(self):
        if self.num_channels is None:
            # Default: 4 layers with increasing then decreasing channels
            self.num_channels = [128, 256, 256, 128]


class CausalConv1d(nn.Module):
    """Causal convolution for time series (no future leakage)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal padding."""
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolutions and residual connection.

    Architecture:
    1. Dilated causal conv -> BatchNorm -> ReLU -> Dropout
    2. Dilated causal conv -> BatchNorm -> ReLU -> Dropout
    3. Residual connection (with optional 1x1 conv for channel matching)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for time series prediction.

    Architecture:
    1. Input projection
    2. Stack of temporal blocks with exponentially increasing dilation
    3. Global pooling
    4. Output heads (direction, magnitude, confidence)

    Benefits:
    - Parallelizable (unlike RNNs)
    - Stable gradients (no vanishing/exploding)
    - Flexible receptive field via dilation
    - Memory efficient
    """

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config

        # Input projection: (batch, seq, features) -> (batch, channels, seq)
        self.input_proj = nn.Linear(config.input_dim, config.num_channels[0])

        # Temporal blocks with exponentially increasing dilation
        layers = []
        num_levels = len(config.num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            in_channels = config.num_channels[0] if i == 0 else config.num_channels[i-1]
            out_channels = config.num_channels[i]

            layers.append(TemporalBlock(
                in_channels,
                out_channels,
                config.kernel_size,
                dilation,
                config.dropout
            ))

        self.network = nn.Sequential(*layers)

        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field()

        # Output heads
        final_channels = config.num_channels[-1]

        self.direction_head = nn.Sequential(
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(final_channels // 2, 3)  # [down, neutral, up]
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(final_channels // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(final_channels // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _calculate_receptive_field(self) -> int:
        """Calculate the receptive field of the network."""
        num_levels = len(self.config.num_channels)
        kernel_size = self.config.kernel_size

        # Each level doubles dilation, contributing (kernel_size - 1) * dilation
        receptive_field = 1
        for i in range(num_levels):
            dilation = 2 ** i
            # Each temporal block has 2 conv layers
            receptive_field += 2 * (kernel_size - 1) * dilation

        return receptive_field

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_features: Whether to return intermediate features

        Returns:
            Dict with 'direction', 'magnitude', 'confidence'
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (batch, seq, channels)

        # Transpose for conv: (batch, channels, seq)
        x = x.transpose(1, 2)

        # TCN layers
        features = self.network(x)

        # Global average pooling over time
        pooled = features.mean(dim=-1)  # (batch, channels)

        # Output heads
        direction_logits = self.direction_head(pooled)
        magnitude = self.magnitude_head(pooled)
        confidence = self.confidence_head(pooled)

        result = {
            'direction': direction_logits,
            'direction_probs': F.softmax(direction_logits, dim=-1),
            'magnitude': magnitude.squeeze(-1),
            'confidence': confidence.squeeze(-1)
        }

        if return_features:
            result['features'] = features.transpose(1, 2)  # (batch, seq, channels)

        return result

    def predict(self, x: torch.Tensor) -> Tuple[int, float, float]:
        """
        Make a prediction.

        Args:
            x: Input tensor (1, seq_len, input_dim)

        Returns:
            Tuple of (direction, magnitude, confidence)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            direction_probs = output['direction_probs'][0]
            direction = torch.argmax(direction_probs).item() - 1  # Map [0,1,2] to [-1,0,1]

            magnitude = output['magnitude'][0].item()
            confidence = output['confidence'][0].item()

        return direction, magnitude, confidence


class TCNAttention(TCN):
    """
    TCN with attention mechanism for weighted temporal aggregation.

    Instead of simple global average pooling, uses attention
    to learn which time steps are most important.
    """

    def __init__(self, config: TCNConfig):
        super().__init__(config)

        final_channels = config.num_channels[-1]

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(final_channels, final_channels // 4),
            nn.Tanh(),
            nn.Linear(final_channels // 4, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> dict:
        """Forward pass with attention."""
        batch_size, seq_len, _ = x.shape

        # Project and transpose
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        # TCN layers
        features = self.network(x)  # (batch, channels, seq)

        # Transpose back for attention
        features_t = features.transpose(1, 2)  # (batch, seq, channels)

        # Compute attention weights
        attn_scores = self.attention(features_t).squeeze(-1)  # (batch, seq)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq)

        # Weighted sum
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq)
            features_t  # (batch, seq, channels)
        ).squeeze(1)  # (batch, channels)

        # Output heads
        direction_logits = self.direction_head(pooled)
        magnitude = self.magnitude_head(pooled)
        confidence = self.confidence_head(pooled)

        result = {
            'direction': direction_logits,
            'direction_probs': F.softmax(direction_logits, dim=-1),
            'magnitude': magnitude.squeeze(-1),
            'confidence': confidence.squeeze(-1)
        }

        if return_attention:
            result['attention'] = attn_weights

        return result


def create_tcn(config: TCNConfig = None) -> TCN:
    """Factory function to create TCN model."""
    if config is None:
        config = TCNConfig()
    return TCN(config)


def create_tcn_attention(config: TCNConfig = None) -> TCNAttention:
    """Factory function to create TCN with attention."""
    if config is None:
        config = TCNConfig()
    return TCNAttention(config)
