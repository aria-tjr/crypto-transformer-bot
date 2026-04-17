"""
Contrastive pre-training for time series representations.
Implements TF-C style time-frequency consistency learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""
    input_dim: int = 64
    encoder_dim: int = 128
    projection_dim: int = 64
    temperature: float = 0.07
    jitter_ratio: float = 0.1
    scaling_ratio: float = 0.1
    permutation_max_segments: int = 5
    mask_ratio: float = 0.3


class TimeSeriesEncoder(nn.Module):
    """
    Encoder for time series data.

    Uses dilated causal convolutions for temporal features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Dilated causal convolutions
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim, hidden_dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, output_dim) encoded representation
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq, hidden)

        # Transpose for conv
        x = x.transpose(1, 2)  # (batch, hidden, seq)

        # Dilated convolutions with residual
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            x = x[:, :, :residual.size(2)]  # Trim to match
            x = x + residual

        # Global average pooling
        x = x.mean(dim=-1)  # (batch, hidden)

        # Output projection
        x = self.output_proj(x)

        return x


class FrequencyEncoder(nn.Module):
    """
    Encoder for frequency domain representation.

    Applies FFT and encodes spectral features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Real + imaginary
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode frequency representation.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, output_dim) encoded representation
        """
        # Apply FFT along time dimension
        x_fft = torch.fft.rfft(x, dim=1)

        # Take magnitude and phase
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # Concatenate and flatten frequency bins
        freq_features = torch.cat([magnitude, phase], dim=-1)
        freq_features = freq_features.mean(dim=1)  # Average over frequency bins

        return self.encoder(freq_features)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSeriesAugmentation:
    """Augmentation strategies for time series data."""

    def __init__(self, config: ContrastiveConfig):
        self.config = config

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.config.jitter_ratio
        return x + noise * x.std()

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random scaling."""
        scale = 1 + (torch.rand(x.size(0), 1, 1, device=x.device) - 0.5) * 2 * self.config.scaling_ratio
        return x * scale

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        """Random segment permutation."""
        batch_size, seq_len, feat_dim = x.shape

        num_segments = np.random.randint(2, self.config.permutation_max_segments + 1)
        segment_size = seq_len // num_segments

        result = x.clone()
        for b in range(batch_size):
            segments = list(range(num_segments))
            np.random.shuffle(segments)

            new_x = []
            for seg_idx in segments:
                start = seg_idx * segment_size
                end = start + segment_size
                new_x.append(x[b, start:end])

            # Handle remaining
            if num_segments * segment_size < seq_len:
                new_x.append(x[b, num_segments * segment_size:])

            result[b] = torch.cat(new_x, dim=0)[:seq_len]

        return result

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """Random masking."""
        mask = torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.config.mask_ratio
        return x * mask

    def augment(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to create two views.

        Returns:
            Tuple of two augmented views
        """
        # View 1: jitter + scale
        view1 = self.scale(self.jitter(x))

        # View 2: permute + mask
        view2 = self.mask(self.permute(x))

        return view1, view2


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z1: First view embeddings (batch, dim)
            z2: Second view embeddings (batch, dim)

        Returns:
            Contrastive loss value
        """
        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.t()) / self.temperature

        # Labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=z1.device)

        # Cross-entropy loss (symmetric)
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.t(), labels)

        return (loss_1 + loss_2) / 2


class TFCModel(nn.Module):
    """
    Time-Frequency Consistency (TF-C) contrastive model.

    Learns representations by enforcing consistency between
    time domain and frequency domain encodings.
    """

    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.config = config

        # Time domain encoder
        self.time_encoder = TimeSeriesEncoder(
            config.input_dim,
            hidden_dim=config.encoder_dim // 2,
            output_dim=config.encoder_dim
        )

        # Frequency domain encoder
        self.freq_encoder = FrequencyEncoder(
            config.input_dim,
            hidden_dim=config.encoder_dim // 2,
            output_dim=config.encoder_dim
        )

        # Projection heads
        self.time_projector = ProjectionHead(
            config.encoder_dim,
            config.encoder_dim,
            config.projection_dim
        )
        self.freq_projector = ProjectionHead(
            config.encoder_dim,
            config.encoder_dim,
            config.projection_dim
        )

        # Augmentation
        self.augmentation = TimeSeriesAugmentation(config)

        # Loss
        self.contrastive_loss = ContrastiveLoss(config.temperature)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input in both time and frequency domains.

        Returns:
            Tuple of (time_encoding, freq_encoding)
        """
        time_enc = self.time_encoder(x)
        freq_enc = self.freq_encoder(x)
        return time_enc, freq_enc

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True
    ) -> dict:
        """
        Forward pass for training.

        Args:
            x: Input time series (batch, seq_len, input_dim)
            return_loss: Whether to compute and return loss

        Returns:
            Dict with encodings and optionally loss
        """
        # Augment to create views
        view1, view2 = self.augmentation.augment(x)

        # Encode both views
        time_enc1, freq_enc1 = self.encode(view1)
        time_enc2, freq_enc2 = self.encode(view2)

        # Project for contrastive loss
        time_proj1 = self.time_projector(time_enc1)
        time_proj2 = self.time_projector(time_enc2)
        freq_proj1 = self.freq_projector(freq_enc1)
        freq_proj2 = self.freq_projector(freq_enc2)

        result = {
            'time_encoding': time_enc1,
            'freq_encoding': freq_enc1,
            'time_projection': time_proj1,
            'freq_projection': freq_proj1
        }

        if return_loss:
            # Time-time consistency
            loss_tt = self.contrastive_loss(time_proj1, time_proj2)

            # Frequency-frequency consistency
            loss_ff = self.contrastive_loss(freq_proj1, freq_proj2)

            # Time-frequency consistency (cross-modal)
            loss_tf = self.contrastive_loss(time_proj1, freq_proj1)
            loss_tf += self.contrastive_loss(time_proj2, freq_proj2)
            loss_tf /= 2

            # Total loss
            total_loss = loss_tt + loss_ff + loss_tf

            result['loss'] = total_loss
            result['loss_tt'] = loss_tt
            result['loss_ff'] = loss_ff
            result['loss_tf'] = loss_tf

        return result

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get learned representation for downstream tasks.

        Concatenates time and frequency encodings.
        """
        time_enc, freq_enc = self.encode(x)
        return torch.cat([time_enc, freq_enc], dim=-1)


class ContrastivePretrainer:
    """Training loop for contrastive pre-training."""

    def __init__(
        self,
        model: TFCModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=learning_rate / 100
        )

    def train_epoch(self, dataloader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tt = 0
        total_ff = 0
        total_tf = 0
        num_batches = 0

        for batch in dataloader:
            x = batch.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(x, return_loss=True)
            loss = output['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_tt += output['loss_tt'].item()
            total_ff += output['loss_ff'].item()
            total_tf += output['loss_tf'].item()
            num_batches += 1

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'loss_tt': total_tt / num_batches,
            'loss_ff': total_ff / num_batches,
            'loss_tf': total_tf / num_batches,
            'lr': self.scheduler.get_last_lr()[0]
        }

    def save_encoder(self, path: str):
        """Save encoder weights for downstream use."""
        torch.save({
            'time_encoder': self.model.time_encoder.state_dict(),
            'freq_encoder': self.model.freq_encoder.state_dict()
        }, path)

    def load_encoder(self, path: str):
        """Load pre-trained encoder weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.time_encoder.load_state_dict(checkpoint['time_encoder'])
        self.model.freq_encoder.load_state_dict(checkpoint['freq_encoder'])


def create_model(config: ContrastiveConfig = None) -> TFCModel:
    """Factory function to create contrastive model."""
    if config is None:
        config = ContrastiveConfig()
    return TFCModel(config)
