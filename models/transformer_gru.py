"""
Hybrid Transformer+GRU model for time series forecasting.
Combines attention mechanisms with recurrent processing.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TransformerGRUConfig:
    """Configuration for Transformer+GRU model."""
    input_dim: int = 64
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    gru_hidden: int = 128
    gru_layers: int = 2
    bidirectional: bool = True
    output_dim: int = 3  # [direction, magnitude, confidence]
    max_seq_len: int = 500


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))

        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with pre-norm architecture."""
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed, mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


class TransformerGRU(nn.Module):
    """
    Hybrid Transformer+GRU model for time series prediction.

    Architecture:
    1. Input projection
    2. Positional encoding
    3. Transformer encoder (long-range dependencies)
    4. Bidirectional GRU (sequential patterns)
    5. Output projection
    """

    def __init__(self, config: TransformerGRUConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout
        )

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                causal=True
            )
            for _ in range(config.n_encoder_layers)
        ])

        # GRU
        gru_input_size = config.d_model
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        # Output dimension after GRU
        gru_output_size = config.gru_hidden * (2 if config.bidirectional else 1)

        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gru_output_size // 2, 3)  # [down, neutral, up]
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gru_output_size // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gru_output_size // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional padding mask
            return_attention: Whether to return attention weights

        Returns:
            Dict with 'direction', 'magnitude', 'confidence', optionally 'attention'
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn = layer(x, mask)
            if return_attention:
                attention_weights.append(attn)

        # GRU
        gru_out, hidden = self.gru(x)

        # Use last hidden state for prediction
        if self.config.bidirectional:
            # Concatenate forward and backward final hidden states
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            final_hidden = hidden[-1]

        # Output heads
        direction_logits = self.direction_head(final_hidden)
        magnitude = self.magnitude_head(final_hidden)
        confidence = self.confidence_head(final_hidden)

        result = {
            'direction': direction_logits,
            'direction_probs': F.softmax(direction_logits, dim=-1),
            'magnitude': magnitude.squeeze(-1),
            'confidence': confidence.squeeze(-1)
        }

        if return_attention:
            result['attention'] = attention_weights

        return result

    def predict(self, x: torch.Tensor) -> Tuple[int, float, float]:
        """
        Make a prediction.

        Args:
            x: Input tensor (1, seq_len, input_dim)

        Returns:
            Tuple of (direction, magnitude, confidence)
            direction: -1, 0, or 1
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            direction_probs = output['direction_probs'][0]
            direction = torch.argmax(direction_probs).item() - 1  # Map [0,1,2] to [-1,0,1]

            magnitude = output['magnitude'][0].item()
            confidence = output['confidence'][0].item()

        return direction, magnitude, confidence

    def get_feature_importance(
        self,
        x: torch.Tensor,
        feature_names: list = None
    ) -> dict:
        """
        Get feature importance via attention weights.

        Returns dict mapping feature indices/names to importance scores.
        """
        output = self.forward(x, return_attention=True)

        # Average attention across heads and layers
        all_attn = torch.stack(output['attention'], dim=0)  # (layers, batch, heads, seq, seq)
        avg_attn = all_attn.mean(dim=(0, 1, 2))  # (seq, seq)

        # Sum attention received by each position
        importance = avg_attn.sum(dim=0).cpu().numpy()

        if feature_names:
            return {name: float(imp) for name, imp in zip(feature_names, importance)}

        return {i: float(imp) for i, imp in enumerate(importance)}


class TransformerGRUWithMemory(TransformerGRU):
    """
    Extended model with explicit memory for longer-term patterns.

    Adds a memory bank that persists across sequences.
    """

    def __init__(self, config: TransformerGRUConfig, memory_size: int = 64):
        super().__init__(config)

        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(1, memory_size, config.d_model) * 0.02)

        # Cross-attention to memory
        self.memory_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            causal=False
        )
        self.memory_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> dict:
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1)

        # Transformer layers with memory attention
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn = layer(x, mask)
            if return_attention:
                attention_weights.append(attn)

            # Cross-attend to memory
            normed = self.memory_norm(x)
            mem_out, _ = self.memory_attn(
                torch.cat([normed, memory], dim=1)
            )
            x = x + mem_out[:, :seq_len]

        # GRU
        gru_out, hidden = self.gru(x)

        if self.config.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            final_hidden = hidden[-1]

        # Output heads
        direction_logits = self.direction_head(final_hidden)
        magnitude = self.magnitude_head(final_hidden)
        confidence = self.confidence_head(final_hidden)

        result = {
            'direction': direction_logits,
            'direction_probs': F.softmax(direction_logits, dim=-1),
            'magnitude': magnitude.squeeze(-1),
            'confidence': confidence.squeeze(-1)
        }

        if return_attention:
            result['attention'] = attention_weights

        return result


def create_model(config: TransformerGRUConfig = None) -> TransformerGRU:
    """Factory function to create model."""
    if config is None:
        config = TransformerGRUConfig()
    return TransformerGRU(config)
