"""
Graph Neural Network for cross-asset relationship modeling.
Captures correlations between BTC, ETH, SOL and other market factors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class GNNConfig:
    """Configuration for GNN model."""
    node_input_dim: int = 32
    hidden_channels: int = 64
    num_layers: int = 3
    num_assets: int = 3  # BTC, ETH, SOL
    num_auxiliary_nodes: int = 2  # Sentiment, macro
    heads: int = 4
    dropout: float = 0.1
    temporal_window: int = 20
    output_dim: int = 3  # Per-asset prediction


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Learns attention weights between nodes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat

        # Linear transformations for each head
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention mechanism
        self.a_src = nn.Parameter(torch.zeros(1, heads, out_channels))
        self.a_dst = nn.Parameter(torch.zeros(1, heads, out_channels))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connections (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)

        Returns:
            Updated node features
        """
        num_nodes = x.size(0)

        # Linear transformation
        x = self.W(x).view(num_nodes, self.heads, self.out_channels)

        # Calculate attention coefficients
        src_idx, dst_idx = edge_index[0], edge_index[1]

        alpha_src = (x[src_idx] * self.a_src).sum(dim=-1)
        alpha_dst = (x[dst_idx] * self.a_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst

        alpha = self.leaky_relu(alpha)

        # Apply edge weights if provided
        if edge_weight is not None:
            alpha = alpha * edge_weight.unsqueeze(-1)

        # Softmax normalization per node
        alpha = self._softmax_per_node(alpha, dst_idx, num_nodes)
        alpha = self.dropout(alpha)

        # Aggregate messages
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_channels),
                         x[src_idx] * alpha.unsqueeze(-1))

        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        return out

    def _softmax_per_node(
        self,
        alpha: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Compute softmax per destination node."""
        alpha_max = torch.zeros(num_nodes, self.heads, device=alpha.device)
        alpha_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(-1, self.heads), alpha, reduce='amax')
        alpha = alpha - alpha_max[dst_idx]
        alpha = torch.exp(alpha)

        alpha_sum = torch.zeros(num_nodes, self.heads, device=alpha.device)
        alpha_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.heads), alpha)
        alpha = alpha / (alpha_sum[dst_idx] + 1e-8)

        return alpha


class TemporalConvBlock(nn.Module):
    """Temporal convolution for processing time series on each node."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        """
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # Causal padding
        out = self.norm(out)
        out = self.activation(out)
        return out


class CrossAssetGNN(nn.Module):
    """
    Graph Neural Network for cross-asset prediction.

    Architecture:
    1. Temporal convolution on each asset's features
    2. Graph attention for cross-asset relationships
    3. Output prediction per asset
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.num_assets + config.num_auxiliary_nodes

        # Node feature embedding
        self.node_embedding = nn.Linear(config.node_input_dim, config.hidden_channels)

        # Temporal processing
        self.temporal_conv = nn.Sequential(
            TemporalConvBlock(config.hidden_channels, config.hidden_channels, 3, 1),
            TemporalConvBlock(config.hidden_channels, config.hidden_channels, 3, 2),
            TemporalConvBlock(config.hidden_channels, config.hidden_channels, 3, 4)
        )

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        in_channels = config.hidden_channels

        for i in range(config.num_layers):
            concat = i < config.num_layers - 1
            out_channels = config.hidden_channels if concat else config.hidden_channels // config.heads

            self.gat_layers.append(
                GraphAttentionLayer(
                    in_channels,
                    out_channels,
                    config.heads,
                    config.dropout,
                    concat
                )
            )

            if concat:
                in_channels = out_channels * config.heads
            else:
                in_channels = out_channels * config.heads

        # Output heads for each asset
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_channels, config.hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_channels // 2, config.output_dim)
            )
            for _ in range(config.num_assets)
        ])

        # Learnable edge weights (correlation strengths)
        self.edge_weight_net = nn.Sequential(
            nn.Linear(config.hidden_channels * 2, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, 1),
            nn.Sigmoid()
        )

        # Build default fully connected graph
        self._build_default_edges()

    def _build_default_edges(self):
        """Create fully connected edge index."""
        edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    edges.append([i, j])

        self.register_buffer(
            'default_edge_index',
            torch.tensor(edges, dtype=torch.long).t()
        )

    def compute_edge_weights(self, node_features: torch.Tensor) -> torch.Tensor:
        """Compute dynamic edge weights based on node features."""
        src_idx, dst_idx = self.default_edge_index

        src_features = node_features[src_idx]
        dst_features = node_features[dst_idx]

        edge_features = torch.cat([src_features, dst_features], dim=-1)
        weights = self.edge_weight_net(edge_features).squeeze(-1)

        return weights

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        return_edge_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features (batch, num_nodes, temporal_window, node_input_dim)
            edge_index: Optional custom edge connections
            return_edge_weights: Whether to return computed edge weights

        Returns:
            Dict with predictions per asset
        """
        batch_size, num_nodes, seq_len, feat_dim = x.shape

        # Embed node features
        x = self.node_embedding(x)  # (batch, nodes, time, hidden)

        # Temporal convolution per node
        x = x.permute(0, 1, 3, 2)  # (batch, nodes, hidden, time)
        x = x.reshape(batch_size * num_nodes, self.config.hidden_channels, seq_len)
        x = self.temporal_conv(x)
        x = x.reshape(batch_size, num_nodes, self.config.hidden_channels, -1)
        x = x[:, :, :, -1]  # Take last timestep (batch, nodes, hidden)

        # Use default edges if not provided
        if edge_index is None:
            edge_index = self.default_edge_index

        # Process each batch item through GNN
        outputs = []
        all_edge_weights = []

        for b in range(batch_size):
            node_feat = x[b]  # (nodes, hidden)

            # Compute dynamic edge weights
            edge_weights = self.compute_edge_weights(node_feat)
            all_edge_weights.append(edge_weights)

            # Graph attention layers
            for gat_layer in self.gat_layers:
                node_feat = gat_layer(node_feat, edge_index, edge_weights)
                node_feat = F.elu(node_feat)
                node_feat = F.dropout(node_feat, p=self.config.dropout, training=self.training)

            outputs.append(node_feat)

        x = torch.stack(outputs, dim=0)  # (batch, nodes, hidden)

        # Generate predictions for each asset
        predictions = {}
        for i in range(self.config.num_assets):
            asset_feat = x[:, i]  # (batch, hidden)
            pred = self.output_heads[i](asset_feat)  # (batch, output_dim)
            predictions[f'asset_{i}'] = pred

        # Direction probabilities
        for i in range(self.config.num_assets):
            predictions[f'asset_{i}_probs'] = F.softmax(predictions[f'asset_{i}'], dim=-1)

        if return_edge_weights:
            predictions['edge_weights'] = torch.stack(all_edge_weights, dim=0)

        return predictions

    def get_correlation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get learned correlation matrix between assets.

        Returns matrix showing relationship strengths.
        """
        output = self.forward(x, return_edge_weights=True)
        edge_weights = output['edge_weights'].mean(dim=0)  # Average across batch

        # Build correlation matrix
        corr_matrix = torch.zeros(self.num_nodes, self.num_nodes, device=x.device)
        src_idx, dst_idx = self.default_edge_index
        corr_matrix[src_idx, dst_idx] = edge_weights

        return corr_matrix


class DynamicGraphBuilder:
    """
    Builds dynamic graphs based on market correlations.

    Updates edge connections based on rolling correlations.
    """

    def __init__(
        self,
        num_assets: int,
        correlation_window: int = 50,
        edge_threshold: float = 0.3
    ):
        self.num_assets = num_assets
        self.correlation_window = correlation_window
        self.edge_threshold = edge_threshold
        self.price_history: Dict[int, List[float]] = {i: [] for i in range(num_assets)}

    def update(self, asset_prices: Dict[int, float]):
        """Update price history with new prices."""
        for asset_id, price in asset_prices.items():
            self.price_history[asset_id].append(price)
            if len(self.price_history[asset_id]) > self.correlation_window:
                self.price_history[asset_id].pop(0)

    def build_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge index based on correlations.

        Returns:
            Tuple of (edge_index, edge_weights)
        """
        if len(self.price_history[0]) < 10:
            # Not enough data, return fully connected
            edges = []
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    if i != j:
                        edges.append([i, j])
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.ones(edge_index.size(1))
            return edge_index, edge_weights

        # Calculate returns
        returns = {}
        for asset_id, prices in self.price_history.items():
            if len(prices) > 1:
                prices_arr = np.array(prices)
                returns[asset_id] = np.diff(prices_arr) / prices_arr[:-1]

        # Build correlation-based edges
        edges = []
        weights = []

        for i in range(self.num_assets):
            for j in range(self.num_assets):
                if i != j and i in returns and j in returns:
                    corr = np.corrcoef(returns[i], returns[j])[0, 1]

                    if not np.isnan(corr) and abs(corr) > self.edge_threshold:
                        edges.append([i, j])
                        weights.append(abs(corr))

        if not edges:
            # Fallback to fully connected
            for i in range(self.num_assets):
                for j in range(self.num_assets):
                    if i != j:
                        edges.append([i, j])
                        weights.append(0.5)

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weights = torch.tensor(weights, dtype=torch.float32)

        return edge_index, edge_weights


def create_model(config: GNNConfig = None) -> CrossAssetGNN:
    """Factory function to create GNN model."""
    if config is None:
        config = GNNConfig()
    return CrossAssetGNN(config)
