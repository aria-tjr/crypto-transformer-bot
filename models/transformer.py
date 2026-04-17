import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BTCTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        # Input Projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=d_model*4, 
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Heads
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3) # [Short, Neutral, Long]
        )
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Transformer Pass
        x = self.transformer_encoder(x)
        
        # Global Average Pooling (Focus on the whole sequence context)
        x = x.mean(dim=1)
        
        # Return dict to match bot interface
        logits = self.direction_head(x)
        return {
            'direction_probs': torch.softmax(logits, dim=-1)
        }
