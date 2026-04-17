import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustLoss(nn.Module):
    """
    Loss function designed to penalize errors in high-volatility conditions
    more severely than in calm conditions.
    """
    def __init__(self, volatility_penalty=2.0, smoothing=0.1):
        super().__init__()
        self.vol_penalty = volatility_penalty
        self.smoothing = smoothing

    def forward(self, predictions, targets, volatility=None):
        """
        predictions: [batch, classes] (Logits)
        targets: [batch] (Class indices)
        volatility: [batch] (Normalized volatility metric, optional)
        """
        # Label Smoothing
        n_classes = predictions.size(1)
        one_hot = torch.zeros_like(predictions).scatter(1, targets.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_classes - 1)
        
        log_probs = F.log_softmax(predictions, dim=1)
        loss = -(one_hot * log_probs).sum(dim=1)
        
        # Volatility Weighting
        if volatility is not None:
            # If volatility is high, the model MUST be careful.
            # We weight the loss higher for high-volatility samples.
            # volatility should be normalized (e.g., 0 to 1 or z-score)
            weights = 1.0 + (volatility * self.vol_penalty)
            loss = loss * weights
            
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights well-classified examples and focuses on hard examples.
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()
