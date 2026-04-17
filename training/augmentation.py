import numpy as np
import torch

class DataAugmenter:
    """
    Applies random transformations to financial time series data 
    to improve model robustness against noise and market irregularities.
    """
    def __init__(self, noise_level=0.02, mask_prob=0.1, time_warp_prob=0.2):
        self.noise_level = noise_level
        self.mask_prob = mask_prob
        self.time_warp_prob = time_warp_prob

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to a single sequence.
        x shape: [seq_len, features]
        """
        x_aug = x.clone()
        
        # 1. Gaussian Noise Injection
        # Simulates market noise and minor price fluctuations
        if np.random.random() < 0.5:
            noise = torch.randn_like(x_aug) * self.noise_level
            x_aug += noise

        # 2. Feature Masking (Dropout on Inputs)
        # Simulates missing data or indicator failure
        if np.random.random() < 0.3:
            mask = torch.bernoulli(torch.full_like(x_aug, 1 - self.mask_prob))
            x_aug *= mask
            
        # 3. Channel Scaling
        # Randomly scale specific features (e.g., volume spikes)
        if np.random.random() < 0.3:
            scale = 1.0 + (torch.randn(x_aug.shape[1]) * 0.1).to(x.device)
            x_aug *= scale

        return x_aug

    def mixup(self, x1, y1, x2, y2, alpha=0.2):
        """
        Blends two market scenarios.
        """
        lam = np.random.beta(alpha, alpha)
        x_mix = lam * x1 + (1 - lam) * x2
        # For classification, we might need soft labels, but for now let's keep it simple
        # This is more complex for classification targets, usually used with CrossEntropy
        return x_mix, lam * y1 + (1 - lam) * y2
