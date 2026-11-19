"""
model.py

The same MLP is used for every technique compared in imbalance_methods.py (principle 2 in the README:
separate the effect of the "imbalance-handling technique" from the "model architecture"). Defaults follow
prior work (200 hidden nodes, ReLU) for direct comparability with the baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FirewallMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 200, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits (raw, before softmax)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) - compared against cost-sensitive CE in RQ1.
    gamma: higher values down-weight easily-classified (majority-class) samples more.
    alpha: per-class weight (e.g. inverse class frequency, or None).
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = -focal_weight * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(y_train_encoded: torch.Tensor, n_classes: int) -> torch.Tensor:
    """weight proportional to inverse class frequency - used for both weighted-CE and focal-loss alpha."""
    counts = torch.bincount(y_train_encoded, minlength=n_classes).float()
    counts = torch.clamp(counts, min=1.0)  # avoid division by zero for absent classes
    weights = counts.sum() / (n_classes * counts)
    return weights
