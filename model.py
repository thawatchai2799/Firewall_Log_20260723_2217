"""
model.py

MLP เดียวกันสำหรับทุกเทคนิคเปรียบเทียบใน imbalance_methods.py (ตามหลักการข้อ 2 ใน README:
แยกผลของ "เทคนิคจัดการ imbalance" ออกจาก "สถาปัตยกรรมโมเดล") — ค่า default ตั้งตาม
งานเดิม (200 hidden nodes, ReLU) เพื่อให้เทียบกับ baseline ได้ตรง
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
        return self.net(x)  # logits (raw, ยังไม่ softmax)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) — ใช้เปรียบเทียบกับ cost-sensitive CE ใน RQ1
    gamma: ยิ่งสูง ยิ่ง down-weight ตัวอย่างที่ classify ง่าย (majority class)
    alpha: per-class weight (เทียบเท่า inverse class frequency ก็ได้ หรือปล่อย None)
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
    """weight ∝ inverse class frequency — ใช้กับทั้ง weighted-CE และ alpha ของ focal loss"""
    counts = torch.bincount(y_train_encoded, minlength=n_classes).float()
    counts = torch.clamp(counts, min=1.0)  # กัน division by zero สำหรับ class ที่ไม่ปรากฏ
    weights = counts.sum() / (n_classes * counts)
    return weights
