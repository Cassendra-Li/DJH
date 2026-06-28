"""
Loss Functions for GW-based Domain Alignment
=============================================

Includes:
- GW alignment loss (from GWAlignment module)
- MMD loss (for DAGCN-style baseline comparison)
- Combined training loss
"""

import torch
import torch.nn as nn
from typing import Dict


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (MMD) loss — used for DAGCN baseline comparison.

    MMD² = E[k(x,x)] + E[k(y,y)] - 2·E[k(x,y)]
    where k is a multi-kernel RBF (Gaussian).

    This is a direct PyTorch re-implementation of DAGCN's DAN loss.
    """
    n_s = source.size(0)
    n_t = target.size(0)
    total = torch.cat([source, target], dim=0)

    # Compute pairwise L2 distances
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_dist = ((total0 - total1) ** 2).sum(2)

    # Multi-kernel bandwidth
    bandwidth = torch.sum(L2_dist.data) / (total.size(0) ** 2 - total.size(0))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # Sum of Gaussian kernels
    kernel_sum = sum(torch.exp(-L2_dist / bw) for bw in bandwidth_list)

    # MMD²
    XX = kernel_sum[:n_s, :n_s]
    YY = kernel_sum[n_s:, n_s:]
    XY = kernel_sum[:n_s, n_s:]

    loss = torch.mean(XX + YY - 2 * XY)
    return loss


class CombinedLoss(nn.Module):
    """
    Combined training loss for RPGW-Net.

    L_total = λ_cls · L_cls + λ_gw · L_gw + λ_mmd · L_mmd

    Where:
    - L_cls: cross-entropy on source domain (labeled)
    - L_gw:  GW structure alignment loss
    - L_mmd: optional MMD loss on features (ablated)
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        gw_weight: float = 0.5,
        mmd_weight: float = 0.3,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.gw_weight = gw_weight
        self.mmd_weight = mmd_weight

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: dict with:
                'logits':      (B_s, num_classes) classification logits
                'labels':      (B_s,) ground truth labels
                'gw_loss':     scalar GW alignment loss
                'mmd_loss':    (optional) scalar MMD loss

        Returns:
            {'total': total_loss, 'cls': cls_loss, 'gw': gw_loss, ...}
        """
        cls_loss = self.ce_loss(outputs["logits"], outputs["labels"])

        total = self.cls_weight * cls_loss

        if "gw_loss" in outputs:
            total = total + self.gw_weight * outputs["gw_loss"]

        if "mmd_loss" in outputs:
            total = total + self.mmd_weight * outputs["mmd_loss"]

        loss_dict = {
            "total": total,
            "cls":   cls_loss,
            "gw":    outputs.get("gw_loss", torch.tensor(0.0)),
            "mmd":   outputs.get("mmd_loss", torch.tensor(0.0)),
        }
        return loss_dict
