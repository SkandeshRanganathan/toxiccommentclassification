"""Loss construction for independent multi-label targets."""

from __future__ import annotations

import torch
from torch import nn


class MultiLabelObjective(nn.Module):
    """BCE-with-logits plus an explicitly weighted architectural regularizer."""

    def __init__(self, positive_weights: torch.Tensor | None = None, auxiliary_weight: float = 0.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
        self.auxiliary_weight = float(auxiliary_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, auxiliary_loss: torch.Tensor | None = None
    ) -> torch.Tensor:
        loss = self.bce(logits, targets.float())
        if auxiliary_loss is not None and self.auxiliary_weight > 0:
            loss = loss + self.auxiliary_weight * auxiliary_loss
        return loss

