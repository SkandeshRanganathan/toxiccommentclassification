"""Learnable graph-style propagation between label-specific representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CorrelationOutput:
    logits: torch.Tensor
    correlation_matrix: torch.Tensor
    regularization: torch.Tensor


class LabelCorrelationModel(nn.Module):
    def __init__(self, embedding_dim: int, num_labels: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.label_embeddings = nn.Parameter(torch.empty(num_labels, embedding_dim))
        nn.init.normal_(self.label_embeddings, std=0.02)
        self.label_encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.correlation_logits = nn.Parameter(torch.eye(num_labels) * 2.0)
        self.propagation_strength = nn.Parameter(torch.tensor(-1.0))

    def forward(self, representation: torch.Tensor) -> CorrelationOutput:
        label_states = representation.unsqueeze(1) + self.label_embeddings.unsqueeze(0)
        base_logits = self.label_encoder(label_states).squeeze(-1)
        correlation = torch.softmax(self.correlation_logits, dim=-1)
        propagated = base_logits @ correlation.transpose(0, 1)
        logits = base_logits + torch.sigmoid(self.propagation_strength) * propagated
        identity = torch.eye(self.num_labels, device=correlation.device, dtype=correlation.dtype)
        symmetry = (correlation - correlation.transpose(0, 1)).pow(2).mean()
        diagonal_anchor = ((correlation * identity).sum(dim=-1) - 1.0).pow(2).mean()
        return CorrelationOutput(logits, correlation, symmetry + 0.1 * diagonal_anchor)

