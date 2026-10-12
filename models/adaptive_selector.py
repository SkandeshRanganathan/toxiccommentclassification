"""Differentiable, leakage-free scoring of retrieved prototype candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SelectorOutput:
    scores: torch.Tensor
    weights: torch.Tensor
    selected_mask: torch.Tensor
    selected_indices: torch.Tensor


class AdaptivePrototypeSelector(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        temperature: float = 0.7,
        hard_top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.hard_top_k = hard_top_k
        feature_dim = 4 * embedding_dim + 4
        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor,
        similarities: torch.Tensor,
        prototype_labels: torch.Tensor,
        provisional_probabilities: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> SelectorOutput:
        batch, candidates, _ = prototypes.shape
        expanded_query = query.unsqueeze(1).expand_as(prototypes)
        probabilities = provisional_probabilities.unsqueeze(1).expand(-1, candidates, -1)
        compatibility = 1.0 - torch.abs(probabilities - prototype_labels).mean(dim=-1)
        eps = torch.finfo(provisional_probabilities.dtype).eps
        entropy = -(provisional_probabilities.clamp(eps, 1 - eps) * provisional_probabilities.clamp(eps, 1 - eps).log() + (1 - provisional_probabilities).clamp(eps, 1 - eps) * (1 - provisional_probabilities).clamp(eps, 1 - eps).log()).mean(dim=-1)
        confidence = 1.0 - entropy / torch.log(torch.tensor(2.0, device=query.device))
        label_density = prototype_labels.mean(dim=-1)
        safe_similarities = similarities.masked_fill(~valid, 0.0)
        features = torch.cat(
            [expanded_query, prototypes, expanded_query * prototypes, torch.abs(expanded_query - prototypes), safe_similarities.unsqueeze(-1), compatibility.unsqueeze(-1), confidence[:, None, None].expand(batch, candidates, 1), label_density.unsqueeze(-1)],
            dim=-1,
        )
        scores = self.scorer(features).squeeze(-1)
        valid = torch.ones_like(scores, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        scores = scores.masked_fill(~valid, -torch.inf)
        selected_mask = valid.clone()
        selection_count = min(self.hard_top_k or candidates, candidates)
        if selection_count < candidates:
            selected_indices = scores.topk(selection_count, dim=-1).indices
            selected_mask = torch.zeros_like(valid).scatter(1, selected_indices, True) & valid
        else:
            selected_indices = torch.arange(candidates, device=query.device).expand(batch, -1)
        masked_scores = scores.masked_fill(~selected_mask, -torch.inf)
        weights = torch.softmax(masked_scores / self.temperature, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return SelectorOutput(scores, weights, selected_mask, selected_indices)
