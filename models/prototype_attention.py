"""Prototype cross-attention and gated residual fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AttentionOutput:
    fused: torch.Tensor
    context: torch.Tensor
    attention_weights: torch.Tensor
    fusion_gate: torch.Tensor


class GatedPrototypeFusion(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.context_projection = nn.Linear(embedding_dim, embedding_dim)
        self.gate = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.context_projection(context)
        gate = torch.sigmoid(
            self.gate(torch.cat([query, projected, torch.abs(query - projected), query * projected], dim=-1))
        )
        return self.norm(query + self.dropout(gate * projected)), gate


class PrototypeAwareCrossAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.prototype_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fusion = GatedPrototypeFusion(embedding_dim, dropout)

    def forward(
        self,
        query: torch.Tensor,
        prototypes: torch.Tensor,
        selector_weights: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> AttentionOutput:
        values = self.prototype_norm(prototypes)
        if selector_weights is not None:
            # Reweight values while preserving their scale under uniform selection.
            values = values * (selector_weights * prototypes.shape[1]).unsqueeze(-1)
        context, attention = self.attention(
            self.query_norm(query).unsqueeze(1),
            values,
            values,
            key_padding_mask=None if valid_mask is None else ~valid_mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        context = context.squeeze(1)
        attention = attention.squeeze(1)
        if selector_weights is not None:
            attention = attention * selector_weights
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(attention.dtype).eps
            )
        fused, gate = self.fusion(query, context)
        return AttentionOutput(fused, context, attention, gate)
