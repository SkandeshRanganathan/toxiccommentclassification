"""Reusable DeBERTa encoder whose CLS vectors power classification and retrieval."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class DebertaEncoder(nn.Module):
    def __init__(self, model_name: str, projection_dim: int | None = None) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        native_dim = int(self.transformer.config.hidden_size)
        self.output_dim = int(projection_dim or native_dim)
        self.projection = (
            nn.Sequential(nn.Linear(native_dim, self.output_dim), nn.LayerNorm(self.output_dim))
            if projection_dim and projection_dim != native_dim
            else nn.Identity()
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        tokens = outputs.last_hidden_state
        cls = self.projection(tokens[:, 0])
        return cls, tokens
