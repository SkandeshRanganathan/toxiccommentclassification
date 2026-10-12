"""DeBERTa CLS baseline for six independent binary decisions."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.deberta_encoder import DebertaEncoder
from models.outputs import ClassifierOutput


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 6,
        dropout: float = 0.2,
        projection_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = DebertaEncoder(encoder_name, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.output_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_ids: Sequence[str] | None = None,
        **kwargs: torch.Tensor,
    ) -> ClassifierOutput:
        del sample_ids
        cls, _ = self.encoder(input_ids, attention_mask, **kwargs)
        logits = self.classifier(self.dropout(cls))
        return ClassifierOutput(logits=logits, cls_embeddings=cls, fused_embeddings=cls)
