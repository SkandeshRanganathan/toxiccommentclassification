"""Typed output contracts shared by training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ClassifierOutput:
    logits: torch.Tensor
    cls_embeddings: torch.Tensor
    fused_embeddings: torch.Tensor | None = None
    retrieved_indices: torch.Tensor | None = None
    similarities: torch.Tensor | None = None
    selector_scores: torch.Tensor | None = None
    selector_weights: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None
    prototype_labels: torch.Tensor | None = None
    retrieved_sample_ids: Any | None = None
    retrieved_comments: Any | None = None
    correlation_matrix: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None
