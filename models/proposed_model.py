"""Composable end-to-end prototype-augmented classifier used by every ablation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.adaptive_selector import AdaptivePrototypeSelector
from models.deberta_encoder import DebertaEncoder
from models.label_correlation import LabelCorrelationModel
from models.outputs import ClassifierOutput
from models.prototype_attention import GatedPrototypeFusion, PrototypeAwareCrossAttention
from models.prototype_memory import PrototypeMemory


class PrototypeAugmentedClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 6,
        dropout: float = 0.2,
        projection_dim: int | None = None,
        memory: PrototypeMemory | None = None,
        use_prototypes: bool = True,
        use_adaptive_selector: bool = True,
        use_prototype_attention: bool = True,
        use_label_correlation: bool = True,
        selector_hidden_dim: int = 256,
        selector_dropout: float = 0.1,
        selector_temperature: float = 0.7,
        selected_k: int = 8,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        correlation_hidden_dim: int = 256,
        correlation_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if use_prototypes and memory is None:
            raise ValueError("Prototype-enabled models require a loaded PrototypeMemory")
        self.encoder = DebertaEncoder(encoder_name, projection_dim)
        dimension = self.encoder.output_dim
        self.memory = memory
        self.use_prototypes = use_prototypes
        self.use_adaptive_selector = use_adaptive_selector
        self.use_prototype_attention = use_prototype_attention
        self.use_label_correlation = use_label_correlation
        self.dropout = nn.Dropout(dropout)
        self.base_classifier = nn.Linear(dimension, num_labels)
        self.final_classifier = nn.Linear(dimension, num_labels)
        self.neighbor_logit_scale = nn.Parameter(torch.tensor(-1.0))
        self.selector = AdaptivePrototypeSelector(
            dimension, selector_hidden_dim, selector_dropout, selector_temperature, selected_k
        )
        self.cross_attention = PrototypeAwareCrossAttention(
            dimension, attention_heads, attention_dropout
        )
        self.simple_fusion = GatedPrototypeFusion(dimension, attention_dropout)
        self.correlation = LabelCorrelationModel(
            dimension, num_labels, correlation_hidden_dim, correlation_dropout
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_ids: Sequence[str] | None = None,
        **kwargs: torch.Tensor,
    ) -> ClassifierOutput:
        cls, _ = self.encoder(input_ids, attention_mask, **kwargs)
        preliminary_logits = self.base_classifier(self.dropout(cls))
        representation = cls
        retrieval = selector = attention = None
        if self.use_prototypes:
            assert self.memory is not None
            retrieval = self.memory.retrieve(cls, sample_ids)
            if self.use_adaptive_selector:
                selector = self.selector(
                    cls,
                    retrieval.embeddings,
                    retrieval.similarities,
                    retrieval.labels,
                    torch.sigmoid(preliminary_logits),
                    retrieval.valid_mask,
                )
                weights, selected_mask = selector.weights, selector.selected_mask
            else:
                similarities = retrieval.similarities.masked_fill(~retrieval.valid_mask, -torch.inf)
                weights = torch.nan_to_num(torch.softmax(similarities, dim=-1), nan=0.0)
                selected_mask = retrieval.valid_mask
            if self.use_prototype_attention:
                attention = self.cross_attention(
                    cls, retrieval.embeddings, weights, selected_mask
                )
                representation = attention.fused
            elif self.use_adaptive_selector:
                context = torch.sum(weights.unsqueeze(-1) * retrieval.embeddings, dim=1)
                representation, _ = self.simple_fusion(cls, context)

        auxiliary_loss = None
        correlation_matrix = None
        if self.use_label_correlation:
            correlated = self.correlation(self.dropout(representation))
            logits = correlated.logits
            correlation_matrix = correlated.correlation_matrix
            auxiliary_loss = correlated.regularization
        elif self.use_prototypes and not self.use_prototype_attention and not self.use_adaptive_selector:
            # Retrieval-only control: similarity-weighted prototype label evidence calibrates logits.
            assert retrieval is not None
            weights = torch.nan_to_num(
                torch.softmax(retrieval.similarities.masked_fill(~retrieval.valid_mask, -torch.inf), dim=-1), nan=0.0
            )
            neighbor_probability = (weights.unsqueeze(-1) * retrieval.labels).sum(dim=1).clamp(1e-4, 1 - 1e-4)
            neighbor_logits = torch.logit(neighbor_probability)
            logits = preliminary_logits + torch.sigmoid(self.neighbor_logit_scale) * neighbor_logits
        elif self.use_prototypes:
            logits = self.final_classifier(self.dropout(representation))
        else:
            logits = preliminary_logits
        return ClassifierOutput(
            logits=logits,
            cls_embeddings=cls,
            fused_embeddings=representation,
            retrieved_indices=None if retrieval is None else retrieval.indices,
            similarities=None if retrieval is None else retrieval.similarities,
            selector_scores=None if selector is None else selector.scores,
            selector_weights=None if selector is None else selector.weights,
            attention_weights=None if attention is None else attention.attention_weights,
            prototype_labels=None if retrieval is None else retrieval.labels,
            retrieved_sample_ids=None if retrieval is None else retrieval.sample_ids,
            retrieved_comments=None if retrieval is None else retrieval.comments,
            correlation_matrix=correlation_matrix,
            auxiliary_loss=auxiliary_loss,
        )
