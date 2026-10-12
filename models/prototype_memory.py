"""Torch-facing adapter around the immutable FAISS prototype store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from retrieval.prototype_store import PrototypeStore, RetrievalBatch


@dataclass
class TorchRetrievalBatch:
    embeddings: torch.Tensor
    labels: torch.Tensor
    similarities: torch.Tensor
    indices: torch.Tensor
    valid_mask: torch.Tensor
    sample_ids: object
    comments: object


class PrototypeMemory:
    def __init__(self, store: PrototypeStore, top_k: int, exclude_self: bool = True) -> None:
        self.store = store
        self.top_k = int(top_k)
        self.exclude_self = exclude_self

    def retrieve(
        self, query_embeddings: torch.Tensor, query_ids: Sequence[str] | None = None
    ) -> TorchRetrievalBatch:
        result: RetrievalBatch = self.store.retrieve(
            query_embeddings.detach().float().cpu().numpy(),
            self.top_k,
            query_ids,
            self.exclude_self,
        )
        device = query_embeddings.device
        return TorchRetrievalBatch(
            embeddings=torch.from_numpy(result.embeddings).to(device),
            labels=torch.from_numpy(result.labels).to(device),
            similarities=torch.from_numpy(result.similarities).to(device),
            indices=torch.from_numpy(result.indices).to(device),
            valid_mask=torch.from_numpy(result.valid_mask).to(device),
            sample_ids=result.sample_ids,
            comments=result.comments,
        )

