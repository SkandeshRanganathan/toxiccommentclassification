"""Batched, inference-only extraction of float32 CLS prototype vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class ExtractedEmbeddings:
    embeddings: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    comments: list[str]


@torch.inference_mode()
def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_context: Any | None = None) -> ExtractedEmbeddings:
    model.eval()
    vectors, labels, sample_ids, comments = [], [], [], []
    for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor) and key != "labels"}
        context = amp_context() if amp_context else torch.autocast(device_type=device.type, enabled=False)
        with context:
            output = model(**inputs)
        vectors.append(output.cls_embeddings.float().cpu().numpy())
        labels.append(batch["labels"].numpy())
        sample_ids.extend(batch["sample_ids"])
        comments.extend(batch["comment_texts"])
    return ExtractedEmbeddings(np.concatenate(vectors), np.concatenate(labels), sample_ids, comments)

