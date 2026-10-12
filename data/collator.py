"""Dynamic-padding collator that preserves metadata outside tokenizer tensors."""

from __future__ import annotations

from typing import Any

import torch
from transformers import DataCollatorWithPadding


class ToxicDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = None) -> None:
        self.padding = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest", pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        metadata = {
            "sample_ids": [item.pop("sample_id") for item in features],
            "comment_texts": [item.pop("comment_text") for item in features],
        }
        labels = [item.pop("labels") for item in features] if "labels" in features[0] else None
        batch = dict(self.padding(features))
        batch.update(metadata)
        if labels is not None:
            batch["labels"] = torch.stack(labels)
        return batch

