"""PyTorch dataset that tokenizes comments on demand."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset


class ToxicCommentDataset(Dataset[dict[str, Any]]):
    """Keep raw text in memory and defer tokenization to each sample access."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        tokenizer: Any,
        labels: Sequence[str],
        max_length: int,
        text_column: str = "comment_text",
        id_column: str = "id",
        include_labels: bool = True,
    ) -> None:
        self.frame = pd.read_csv(data) if isinstance(data, (str, Path)) else data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.labels = tuple(labels)
        self.max_length = int(max_length)
        self.text_column = text_column
        self.id_column = id_column
        self.include_labels = include_labels
        required = {text_column, id_column, *(self.labels if include_labels else ())}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Dataset input is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row[self.text_column]),
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        item: dict[str, Any] = {
            **encoded,
            "sample_id": str(row[self.id_column]),
            "comment_text": str(row[self.text_column]),
        }
        if self.include_labels:
            item["labels"] = torch.tensor(row[list(self.labels)].to_numpy(dtype="float32"))
        return item

