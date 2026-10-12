"""Factories shared by command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.collator import ToxicDataCollator
from data.dataset import ToxicCommentDataset
from utils.seed import seed_worker


def build_tokenizer(config: Any) -> Any:
    return AutoTokenizer.from_pretrained(config.model.encoder_name, use_fast=True)


def build_loader(
    config: Any,
    tokenizer: Any,
    split: str,
    shuffle: bool,
    include_labels: bool = True,
    batch_size: int | None = None,
) -> DataLoader:
    path = Path(config.data.processed_dir) / f"{split}.csv"
    dataset = ToxicCommentDataset(
        path,
        tokenizer,
        config.data.labels,
        config.data.max_length,
        config.data.text_column,
        config.data.id_column,
        include_labels,
    )
    generator = torch.Generator().manual_seed(int(config.training.seed))
    return DataLoader(
        dataset,
        batch_size=batch_size or (config.training.batch_size if shuffle else config.training.eval_batch_size),
        shuffle=shuffle,
        num_workers=int(config.data.num_workers),
        pin_memory=bool(config.data.pin_memory and torch.cuda.is_available()),
        collate_fn=ToxicDataCollator(tokenizer, 8 if torch.cuda.is_available() else None),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=int(config.data.num_workers) > 0,
    )

