"""Versioned storage that keeps FAISS rows aligned with comments and labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from retrieval.faiss_index import FaissCosineIndex, normalize_embeddings
from utils.io import atomic_write_json


@dataclass
class RetrievalBatch:
    embeddings: np.ndarray
    labels: np.ndarray
    similarities: np.ndarray
    indices: np.ndarray
    sample_ids: np.ndarray
    comments: np.ndarray
    valid_mask: np.ndarray


class PrototypeStore:
    VERSION = 1

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        labels: Sequence[str],
        index: FaissCosineIndex,
    ) -> None:
        if len(embeddings) != len(metadata) or index.size != len(metadata):
            raise ValueError("Prototype embeddings, metadata, and FAISS index are misaligned")
        self.embeddings = normalize_embeddings(embeddings)
        self.metadata = metadata.reset_index(drop=True)
        self.labels = tuple(labels)
        self.index = index
        self.id_to_row = {str(value): row for row, value in enumerate(metadata["id"])}

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        sample_ids: Sequence[str],
        comments: Sequence[str],
        label_values: np.ndarray,
        labels: Sequence[str],
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
    ) -> "PrototypeStore":
        values = normalize_embeddings(embeddings)
        metadata = pd.DataFrame({"id": sample_ids, "comment_text": comments})
        for column, value in zip(labels, label_values.T, strict=True):
            metadata[column] = value.astype(np.int8)
        index = FaissCosineIndex(values.shape[1], index_type, nlist, nprobe, hnsw_m)
        index.add(values)
        return cls(values, metadata, labels, index)

    def retrieve(
        self,
        queries: np.ndarray,
        top_k: int,
        query_ids: Sequence[str] | None = None,
        exclude_self: bool = True,
    ) -> RetrievalBatch:
        extra = 1 if exclude_self and query_ids is not None else 0
        scores, rows = self.index.search(queries, min(top_k + extra, self.index.size))
        output_rows = np.full((len(queries), top_k), -1, dtype=np.int64)
        output_scores = np.full((len(queries), top_k), -np.inf, dtype=np.float32)
        valid = np.zeros((len(queries), top_k), dtype=bool)
        for batch_row in range(len(queries)):
            candidates = [
                (int(row), float(score))
                for row, score in zip(rows[batch_row], scores[batch_row], strict=True)
                if row >= 0
                and not (
                    exclude_self
                    and query_ids is not None
                    and str(self.metadata.iloc[int(row)]["id"]) == str(query_ids[batch_row])
                )
            ][:top_k]
            for rank, (row, score) in enumerate(candidates):
                output_rows[batch_row, rank] = row
                output_scores[batch_row, rank] = score
                valid[batch_row, rank] = True
        safe_rows = np.maximum(output_rows, 0)
        flat = safe_rows.reshape(-1)
        return RetrievalBatch(
            embeddings=self.embeddings[flat].reshape(len(queries), top_k, -1),
            labels=self.metadata.loc[flat, list(self.labels)].to_numpy(np.float32).reshape(len(queries), top_k, -1),
            similarities=output_scores,
            indices=output_rows,
            sample_ids=self.metadata.loc[flat, "id"].astype(str).to_numpy().reshape(len(queries), top_k),
            comments=self.metadata.loc[flat, "comment_text"].astype(str).to_numpy().reshape(len(queries), top_k),
            valid_mask=valid,
        )

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "embeddings.npy", self.embeddings)
        self.metadata.to_csv(destination / "metadata.csv", index=False)
        self.index.save(destination / "index.faiss")
        atomic_write_json(
            {"version": self.VERSION, "rows": len(self.metadata), "dimension": self.embeddings.shape[1], "labels": list(self.labels), "index_type": self.index.index_type},
            destination / "manifest.json",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "PrototypeStore":
        source = Path(directory)
        with (source / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("version") != cls.VERSION:
            raise ValueError(f"Unsupported prototype store version: {manifest.get('version')}")
        embeddings = np.load(source / "embeddings.npy")
        metadata = pd.read_csv(source / "metadata.csv", dtype={"id": str})
        index = FaissCosineIndex.load(source / "index.faiss", manifest["index_type"])
        if embeddings.shape != (manifest["rows"], manifest["dimension"]):
            raise ValueError("Prototype manifest does not match persisted embeddings")
        return cls(embeddings, metadata, manifest["labels"], index)

