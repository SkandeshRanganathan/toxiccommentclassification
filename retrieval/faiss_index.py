"""Cosine-similarity FAISS indexes with a stable NumPy-facing API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - exercised on installations without FAISS
    faiss = None  # type: ignore[assignment]
    _FAISS_ERROR = exc
else:
    _FAISS_ERROR = None


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


class FaissCosineIndex:
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
    ) -> None:
        if faiss is None:
            raise ImportError("FAISS is required for prototype retrieval") from _FAISS_ERROR
        self.dimension = int(dimension)
        self.index_type = index_type.lower()
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, int(nlist), faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = int(nprobe)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(
                self.dimension, int(hnsw_m), faiss.METRIC_INNER_PRODUCT
            )
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        normalized = normalize_embeddings(vectors)
        if not self.index.is_trained:
            self.index.train(normalized)
        self.index.add(normalized)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise RuntimeError("Cannot search an empty FAISS index")
        k = min(int(top_k), self.size)
        scores, indices = self.index.search(normalize_embeddings(queries), k)
        return scores.astype(np.float32), indices.astype(np.int64)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(destination))

    @classmethod
    def load(cls, path: str | Path, index_type: str = "flat") -> "FaissCosineIndex":
        if faiss is None:
            raise ImportError("FAISS is required for prototype retrieval") from _FAISS_ERROR
        loaded = faiss.read_index(str(path))
        instance = cls.__new__(cls)
        instance.index = loaded
        instance.dimension = int(loaded.d)
        instance.index_type = index_type
        return instance

