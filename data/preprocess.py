"""Validate, clean, and reproducibly partition the official training data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import load_config
from utils.io import atomic_write_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_clean(
    frame: pd.DataFrame,
    id_column: str,
    text_column: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    required = [id_column, text_column, *labels]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    clean = frame.loc[:, required].dropna(subset=required).copy()
    clean[text_column] = clean[text_column].astype(str).str.strip()
    clean = clean.loc[clean[text_column].str.len() > 0]
    if clean[id_column].duplicated().any():
        raise ValueError("Official sample IDs must be unique")
    for label in labels:
        values = set(pd.to_numeric(clean[label], errors="raise").unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"Label {label!r} contains non-binary values: {values}")
        clean[label] = clean[label].astype(np.int8)
    return clean.reset_index(drop=True)


def _prevalence_error(parts: Sequence[pd.DataFrame], labels: Sequence[str], target: np.ndarray) -> float:
    return float(sum(np.abs(part[list(labels)].mean().to_numpy() - target).mean() for part in parts))


def prevalence_preserving_split(
    frame: pd.DataFrame,
    labels: Sequence[str],
    validation_size: float,
    test_size: float,
    seed: int,
    trials: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Choose the best seeded split by multilabel prevalence without extra dependencies."""
    target = frame[list(labels)].mean().to_numpy()
    best: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    best_error = float("inf")
    holdout_size = validation_size + test_size
    relative_test = test_size / holdout_size
    # Rare label combinations cannot be safely stratified by sklearn, hence repeated seeded search.
    for offset in range(max(1, trials)):
        train, holdout = train_test_split(
            frame, test_size=holdout_size, random_state=seed + offset, shuffle=True
        )
        validation, test = train_test_split(
            holdout, test_size=relative_test, random_state=seed + offset, shuffle=True
        )
        error = _prevalence_error((train, validation, test), labels, target)
        all_labels_present = all((part[list(labels)].sum(axis=0) > 0).all() for part in (train, validation, test))
        if all_labels_present and error < best_error:
            best, best_error = (train, validation, test), error
    if best is None:
        raise RuntimeError("Could not create splits containing positives for every label")
    return tuple(part.sample(frac=1, random_state=seed).reset_index(drop=True) for part in best)  # type: ignore[return-value]


def preprocess(config_path: str | Path = "configs/config.yaml") -> dict[str, object]:
    config = load_config(config_path)
    raw_path = Path(config.data.raw_train_path)
    frame = validate_and_clean(
        pd.read_csv(raw_path), config.data.id_column, config.data.text_column, config.data.labels
    )
    train, validation, test = prevalence_preserving_split(
        frame,
        config.data.labels,
        float(config.data.validation_size),
        float(config.data.test_size),
        int(config.training.seed),
        int(config.data.split_search_trials),
    )
    output = Path(config.data.processed_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "validation": validation, "test": test}
    for name, split in splits.items():
        split.to_csv(output / f"{name}.csv", index=False)
    manifest: dict[str, object] = {
        "source": str(raw_path),
        "source_sha256": file_sha256(raw_path),
        "seed": int(config.training.seed),
        "removed_rows": int(len(pd.read_csv(raw_path, usecols=[config.data.id_column])) - len(frame)),
        "labels": list(config.data.labels),
        "splits": {
            name: {
                "rows": len(split),
                "sha256": file_sha256(output / f"{name}.csv"),
                "prevalence": split[list(config.data.labels)].mean().to_dict(),
                "positives": split[list(config.data.labels)].sum().astype(int).to_dict(),
            }
            for name, split in splits.items()
        },
    }
    atomic_write_json(manifest, output / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    manifest = preprocess(args.config)
    print(f"Prepared {sum(item['rows'] for item in manifest['splits'].values())} samples")  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
