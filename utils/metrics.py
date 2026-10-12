"""Complete multi-label metric suite with safe rare-label handling."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score, roc_auc_score


def apply_thresholds(probabilities: np.ndarray, thresholds: float | Sequence[float]) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(probabilities.shape[1], float(values), dtype=np.float32)
    if values.shape != (probabilities.shape[1],):
        raise ValueError("One threshold is required per label")
    return (probabilities >= values[None, :]).astype(np.int8)


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: float | Sequence[float] = 0.5,
    label_names: Sequence[str] | None = None,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError("targets and probabilities must be equally shaped 2-D arrays")
    predictions = apply_thresholds(probabilities, thresholds)
    metrics = {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(targets, predictions, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(targets, predictions, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "exact_match_ratio": float(accuracy_score(targets, predictions)),
        "subset_accuracy": float(accuracy_score(targets, predictions)),
    }
    aucs: list[float] = []
    names = list(label_names or [f"label_{index}" for index in range(targets.shape[1])])
    for index, name in enumerate(names):
        metrics[f"{name}_f1"] = float(f1_score(targets[:, index], predictions[:, index], zero_division=0))
        metrics[f"{name}_precision"] = float(precision_score(targets[:, index], predictions[:, index], zero_division=0))
        metrics[f"{name}_recall"] = float(recall_score(targets[:, index], predictions[:, index], zero_division=0))
        if np.unique(targets[:, index]).size == 2:
            auc = float(roc_auc_score(targets[:, index], probabilities[:, index]))
            metrics[f"{name}_roc_auc"] = auc
            aucs.append(auc)
        else:
            metrics[f"{name}_roc_auc"] = float("nan")
    metrics["macro_roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return metrics

