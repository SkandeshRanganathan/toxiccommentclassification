"""Validation-only per-label threshold selection for macro-F1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class ThresholdResult:
    thresholds: np.ndarray
    per_label_f1: np.ndarray
    macro_f1: float


class PerLabelThresholdOptimizer:
    def __init__(self, grid_min: float = 0.05, grid_max: float = 0.95, grid_steps: int = 91, default: float = 0.5) -> None:
        if not 0 < grid_min < grid_max < 1:
            raise ValueError("Threshold grid must lie strictly inside (0, 1)")
        self.grid = np.linspace(grid_min, grid_max, grid_steps, dtype=np.float32)
        self.default = float(default)
        self.thresholds_: np.ndarray | None = None

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> ThresholdResult:
        if probabilities.shape != targets.shape or probabilities.ndim != 2:
            raise ValueError("probabilities and targets must be equally shaped 2-D arrays")
        thresholds = np.full(probabilities.shape[1], self.default, dtype=np.float32)
        scores = np.zeros(probabilities.shape[1], dtype=np.float32)
        for label in range(probabilities.shape[1]):
            if targets[:, label].sum() == 0:
                continue
            candidates = np.array([
                f1_score(targets[:, label], probabilities[:, label] >= threshold, zero_division=0)
                for threshold in self.grid
            ])
            best = np.flatnonzero(candidates == candidates.max())
            # Ties prefer the candidate closest to 0.5 for a conservative deterministic result.
            chosen = best[np.argmin(np.abs(self.grid[best] - self.default))]
            thresholds[label], scores[label] = self.grid[chosen], candidates[chosen]
        self.thresholds_ = thresholds
        return ThresholdResult(thresholds, scores, float(scores.mean()))

