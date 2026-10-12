"""Deterministic evaluator returning metrics and reusable prediction arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.losses import MultiLabelObjective
from utils.metrics import multilabel_metrics


@dataclass
class EvaluationResult:
    loss: float
    metrics: dict[str, float]
    targets: np.ndarray
    probabilities: np.ndarray
    sample_ids: list[str]
    comments: list[str]


def model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for key, value in batch.items():
        if key in {"labels", "comment_texts"}:
            continue
        inputs[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return inputs


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    objective: MultiLabelObjective,
    device: torch.device,
    label_names: Sequence[str],
    thresholds: float | Sequence[float] = 0.5,
    amp_context: Any | None = None,
    description: str = "Evaluating",
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    all_targets, all_probabilities = [], []
    sample_ids: list[str] = []
    comments: list[str] = []
    for batch in tqdm(loader, desc=description, leave=False):
        labels = batch["labels"].to(device, non_blocking=True)
        context = amp_context() if amp_context is not None else torch.autocast(device_type=device.type, enabled=False)
        with context:
            output = model(**model_inputs(batch, device))
            loss = objective(output.logits, labels, output.auxiliary_loss)
        total_loss += float(loss.item()) * labels.shape[0]
        all_targets.append(labels.cpu().numpy())
        all_probabilities.append(torch.sigmoid(output.logits.float()).cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        comments.extend(batch["comment_texts"])
    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    metrics = multilabel_metrics(targets, probabilities, thresholds, label_names)
    average_loss = total_loss / len(loader.dataset)
    metrics["loss"] = average_loss
    return EvaluationResult(average_loss, metrics, targets, probabilities, sample_ids, comments)

