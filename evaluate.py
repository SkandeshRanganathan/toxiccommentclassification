"""Evaluate a checkpoint, optimize per-label thresholds, and create paper-ready plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.factory import build_model
from models.threshold_optimizer import PerLabelThresholdOptimizer
from training.evaluator import evaluate_model
from training.losses import MultiLabelObjective
from utils.checkpoint import load_checkpoint, update_checkpoint_thresholds
from utils.config import load_config
from utils.device import resolve_device
from utils.io import atomic_write_json
from utils.metrics import apply_thresholds, multilabel_metrics
from utils.runtime import build_loader, build_tokenizer
from utils.visualization import plot_confusion_matrices, plot_per_label_metrics, plot_precision_recall_curves, plot_roc_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--output", default="outputs/evaluation")
    args = parser.parse_args()
    config = load_config(args.config, args.experiment)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = resolve_device(config.training.mixed_precision)
    tokenizer = build_tokenizer(config)
    model = build_model(config).to(policy.device)
    load_checkpoint(args.checkpoint, model, device=policy.device)
    objective = MultiLabelObjective(auxiliary_weight=config.label_correlation.regularization_weight)
    validation = evaluate_model(model, build_loader(config, tokenizer, "validation", False), objective, policy.device, config.data.labels, amp_context=policy.autocast, description="Validation")
    thresholds = np.full(config.model.num_labels, config.thresholds.default, dtype=np.float32)
    if config.thresholds.optimize:
        optimized = PerLabelThresholdOptimizer(config.thresholds.grid_min, config.thresholds.grid_max, config.thresholds.grid_steps, config.thresholds.default).fit(validation.probabilities, validation.targets)
        thresholds = optimized.thresholds
        update_checkpoint_thresholds(args.checkpoint, thresholds)
    atomic_write_json({"labels": list(config.data.labels), "thresholds": thresholds.tolist()}, output_dir / "thresholds.json")
    test = evaluate_model(model, build_loader(config, tokenizer, "test", False), objective, policy.device, config.data.labels, thresholds, policy.autocast, "Test")
    atomic_write_json({"labels": list(config.data.labels), "thresholds": thresholds.tolist(), "validation_metrics_fixed_0.5": validation.metrics, "test_metrics": test.metrics}, output_dir / "metrics.json")
    pd.DataFrame({"id": test.sample_ids, **{f"target_{label}": test.targets[:, i] for i, label in enumerate(config.data.labels)}, **{f"probability_{label}": test.probabilities[:, i] for i, label in enumerate(config.data.labels)}}).to_csv(output_dir / "predictions.csv", index=False)
    if config.evaluation.generate_plots:
        predictions = apply_thresholds(test.probabilities, thresholds)
        plot_confusion_matrices(test.targets, predictions, config.data.labels, output_dir / "confusion_matrices.png")
        plot_roc_curves(test.targets, test.probabilities, config.data.labels, output_dir / "roc_curves.png")
        plot_precision_recall_curves(test.targets, test.probabilities, config.data.labels, output_dir / "precision_recall_curves.png")
        plot_per_label_metrics(test.metrics, config.data.labels, output_dir / "per_label_metrics.png")
    print(json.dumps(test.metrics, indent=2))


if __name__ == "__main__":
    main()
