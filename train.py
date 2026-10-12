"""Train a baseline or configured prototype-aware ablation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from models.factory import build_model
from models.threshold_optimizer import PerLabelThresholdOptimizer
from training.evaluator import evaluate_model
from training.losses import MultiLabelObjective
from training.trainer import Trainer
from utils.checkpoint import load_checkpoint, update_checkpoint_thresholds
from utils.config import load_config, save_resolved_config
from utils.device import resolve_device
from utils.experiment_tracking import append_result
from utils.logger import configure_logger
from utils.io import atomic_write_json
from utils.runtime import build_loader, build_tokenizer
from utils.seed import seed_everything


def initialize_from_baseline(model: object, checkpoint: str, device: object) -> None:
    """Transfer compatible encoder/head weights while leaving novel modules initialized."""
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    source = payload["model_state"]
    remapped = dict(source)
    if "classifier.weight" in source:
        remapped["base_classifier.weight"] = source["classifier.weight"]
        remapped["base_classifier.bias"] = source["classifier.bias"]
        remapped["final_classifier.weight"] = source["classifier.weight"]
        remapped["final_classifier.bias"] = source["classifier.bias"]
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in remapped.items()
        if key in target and target[key].shape == value.shape
    }
    target.update(compatible)
    model.load_state_dict(target)


def run_training(
    base_config: str,
    experiment_config: str | None = None,
    initialization_checkpoint: str | None = None,
) -> dict[str, float]:
    config = load_config(base_config, experiment_config)
    seed_everything(config.training.seed, config.training.deterministic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.project.output_dir) / config.project.experiment_name / timestamp
    logger = configure_logger(f"train.{config.project.experiment_name}", run_dir / "train.log")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    policy = resolve_device(config.training.mixed_precision)
    logger.info("Using device=%s mixed_precision=%s", policy.device, policy.amp_enabled)
    tokenizer = build_tokenizer(config)
    train_loader = build_loader(config, tokenizer, "train", True)
    validation_loader = build_loader(config, tokenizer, "validation", False)
    model = build_model(config)
    initialization = initialization_checkpoint or config.model.initialization_checkpoint
    if initialization:
        initialize_from_baseline(model, initialization, policy.device)
        logger.info("Initialized architecture from baseline checkpoint %s", initialization)
    objective = MultiLabelObjective(auxiliary_weight=config.label_correlation.regularization_weight)
    metrics = Trainer(model, train_loader, validation_loader, objective, config, policy, logger, config.data.labels, run_dir).fit()
    if config.thresholds.optimize and (run_dir / "best.pt").is_file():
        load_checkpoint(run_dir / "best.pt", model, device=policy.device)
        validation = evaluate_model(model, validation_loader, objective, policy.device, config.data.labels, amp_context=policy.autocast, description="Fitting thresholds")
        optimized = PerLabelThresholdOptimizer(config.thresholds.grid_min, config.thresholds.grid_max, config.thresholds.grid_steps, config.thresholds.default).fit(validation.probabilities, validation.targets)
        update_checkpoint_thresholds(run_dir / "best.pt", optimized.thresholds)
        atomic_write_json({"labels": list(config.data.labels), "thresholds": optimized.thresholds.tolist(), "per_label_f1": optimized.per_label_f1.tolist(), "macro_f1": optimized.macro_f1}, run_dir / "thresholds.json")
        metrics = {**metrics, "optimized_macro_f1": optimized.macro_f1}
    append_result(Path(config.project.output_dir) / "experiment_results.csv", {"experiment": config.project.experiment_name, "run_dir": str(run_dir), **metrics})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default="configs/experiments/baseline.yaml")
    parser.add_argument("--initialize-from", help="Baseline checkpoint for encoder/head initialization")
    args = parser.parse_args()
    run_training(args.config, args.experiment, args.initialize_from)


if __name__ == "__main__":
    main()
