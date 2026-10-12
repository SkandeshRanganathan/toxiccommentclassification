"""Programmatic ablation-suite orchestration."""

from __future__ import annotations

from pathlib import Path

from train import run_training


EXPERIMENTS = (
    "baseline_faiss",
    "adaptive_selector",
    "prototype_attention",
    "label_correlation",
    "adaptive_thresholds",
    "full_model",
)


def run_ablation_suite(
    config_path: str = "configs/config.yaml",
    experiments: tuple[str, ...] = EXPERIMENTS,
    baseline_checkpoint: str | None = None,
) -> None:
    for name in experiments:
        override = Path("configs/experiments") / f"{name}.yaml"
        run_training(config_path, str(override), baseline_checkpoint)
