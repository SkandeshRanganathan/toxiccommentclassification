"""Run the full reproducible ablation sequence, including baseline memory construction."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from data.preprocess import preprocess
from train import run_training
from training.experiment_runner import EXPERIMENTS, run_ablation_suite
from utils.config import load_config


def newest_baseline_checkpoint(output_dir: Path) -> Path:
    checkpoints = list((output_dir / "baseline").glob("*/best.pt"))
    if not checkpoints:
        raise FileNotFoundError("Baseline training did not produce a best checkpoint")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--experiments", nargs="*", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.skip_preprocess:
        preprocess(args.config)
    checkpoint = Path(args.baseline_checkpoint) if args.baseline_checkpoint else None
    if checkpoint is None:
        run_training(args.config, "configs/experiments/baseline.yaml")
        checkpoint = newest_baseline_checkpoint(Path(config.project.output_dir))
    if any(name in {"baseline_faiss", "adaptive_selector", "prototype_attention", "full_model"} for name in args.experiments):
        subprocess.run([sys.executable, "build_memory.py", "--checkpoint", str(checkpoint), "--config", args.config], check=True)
    run_ablation_suite(args.config, tuple(args.experiments), str(checkpoint))


if __name__ == "__main__":
    main()
