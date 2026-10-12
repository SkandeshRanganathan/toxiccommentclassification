"""Export an auditable checkpoint bundle for evaluation or deployment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from utils.config import load_config, save_resolved_config
from utils.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-memory", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.experiment)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, args.output / "model.pt")
    save_resolved_config(config, args.output / "config.yaml")
    thresholds = args.checkpoint.parent / "thresholds.json"
    if thresholds.is_file():
        shutil.copy2(thresholds, args.output / "thresholds.json")
    if args.include_memory and config.model.use_prototypes:
        shutil.copytree(config.retrieval.memory_dir, args.output / "prototype_memory", dirs_exist_ok=True)
    atomic_write_json({"format_version": 1, "checkpoint": "model.pt", "config": "config.yaml", "includes_prototype_memory": bool(args.include_memory and config.model.use_prototypes)}, args.output / "manifest.json")
    print(f"Exported model bundle to {args.output}")


if __name__ == "__main__":
    main()

