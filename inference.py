"""Explainable toxicity inference for arbitrary comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.collator import ToxicDataCollator
from data.dataset import ToxicCommentDataset
from models.factory import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import resolve_device
from utils.runtime import build_tokenizer


def load_thresholds(checkpoint_payload: dict[str, Any], checkpoint: Path, count: int, default: float) -> np.ndarray:
    if checkpoint_payload.get("thresholds") is not None:
        return np.asarray(checkpoint_payload["thresholds"], dtype=np.float32)
    candidate = checkpoint.parent / "thresholds.json"
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as handle:
            return np.asarray(json.load(handle)["thresholds"], dtype=np.float32)
    return np.full(count, default, dtype=np.float32)


@torch.inference_mode()
def predict(comments: list[str], checkpoint: str, config_path: str, experiment: str | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path, experiment)
    policy = resolve_device(config.training.mixed_precision)
    tokenizer = build_tokenizer(config)
    frame = pd.DataFrame({config.data.id_column: [f"inference-{i}" for i in range(len(comments))], config.data.text_column: comments})
    dataset = ToxicCommentDataset(frame, tokenizer, config.data.labels, config.data.max_length, config.data.text_column, config.data.id_column, include_labels=False)
    loader = DataLoader(dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=ToxicDataCollator(tokenizer, 8 if policy.device.type == "cuda" else None))
    model = build_model(config).to(policy.device)
    payload = load_checkpoint(checkpoint, model, device=policy.device)
    thresholds = load_thresholds(payload, Path(checkpoint), config.model.num_labels, config.thresholds.default)
    model.eval()
    reports: list[dict[str, Any]] = []
    cursor = 0
    for batch in loader:
        inputs = {key: value.to(policy.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with policy.autocast():
            output = model(**inputs, sample_ids=batch["sample_ids"])
        probabilities = torch.sigmoid(output.logits.float()).cpu().numpy()
        for local_index in range(len(batch["sample_ids"])):
            confidence = {label: float(probabilities[local_index, i]) for i, label in enumerate(config.data.labels)}
            report: dict[str, Any] = {
                "comment": comments[cursor + local_index],
                "predicted_labels": [label for i, label in enumerate(config.data.labels) if probabilities[local_index, i] >= thresholds[i]],
                "confidence_scores": confidence,
                "thresholds": {label: float(thresholds[i]) for i, label in enumerate(config.data.labels)},
                "retrieved_prototypes": [],
            }
            if output.retrieved_indices is not None:
                indices = output.retrieved_indices[local_index].detach().cpu().numpy()
                similarities = output.similarities[local_index].detach().float().cpu().numpy()
                selector = output.selector_weights[local_index].detach().float().cpu().numpy() if output.selector_weights is not None else None
                attention = output.attention_weights[local_index].detach().float().cpu().numpy() if output.attention_weights is not None else None
                labels = output.prototype_labels[local_index].detach().cpu().numpy()
                for rank, row_index in enumerate(indices):
                    if row_index < 0:
                        continue
                    selector_weight = float(selector[rank]) if selector is not None else None
                    report["retrieved_prototypes"].append({
                        "rank": rank + 1,
                        "sample_id": str(output.retrieved_sample_ids[local_index, rank]),
                        "comment": str(output.retrieved_comments[local_index, rank]),
                        "similarity": float(similarities[rank]),
                        "labels": [label for i, label in enumerate(config.data.labels) if labels[rank, i] > 0.5],
                        "selector_weight": selector_weight,
                        "selected": selector_weight is None or selector_weight > 0,
                        "attention_weight": float(attention[rank]) if attention is not None else None,
                    })
            reports.append(report)
        cursor += len(batch["sample_ids"])
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", action="append")
    group.add_argument("--input-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comments = args.text if args.text else [line.strip() for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    reports = predict(comments, args.checkpoint, args.config, args.experiment)
    serialized = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

