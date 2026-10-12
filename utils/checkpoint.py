"""Atomic, resume-complete PyTorch checkpoint persistence."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
    early_stopping: Mapping[str, Any] | None = None,
    thresholds: np.ndarray | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer is not None else None, "scheduler_state": scheduler.state_dict() if scheduler is not None else None, "scaler_state": scaler.state_dict() if scaler is not None else None, "epoch": int(epoch), "global_step": int(global_step), "metrics": dict(metrics), "config": dict(config), "early_stopping": dict(early_stopping or {}), "thresholds": thresholds, "rng_state": capture_rng_state()}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: str | torch.device = "cpu",
    restore_rng: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


def update_checkpoint_thresholds(path: str | Path, thresholds: np.ndarray) -> None:
    """Atomically attach validation-fitted thresholds to an existing checkpoint."""
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["thresholds"] = np.asarray(thresholds, dtype=np.float32)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{source.name}.", suffix=".tmp", dir=source.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, source)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
