"""Configuration loading, recursive overrides, and fail-fast validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration cannot produce a valid experiment."""


class Config(dict[str, Any]):
    """Dictionary configuration with convenient nested attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def clone(self) -> "Config":
        return Config(copy.deepcopy(self))


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {source}")
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate cross-component invariants before expensive work begins."""
    required = {"project", "data", "model", "retrieval", "training", "thresholds"}
    missing = required.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")
    labels = config["data"].get("labels", [])
    if len(labels) != config["model"].get("num_labels"):
        raise ConfigError("data.labels and model.num_labels disagree")
    if len(set(labels)) != len(labels):
        raise ConfigError("Label names must be unique")
    validation_size = float(config["data"].get("validation_size", 0))
    test_size = float(config["data"].get("test_size", 0))
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ConfigError("Validation and test sizes must be positive and sum to less than one")
    if int(config["retrieval"].get("selected_k", 0)) > int(config["retrieval"].get("top_k", 0)):
        raise ConfigError("retrieval.selected_k cannot exceed retrieval.top_k")
    if int(config["training"].get("gradient_accumulation_steps", 0)) < 1:
        raise ConfigError("gradient_accumulation_steps must be at least one")


def load_config(
    base_path: str | Path = "configs/config.yaml",
    override_path: str | Path | None = None,
) -> Config:
    """Load the canonical config and optionally merge an experiment override."""
    data = _read_yaml(base_path)
    if override_path:
        _deep_merge(data, _read_yaml(override_path))
    validate_config(data)
    return Config(data)


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)

