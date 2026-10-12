"""Serializable metric-aware early stopping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    mode: str = "max"
    best: float | None = None
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        improved = self.best is None or (
            value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta
        )
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, object]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.best = None if state.get("best") is None else float(state["best"])
        self.bad_epochs = int(state.get("bad_epochs", 0))

