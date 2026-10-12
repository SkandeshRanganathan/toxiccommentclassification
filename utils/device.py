"""Centralized compute-device and automatic mixed-precision policy."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


@dataclass(frozen=True)
class DevicePolicy:
    device: torch.device
    amp_enabled: bool
    amp_dtype: torch.dtype

    def autocast(self) -> ContextManager[object]:
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)


def resolve_device(use_mixed_precision: bool = True) -> DevicePolicy:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enabled = bool(use_mixed_precision and device.type == "cuda")
    # float16 is broadly supported by DeBERTa GPUs; all persisted embeddings are cast to float32.
    return DevicePolicy(device=device, amp_enabled=enabled, amp_dtype=torch.float16)

