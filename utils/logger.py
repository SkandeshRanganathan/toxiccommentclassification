"""Consistent file/console logging and optional TensorBoard integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def configure_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class SummaryWriterAdapter:
    """No-op-safe TensorBoard adapter; TensorBoard remains an optional dependency."""

    def __init__(self, log_dir: str | Path) -> None:
        self.writer: Any | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            logging.getLogger(__name__).warning(
                "TensorBoard is not installed; scalar logging will continue in CSV/files."
            )

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        if self.writer is not None:
            self.writer.add_scalar(*args, **kwargs)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

