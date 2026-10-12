"""Structured experiment history for paper table generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


def append_result(path: str | Path, row: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    fields = list(row.keys())
    if destination.exists():
        with destination.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
            fields = list(dict.fromkeys([*(reader.fieldnames or []), *fields]))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(dict(row))

