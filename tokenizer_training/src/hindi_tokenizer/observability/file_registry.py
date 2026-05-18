"""Append-only CSV registry of every file produced or consumed by the pipeline."""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_FIELDNAMES = [
    "run_id",
    "timestamp",
    "role",
    "stage",
    "source_id",
    "file_path",
    "file_name",
    "format",
    "size_bytes",
    "row_count",
    "sha256",
    "compression",
    "notes",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FileRegistry:
    def __init__(self, run_id: str, registry_file: str | Path) -> None:
        self.run_id = run_id
        self.registry_file = Path(registry_file)

    def register_file(
        self,
        *,
        path: str | Path,
        role: str,
        stage: str,
        source_id: str = "",
        file_format: str = "",
        row_count: int = -1,
        compression: str = "none",
        notes: str = "",
    ) -> None:
        file_path = Path(path)
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "role": role,
            "stage": stage,
            "source_id": source_id,
            "file_path": str(file_path.resolve()),
            "file_name": file_path.name,
            "format": file_format or file_path.suffix.lstrip("."),
            "size_bytes": file_path.stat().st_size,
            "row_count": row_count,
            "sha256": _sha256(file_path),
            "compression": compression,
            "notes": notes,
        }
        write_header = not self.registry_file.exists() or self.registry_file.stat().st_size == 0
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
