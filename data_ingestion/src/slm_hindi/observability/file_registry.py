"""Append-only CSV registry tracking every file produced or consumed by the pipeline."""

from __future__ import annotations

import csv
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

RoleType = Literal["input", "intermediate", "output", "report"]

_COLUMNS = [
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


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FileRegistry:
    """Thread-safe CSV registry that appends one row per file registered."""

    def __init__(self, run_id: str, registry_path: str | Path) -> None:
        self.run_id = run_id
        self.registry_path = Path(registry_path)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            with open(self.registry_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writeheader()

    def register_file(
        self,
        file_path: str | Path,
        role: RoleType,
        stage: str,
        source_id: str = "",
        file_format: str = "",
        row_count: int = -1,
        compression: str = "none",
        notes: str = "",
    ) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot register non-existent file: {file_path}")

        sha = compute_sha256(file_path)
        size = file_path.stat().st_size

        row = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "stage": stage,
            "source_id": source_id,
            "file_path": str(file_path.resolve()),
            "file_name": file_path.name,
            "format": file_format or file_path.suffix.lstrip("."),
            "size_bytes": size,
            "row_count": row_count,
            "sha256": sha,
            "compression": compression,
            "notes": notes,
        }
        with self._lock:
            with open(self.registry_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writerow(row)
                fh.flush()
