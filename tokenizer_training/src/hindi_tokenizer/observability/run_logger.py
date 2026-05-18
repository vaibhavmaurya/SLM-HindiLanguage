"""Append-only CSV event logger for the tokenizer pipeline."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_FIELDNAMES = [
    "run_id",
    "timestamp",
    "phase",
    "component",
    "source_id",
    "record_id",
    "status",
    "records_in",
    "records_out",
    "records_rejected",
    "duration_seconds",
    "error_message",
    "notes",
]


class TokenizerRunLogger:
    def __init__(self, run_id: str, log_file: str | Path) -> None:
        self.run_id = run_id
        self.log_file = Path(log_file)

    def log_event(
        self,
        *,
        phase: str,
        component: str = "",
        source_id: str = "",
        record_id: str = "",
        status: str = "completed",
        records_in: int = -1,
        records_out: int = -1,
        records_rejected: int = -1,
        duration_seconds: float = 0.0,
        error_message: str = "",
        notes: str = "",
    ) -> None:
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "phase": phase,
            "component": component,
            "source_id": source_id,
            "record_id": record_id,
            "status": status,
            "records_in": records_in,
            "records_out": records_out,
            "records_rejected": records_rejected,
            "duration_seconds": duration_seconds,
            "error_message": error_message,
            "notes": notes,
        }
        write_header = not self.log_file.exists() or self.log_file.stat().st_size == 0
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
