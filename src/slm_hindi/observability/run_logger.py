"""Append-only CSV logger for pipeline activity events."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

StatusType = Literal["started", "completed", "skipped", "failed", "quarantined"]

_COLUMNS = [
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


class IngestionRunLogger:
    """Thread-safe CSV logger that appends one row per pipeline event."""

    def __init__(self, run_id: str, log_path: str | Path) -> None:
        self.run_id = run_id
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writeheader()

    def log_event(
        self,
        phase: str,
        component: str,
        status: StatusType,
        source_id: str = "",
        record_id: str = "",
        records_in: int = 0,
        records_out: int = 0,
        records_rejected: int = 0,
        duration_seconds: float = 0.0,
        error_message: str = "",
        notes: str = "",
    ) -> None:
        row = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "component": component,
            "source_id": source_id,
            "record_id": record_id,
            "status": status,
            "records_in": records_in,
            "records_out": records_out,
            "records_rejected": records_rejected,
            "duration_seconds": round(duration_seconds, 4),
            "error_message": error_message,
            "notes": notes,
        }
        with self._lock:
            with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writerow(row)
                fh.flush()

    def phase_timer(self, phase: str, component: str, source_id: str = "") -> _PhaseTimer:
        """Context manager that logs started/completed/failed events automatically."""
        return _PhaseTimer(self, phase, component, source_id)


class _PhaseTimer:
    def __init__(
        self, logger: IngestionRunLogger, phase: str, component: str, source_id: str
    ) -> None:
        self._logger = logger
        self._phase = phase
        self._component = component
        self._source_id = source_id
        self._start: float = 0.0

    def __enter__(self) -> _PhaseTimer:
        self._start = time.monotonic()
        self._logger.log_event(
            phase=self._phase,
            component=self._component,
            status="started",
            source_id=self._source_id,
        )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        duration = time.monotonic() - self._start
        if exc_type is None:
            self._logger.log_event(
                phase=self._phase,
                component=self._component,
                status="completed",
                source_id=self._source_id,
                duration_seconds=duration,
            )
        else:
            self._logger.log_event(
                phase=self._phase,
                component=self._component,
                status="failed",
                source_id=self._source_id,
                duration_seconds=duration,
                error_message=str(exc_val),
            )
