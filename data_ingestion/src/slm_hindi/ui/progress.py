"""Rich-based logging setup and progress bar utilities for the ingestion CLI."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

# Ensure stdout is UTF-8 on Windows (CP1252 default can't render Rich box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

console = Console(legacy_windows=False)

_LOGGING_CONFIGURED = False


def setup_logging(log_level: str = "INFO") -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
            )
        ],
    )
    _LOGGING_CONFIGURED = True


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def pipeline_progress(description: str, total: int) -> Generator[tuple[Progress, TaskID], None, None]:
    """Context manager yielding (progress, task_id) for a single pipeline stage."""
    with make_progress() as progress:
        task = progress.add_task(description, total=total)
        yield progress, task
