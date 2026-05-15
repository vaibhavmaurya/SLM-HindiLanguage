"""Tests for observability/run_logger.py — IngestionRunLogger."""

import csv
from pathlib import Path

import pytest

from slm_hindi.observability.run_logger import IngestionRunLogger


def test_log_file_created_on_init(tmp_path: Path):
    log_path = tmp_path / "run_log.csv"
    _ = IngestionRunLogger(run_id="run-001", log_path=log_path)
    assert log_path.exists()


def test_header_row_written(tmp_path: Path):
    log_path = tmp_path / "run_log.csv"
    IngestionRunLogger(run_id="run-001", log_path=log_path)
    with open(log_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert "run_id" in (reader.fieldnames or [])
        assert "phase" in (reader.fieldnames or [])
        assert "status" in (reader.fieldnames or [])


def test_log_event_appends_one_row(tmp_path: Path):
    log_path = tmp_path / "run_log.csv"
    logger = IngestionRunLogger(run_id="run-001", log_path=log_path)
    logger.log_event(phase="test_phase", component="test_comp", status="started")
    with open(log_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["phase"] == "test_phase"
    assert rows[0]["status"] == "started"
    assert rows[0]["run_id"] == "run-001"


def test_multiple_events_append_multiple_rows(tmp_path: Path):
    log_path = tmp_path / "run_log.csv"
    logger = IngestionRunLogger(run_id="run-001", log_path=log_path)
    logger.log_event(phase="phase_a", component="comp", status="started")
    logger.log_event(phase="phase_a", component="comp", status="completed", records_out=100)
    with open(log_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[1]["records_out"] == "100"


def test_phase_timer_context_manager(tmp_path: Path):
    log_path = tmp_path / "run_log.csv"
    logger = IngestionRunLogger(run_id="run-001", log_path=log_path)
    with logger.phase_timer("ingest", "loader", source_id="sangraha"):
        pass
    with open(log_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    statuses = [r["status"] for r in rows]
    assert "started" in statuses
    assert "completed" in statuses


def test_phase_timer_logs_failure_on_exception(tmp_path: Path):
    log_path = tmp_path / "run_log.csv"
    logger = IngestionRunLogger(run_id="run-001", log_path=log_path)
    with pytest.raises(ValueError):
        with logger.phase_timer("fail_phase", "comp"):
            raise ValueError("boom")
    with open(log_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    statuses = [r["status"] for r in rows]
    assert "failed" in statuses
    assert any("boom" in r["error_message"] for r in rows)
