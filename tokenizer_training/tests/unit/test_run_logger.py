"""Tests for TokenizerRunLogger — Phase 3."""

from __future__ import annotations

import csv

from hindi_tokenizer.observability.run_logger import TokenizerRunLogger


def test_log_event_creates_csv(tmp_path):
    log_file = tmp_path / "run_log.csv"
    logger = TokenizerRunLogger(run_id="test-uuid", log_file=log_file)
    logger.log_event(phase="test_phase", status="completed")
    assert log_file.exists()


def test_log_event_appends_multiple_rows(tmp_path):
    log_file = tmp_path / "run_log.csv"
    logger = TokenizerRunLogger(run_id="test-uuid", log_file=log_file)
    logger.log_event(phase="phase_a", status="started")
    logger.log_event(phase="phase_a", status="completed")
    rows = list(csv.DictReader(log_file.open(encoding="utf-8")))
    assert len(rows) == 2


def test_log_event_contains_correct_fields(tmp_path):
    log_file = tmp_path / "run_log.csv"
    logger = TokenizerRunLogger(run_id="my-run-id", log_file=log_file)
    logger.log_event(phase="corpus_sample", component="corpus_sampler", status="completed", records_out=100)
    rows = list(csv.DictReader(log_file.open(encoding="utf-8")))
    assert rows[0]["run_id"] == "my-run-id"
    assert rows[0]["phase"] == "corpus_sample"
    assert rows[0]["component"] == "corpus_sampler"
    assert rows[0]["status"] == "completed"
    assert rows[0]["records_out"] == "100"


def test_log_event_run_id_is_consistent(tmp_path):
    log_file = tmp_path / "run_log.csv"
    logger = TokenizerRunLogger(run_id="constant-id", log_file=log_file)
    for i in range(3):
        logger.log_event(phase=f"phase_{i}", status="completed")
    rows = list(csv.DictReader(log_file.open(encoding="utf-8")))
    assert all(r["run_id"] == "constant-id" for r in rows)


def test_log_event_timestamp_is_non_empty(tmp_path):
    log_file = tmp_path / "run_log.csv"
    logger = TokenizerRunLogger(run_id="test-uuid", log_file=log_file)
    logger.log_event(phase="test", status="completed")
    rows = list(csv.DictReader(log_file.open(encoding="utf-8")))
    assert rows[0]["timestamp"] != ""


def test_log_event_appends_across_instances(tmp_path):
    log_file = tmp_path / "run_log.csv"
    TokenizerRunLogger(run_id="run-1", log_file=log_file).log_event(phase="p1", status="completed")
    TokenizerRunLogger(run_id="run-2", log_file=log_file).log_event(phase="p2", status="completed")
    rows = list(csv.DictReader(log_file.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert rows[1]["run_id"] == "run-2"
