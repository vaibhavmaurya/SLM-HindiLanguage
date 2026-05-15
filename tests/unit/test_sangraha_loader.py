"""Tests for ingestion/sangraha_loader.py — SangrahaLoader."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_hindi.config.settings import SangrahaSourceConfig
from slm_hindi.ingestion.sangraha_loader import SangrahaLoader
from slm_hindi.schema.corpus_record import CorpusRecord

SAMPLE_ROWS_PATH = Path(__file__).parents[1] / "fixtures" / "sample_sangraha" / "sample_rows.jsonl"


def _load_sample_rows() -> list[dict]:
    rows = []
    with open(SAMPLE_ROWS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_config(**kwargs) -> SangrahaSourceConfig:
    defaults = {"enabled": True, "dataset_name": "ai4bharat/sangraha", "data_dir": "verified/hin", "split": "train", "max_records": 5}
    defaults.update(kwargs)
    return SangrahaSourceConfig(**defaults)


def test_loads_five_records():
    sample_rows = _load_sample_rows()
    config = _make_config(max_records=5)
    loader = SangrahaLoader(config, run_id="test-001")
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    assert len(records) == 5


def test_all_records_are_corpus_records():
    sample_rows = _load_sample_rows()
    config = _make_config()
    loader = SangrahaLoader(config)
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    assert all(isinstance(r, CorpusRecord) for r in records)


def test_source_type_is_huggingface_dataset():
    sample_rows = _load_sample_rows()
    loader = SangrahaLoader(_make_config())
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    assert all(r.source_type == "huggingface_dataset" for r in records)


def test_cleaning_method_is_deterministic_normalization():
    sample_rows = _load_sample_rows()
    loader = SangrahaLoader(_make_config())
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    assert all(r.cleaning_method == "deterministic_normalization" for r in records)


def test_cleaning_model_is_none():
    sample_rows = _load_sample_rows()
    loader = SangrahaLoader(_make_config())
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    assert all(r.cleaning_model is None for r in records)


def test_text_populated_from_row():
    sample_rows = _load_sample_rows()
    loader = SangrahaLoader(_make_config())
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    first_text = sample_rows[0]["text"]
    assert records[0].raw_text == first_text
    assert records[0].final_text == first_text


def test_run_logger_receives_events(run_logger):
    import csv
    sample_rows = _load_sample_rows()
    loader = SangrahaLoader(_make_config(), run_id="test-001")
    with patch("datasets.load_dataset", return_value=sample_rows):
        loader.load(run_logger=run_logger)
    with open(run_logger.log_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    statuses = [r["status"] for r in rows]
    assert "started" in statuses
    assert "completed" in statuses


def test_max_records_limits_output():
    sample_rows = _load_sample_rows() * 10  # 50 rows
    loader = SangrahaLoader(_make_config(max_records=3))
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    assert len(records) == 3
