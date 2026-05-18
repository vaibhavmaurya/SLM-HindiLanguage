"""Tests for ParquetReader — Phase 3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hindi_tokenizer.corpus.parquet_reader import ParquetReader


def test_read_single_file_returns_strings(sample_parquet_dir):
    reader = ParquetReader(sample_parquet_dir, text_column="final_text")
    texts = reader.read_texts(sample_parquet_dir / "sample_01.parquet")
    assert len(texts) > 0
    assert all(isinstance(t, str) for t in texts)


def test_read_single_file_count(sample_parquet_dir):
    reader = ParquetReader(sample_parquet_dir, text_column="final_text")
    texts = reader.read_texts(sample_parquet_dir / "sample_01.parquet")
    assert len(texts) == 50  # sample_01 has 50 rows, all strings (no nulls)


def test_read_multiple_files(sample_parquet_dir):
    reader = ParquetReader(sample_parquet_dir, text_column="final_text")
    texts = reader.read_texts()
    # sample_01: 50 strings; sample_02: 50 rows with 2 nulls → 48 strings
    assert len(texts) == 98


def test_read_uses_configured_text_column(sample_parquet_dir):
    reader_ok = ParquetReader(sample_parquet_dir, text_column="final_text")
    texts = reader_ok.read_texts(sample_parquet_dir / "sample_01.parquet")
    assert len(texts) > 0

    reader_bad = ParquetReader(sample_parquet_dir, text_column="nonexistent_column")
    with pytest.raises(ValueError, match="nonexistent_column"):
        reader_bad.read_texts(sample_parquet_dir / "sample_01.parquet")


def test_read_skips_null_values(sample_parquet_dir):
    reader = ParquetReader(sample_parquet_dir, text_column="final_text")
    texts = reader.read_texts(sample_parquet_dir / "sample_02.parquet")
    # sample_02 has 50 rows with 2 null final_text → 48 texts returned
    assert len(texts) == 48
    assert all(t is not None for t in texts)


def test_read_skips_non_string_values(tmp_path):
    # Pandas object column with float NaN (non-string) alongside valid strings
    df = pd.DataFrame({"final_text": ["valid text हिंदी", np.nan, "another valid text हिंदी भारत"]})
    df.to_parquet(tmp_path / "mixed.parquet")
    reader = ParquetReader(tmp_path, text_column="final_text")
    texts = reader.read_texts(tmp_path / "mixed.parquet")
    assert len(texts) == 2
    assert all(isinstance(t, str) for t in texts)


def test_read_empty_parquet_returns_empty(tmp_path):
    table = pa.table({"final_text": pa.array([], type=pa.string())})
    pq.write_table(table, tmp_path / "empty.parquet")
    reader = ParquetReader(tmp_path, text_column="final_text")
    texts = reader.read_texts(tmp_path / "empty.parquet")
    assert texts == []


def test_read_glob_pattern_finds_files(sample_parquet_dir):
    reader = ParquetReader(sample_parquet_dir, text_column="final_text", file_pattern="*.parquet")
    texts = reader.read_texts()
    assert len(texts) == 98  # both fixture files discovered


def test_read_missing_directory_raises(tmp_path):
    reader = ParquetReader(tmp_path / "nonexistent", text_column="final_text")
    with pytest.raises(FileNotFoundError):
        reader.read_texts()


def test_read_no_parquet_files_raises(tmp_path):
    (tmp_path / "dummy.txt").write_text("not a parquet")
    reader = ParquetReader(tmp_path, text_column="final_text")
    with pytest.raises(FileNotFoundError):
        reader.read_texts()


def test_read_logs_event_to_run_logger(sample_parquet_dir, mocker):
    mock_logger = mocker.Mock()
    reader = ParquetReader(sample_parquet_dir, text_column="final_text", run_logger=mock_logger)
    reader.read_texts()
    mock_logger.log_event.assert_called()
    call_kwargs = mock_logger.log_event.call_args.kwargs
    assert call_kwargs.get("phase") == "parquet_read"
