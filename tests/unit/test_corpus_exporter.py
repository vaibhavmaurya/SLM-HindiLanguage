"""Tests for ingestion/corpus_exporter.py — CorpusExporter."""

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from slm_hindi.config.settings import ExportConfig
from slm_hindi.ingestion.corpus_exporter import CorpusExporter
from slm_hindi.schema.corpus_record import CorpusRecord

_HINDI_TEXT = "भारत एक विविधताओं से भरा देश है और यहाँ की संस्कृति बहुत समृद्ध है।"


def _records(n: int = 10) -> list[CorpusRecord]:
    return [
        CorpusRecord(
            record_id=f"r{i:04d}", document_id=f"d{i:04d}",
            source_type="huggingface_dataset", source_name="test",
            raw_text=_HINDI_TEXT, final_text=_HINDI_TEXT,
            char_count=len(_HINDI_TEXT), word_count=len(_HINDI_TEXT.split()),
        )
        for i in range(n)
    ]


def _make_config(**kwargs) -> ExportConfig:
    return ExportConfig(**kwargs)


def test_parquet_files_written(tmp_path: Path):
    config = _make_config()
    exporter = CorpusExporter(config, data_root=tmp_path)
    splits = {"train": _records(10), "validation": [], "test": []}
    exporter.export(splits)
    parquet_files = list((tmp_path / "final" / "parquet" / "train").glob("*.parquet"))
    assert len(parquet_files) >= 1


def test_parquet_roundtrip_correct(tmp_path: Path):
    config = _make_config()
    exporter = CorpusExporter(config, data_root=tmp_path)
    splits = {"train": _records(5), "validation": [], "test": []}
    exporter.export(splits)
    parquet_files = list((tmp_path / "final" / "parquet" / "train").glob("*.parquet"))
    df = pd.read_parquet(str(parquet_files[0]))
    assert len(df) == 5
    assert "final_text" in df.columns
    assert all(df["final_text"] == _HINDI_TEXT)


def test_jsonl_gz_files_written(tmp_path: Path):
    config = _make_config()
    exporter = CorpusExporter(config, data_root=tmp_path)
    splits = {"train": _records(5), "validation": [], "test": []}
    exporter.export(splits)
    jsonl_files = list((tmp_path / "final" / "training_jsonl" / "train").glob("*.jsonl.gz"))
    assert len(jsonl_files) >= 1


def test_jsonl_gz_each_line_parseable(tmp_path: Path):
    config = _make_config()
    exporter = CorpusExporter(config, data_root=tmp_path)
    splits = {"train": _records(5), "validation": [], "test": []}
    exporter.export(splits)
    jsonl_files = list((tmp_path / "final" / "training_jsonl" / "train").glob("*.jsonl.gz"))
    with gzip.open(str(jsonl_files[0]), "rt", encoding="utf-8") as fh:
        lines = [json.loads(ln) for ln in fh if ln.strip()]
    assert len(lines) == 5
    assert all("text" in obj for obj in lines)


def test_txt_gz_files_written(tmp_path: Path):
    config = _make_config()
    exporter = CorpusExporter(config, data_root=tmp_path)
    splits = {"train": _records(5), "validation": [], "test": []}
    exporter.export(splits)
    txt_files = list((tmp_path / "final" / "training_text" / "train").glob("*.txt.gz"))
    assert len(txt_files) >= 1


def test_txt_gz_non_empty(tmp_path: Path):
    config = _make_config()
    exporter = CorpusExporter(config, data_root=tmp_path)
    splits = {"train": _records(5), "validation": [], "test": []}
    exporter.export(splits)
    txt_files = list((tmp_path / "final" / "training_text" / "train").glob("*.txt.gz"))
    with gzip.open(str(txt_files[0]), "rt", encoding="utf-8") as fh:
        content = fh.read()
    assert len(content) > 0


def test_file_registry_called(tmp_path: Path):
    from unittest.mock import MagicMock
    config = _make_config()
    exporter = CorpusExporter(config, data_root=tmp_path)
    mock_registry = MagicMock()
    splits = {"train": _records(3), "validation": [], "test": []}
    exporter.export(splits, file_registry=mock_registry)
    assert mock_registry.register_file.called
