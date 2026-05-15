"""Tests for ingestion/pdf_registry.py — PdfRegistry."""

import json
import shutil
from pathlib import Path

import pytest

from slm_hindi.ingestion.pdf_registry import PdfRegistry, PdfSource


def _make_valid_source(parent: Path, source_id: str, sample_pdf: Path) -> Path:
    src_dir = parent / source_id
    src_dir.mkdir(parents=True)
    shutil.copy(sample_pdf, src_dir / "original.pdf")
    metadata = {
        "source_id": source_id,
        "source_name": "Test Source",
        "file_name": "original.pdf",
    }
    (src_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return parent


def test_discovers_valid_source(tmp_path: Path, sample_pdf_path: Path):
    root = _make_valid_source(tmp_path, "pdf_001", sample_pdf_path)
    registry = PdfRegistry(root)
    sources = registry.discover()
    assert len(sources) == 1
    assert isinstance(sources[0], PdfSource)
    assert sources[0].source_id == "pdf_001"


def test_missing_original_pdf_raises(tmp_path: Path):
    src_dir = tmp_path / "pdf_no_pdf"
    src_dir.mkdir()
    (src_dir / "metadata.json").write_text('{"source_id":"x","source_name":"x","file_name":"x"}', encoding="utf-8")
    registry = PdfRegistry(tmp_path)
    with pytest.raises(FileNotFoundError, match="original.pdf"):
        registry.discover()


def test_missing_metadata_raises_when_required(tmp_path: Path, sample_pdf_path: Path):
    src_dir = tmp_path / "pdf_no_meta"
    src_dir.mkdir()
    shutil.copy(sample_pdf_path, src_dir / "original.pdf")
    registry = PdfRegistry(tmp_path, require_metadata=True)
    with pytest.raises(ValueError, match="metadata.json"):
        registry.discover()


def test_missing_metadata_allowed_when_not_required(tmp_path: Path, sample_pdf_path: Path):
    src_dir = tmp_path / "pdf_no_meta"
    src_dir.mkdir()
    shutil.copy(sample_pdf_path, src_dir / "original.pdf")
    registry = PdfRegistry(tmp_path, require_metadata=False)
    sources = registry.discover()
    assert len(sources) == 1


def test_invalid_metadata_schema_raises(tmp_path: Path, sample_pdf_path: Path):
    src_dir = tmp_path / "pdf_bad_meta"
    src_dir.mkdir()
    shutil.copy(sample_pdf_path, src_dir / "original.pdf")
    (src_dir / "metadata.json").write_text('{"invalid_field": true}', encoding="utf-8")
    registry = PdfRegistry(tmp_path)
    with pytest.raises((ValueError, Exception)):
        registry.discover()


def test_empty_input_dir_returns_empty(tmp_path: Path):
    registry = PdfRegistry(tmp_path)
    assert registry.discover() == []


def test_nonexistent_input_dir_returns_empty(tmp_path: Path):
    registry = PdfRegistry(tmp_path / "does_not_exist")
    assert registry.discover() == []
