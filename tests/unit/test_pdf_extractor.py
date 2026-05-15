"""Tests for ingestion/pdf_extractor.py — PdfExtractor."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slm_hindi.config.settings import PdfExtractionConfig
from slm_hindi.ingestion.pdf_extractor import PdfExtractor
from slm_hindi.ingestion.pdf_registry import PdfMetadata, PdfSource
from slm_hindi.schema.corpus_record import CorpusRecord


def _make_config(**kwargs) -> PdfExtractionConfig:
    defaults = {"primary_engine": "pymupdf", "fallback_engine": "pdfplumber", "min_page_text_chars": 10, "max_page_text_chars": 50000}
    defaults.update(kwargs)
    return PdfExtractionConfig(**defaults)


def _make_pdf_source(pdf_path: Path) -> PdfSource:
    return PdfSource(
        source_id="pdf_test_001",
        pdf_path=pdf_path,
        metadata=PdfMetadata(source_id="pdf_test_001", source_name="Test", file_name="sample.pdf"),
    )


def test_extract_returns_records_for_each_page(sample_pdf_path: Path):
    config = _make_config()
    extractor = PdfExtractor(config)
    source = _make_pdf_source(sample_pdf_path)
    records = extractor.extract(source)
    assert len(records) >= 1
    assert all(isinstance(r, CorpusRecord) for r in records)


def test_each_record_has_page_number(sample_pdf_path: Path):
    extractor = PdfExtractor(_make_config())
    source = _make_pdf_source(sample_pdf_path)
    records = extractor.extract(source)
    assert all(r.page_number is not None for r in records)
    assert records[0].page_number == 1


def test_each_record_has_raw_text(sample_pdf_path: Path):
    extractor = PdfExtractor(_make_config())
    source = _make_pdf_source(sample_pdf_path)
    records = extractor.extract(source)
    assert all(len(r.raw_text) >= 10 for r in records)


def test_extraction_method_recorded(sample_pdf_path: Path):
    extractor = PdfExtractor(_make_config())
    source = _make_pdf_source(sample_pdf_path)
    records = extractor.extract(source)
    assert all(r.cleaning_model_version in ("pymupdf", "pdfplumber") for r in records)


def test_falls_back_to_pdfplumber_when_pymupdf_fails(sample_pdf_path: Path):
    config = _make_config()
    extractor = PdfExtractor(config)
    source = _make_pdf_source(sample_pdf_path)

    with patch.object(extractor, "_extract_with_pymupdf", side_effect=Exception("fitz error")):
        records = extractor.extract(source)

    assert len(records) >= 1
    assert all(r.cleaning_model_version == "pdfplumber" for r in records)


def test_source_type_is_pdf(sample_pdf_path: Path):
    extractor = PdfExtractor(_make_config())
    source = _make_pdf_source(sample_pdf_path)
    records = extractor.extract(source)
    assert all(r.source_type == "pdf" for r in records)


def test_run_logger_receives_events(sample_pdf_path: Path, run_logger):
    import csv
    extractor = PdfExtractor(_make_config())
    source = _make_pdf_source(sample_pdf_path)
    extractor.extract(source, run_logger=run_logger)
    with open(run_logger.log_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    statuses = [r["status"] for r in rows]
    assert "started" in statuses
    assert "completed" in statuses
