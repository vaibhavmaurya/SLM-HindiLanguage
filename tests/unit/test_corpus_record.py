"""Tests for schema/corpus_record.py — CorpusRecord pydantic model."""

import pytest
from pydantic import ValidationError

from slm_hindi.schema.corpus_record import CorpusRecord


def _minimal_record(**overrides) -> dict:
    base = {
        "record_id": "doc_001_p0001",
        "document_id": "doc_001",
        "source_type": "huggingface_dataset",
        "source_name": "ai4bharat/sangraha",
        "raw_text": "यह हिंदी पाठ का उदाहरण है।",
    }
    base.update(overrides)
    return base


def test_valid_record_creates_without_error():
    record = CorpusRecord(**_minimal_record())
    assert record.record_id == "doc_001_p0001"
    assert record.language == "hi"
    assert record.script == "Devanagari"


def test_defaults_populated():
    record = CorpusRecord(**_minimal_record())
    assert record.cleaning_status == "pending"
    assert record.split_name is None
    assert record.dedup_hash == ""
    assert record.created_at != ""


def test_invalid_source_type_raises():
    with pytest.raises(ValidationError):
        CorpusRecord(**_minimal_record(source_type="unknown_source"))


def test_negative_char_count_raises():
    with pytest.raises(ValidationError):
        CorpusRecord(**_minimal_record(char_count=-1))


def test_ratio_out_of_range_raises():
    with pytest.raises(ValidationError):
        CorpusRecord(**_minimal_record(devanagari_ratio=1.5))
    with pytest.raises(ValidationError):
        CorpusRecord(**_minimal_record(devanagari_ratio=-0.1))


def test_invalid_cleaning_status_raises():
    with pytest.raises(ValidationError):
        CorpusRecord(**_minimal_record(cleaning_status="approved"))


def test_pdf_source_type_accepted():
    record = CorpusRecord(**_minimal_record(source_type="pdf", source_name="user_provided_pdfs"))
    assert record.source_type == "pdf"


def test_model_dump_roundtrip():
    record = CorpusRecord(**_minimal_record(char_count=30, word_count=5))
    data = record.model_dump()
    restored = CorpusRecord(**data)
    assert restored.record_id == record.record_id
    assert restored.char_count == 30
