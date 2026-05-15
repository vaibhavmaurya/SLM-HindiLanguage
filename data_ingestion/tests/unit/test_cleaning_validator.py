"""Tests for ingestion/cleaning_validator.py — CleaningValidator (7 checks)."""

import pytest

from slm_hindi.config.settings import CleaningValidationConfig
from slm_hindi.ingestion.cleaning_validator import CleaningValidator
from slm_hindi.schema.corpus_record import CorpusRecord

_VALID_HINDI = "यह एक साफ हिंदी वाक्य है जिसमें पर्याप्त देवनागरी अक्षर हैं।"
_SHORT_HINDI = "हिंदी"


def _make_config(**kwargs) -> CleaningValidationConfig:
    defaults = {
        "min_output_char_count": 30,
        "min_output_to_input_length_ratio": 0.50,
        "max_output_to_input_length_ratio": 1.20,
        "min_devanagari_ratio": 0.60,
        "reject_if_output_empty": True,
        "reject_if_prompt_echo": True,
        "reject_if_repeated_lines": True,
    }
    defaults.update(kwargs)
    return CleaningValidationConfig(**defaults)


def _make_record(raw: str, cleaned: str) -> CorpusRecord:
    return CorpusRecord(
        record_id="r1", document_id="d1",
        source_type="pdf", source_name="test",
        raw_text=raw, cleaned_text=cleaned,
    )


def test_valid_hindi_passes():
    validator = CleaningValidator(_make_config())
    record = _make_record(_VALID_HINDI, _VALID_HINDI)
    _, passed = validator.validate(record)
    assert passed
    assert record.cleaning_status == "clean"


def test_empty_output_rejected():
    validator = CleaningValidator(_make_config())
    record = _make_record(_VALID_HINDI, "")
    _, passed = validator.validate(record)
    assert not passed
    assert record.cleaning_status == "quarantined"


def test_output_too_short_rejected():
    validator = CleaningValidator(_make_config())
    # 5 chars < min_output_char_count=30
    record = _make_record(_VALID_HINDI, "हिंदी")
    _, passed = validator.validate(record)
    assert not passed


def test_output_too_compressed_rejected():
    validator = CleaningValidator(_make_config())
    raw = _VALID_HINDI * 4  # ~240 chars
    cleaned = _VALID_HINDI  # ~60 chars → ratio ~0.25 < 0.50
    record = _make_record(raw, cleaned)
    _, passed = validator.validate(record)
    assert not passed


def test_output_too_expanded_rejected():
    validator = CleaningValidator(_make_config())
    raw = "हिंदी पाठ।" * 5  # ~50 chars
    cleaned = _VALID_HINDI * 10  # ~600 chars → ratio ~12.0 > 1.20
    record = _make_record(raw, cleaned)
    _, passed = validator.validate(record)
    assert not passed


def test_low_devanagari_ratio_rejected():
    validator = CleaningValidator(_make_config())
    raw = _VALID_HINDI
    cleaned = "This is entirely in English and has no Devanagari characters at all."
    record = _make_record(raw, cleaned)
    _, passed = validator.validate(record)
    assert not passed


def test_prompt_echo_rejected():
    validator = CleaningValidator(_make_config())
    raw = _VALID_HINDI
    cleaned = "Cleaned Hindi text: " + _VALID_HINDI  # echoes prompt marker
    record = _make_record(raw, cleaned)
    _, passed = validator.validate(record)
    assert not passed


def test_repeated_lines_rejected():
    validator = CleaningValidator(_make_config())
    raw = _VALID_HINDI
    repeated = "यह एक पंक्ति है।\n" * 4  # 4 identical lines
    record = _make_record(raw, repeated)
    _, passed = validator.validate(record)
    assert not passed


def test_all_english_rejected():
    validator = CleaningValidator(_make_config())
    raw = _VALID_HINDI
    cleaned = "This text is entirely in English. " * 3
    record = _make_record(raw, cleaned)
    _, passed = validator.validate(record)
    assert not passed


def test_validate_batch_splits_passed_and_quarantined():
    validator = CleaningValidator(_make_config())
    records = [
        _make_record(_VALID_HINDI, _VALID_HINDI),  # passes
        _make_record(_VALID_HINDI, ""),              # fails (empty)
        _make_record(_VALID_HINDI, _VALID_HINDI),  # passes
    ]
    passed, quarantined = validator.validate_batch(records)
    assert len(passed) == 2
    assert len(quarantined) == 1
