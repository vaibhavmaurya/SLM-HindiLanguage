"""Tests for Pydantic schema records — Phase 2."""

from __future__ import annotations

import pytest

from hindi_tokenizer.schema.records import (
    ComparisonResult,
    SampleManifest,
    ValidationReport,
    ValidationThresholdValues,
)


def _make_thresholds(**overrides: float | int) -> ValidationThresholdValues:
    defaults = {
        "max_unk_rate": 0.001,
        "min_chars_per_token": 3.0,
        "max_tokens_per_word": 2.5,
        "min_roundtrip_success_rate": 0.99,
        "min_devanagari_coverage": 0.995,
        "max_special_token_split_failures": 0,
    }
    defaults.update(overrides)
    return ValidationThresholdValues(**defaults)


def _make_passing_report(variant_name: str = "32k") -> ValidationReport:
    return ValidationReport(
        variant_name=variant_name,
        vocab_size=32000,
        tokenizer_version="hindi_slm_tokenizer_v001",
        unk_rate=0.0005,
        chars_per_token=4.2,
        tokens_per_word=1.8,
        roundtrip_success_rate=0.999,
        devanagari_char_coverage=0.997,
        special_token_split_failures=0,
        thresholds=_make_thresholds(),
    )


def test_sample_manifest_instantiates() -> None:
    manifest = SampleManifest(
        version="hindi_corpus_v001",
        source_folder="../data_ingestion/data/final/parquet/train",
        text_column="final_text",
        target_size_gb=5.0,
        actual_size_gb=4.98,
        written_records=120000,
        random_seed=42,
    )
    assert manifest.version == "hindi_corpus_v001"
    assert manifest.written_records == 120000
    assert manifest.random_seed == 42


def test_sample_manifest_created_at_populated() -> None:
    manifest = SampleManifest(
        version="hindi_corpus_v001",
        source_folder="data/train",
        text_column="final_text",
        target_size_gb=1.0,
        actual_size_gb=0.99,
        written_records=10,
        random_seed=42,
    )
    assert manifest.created_at != ""
    assert "T" in manifest.created_at  # ISO-8601 contains a T separator


def test_validation_report_instantiates() -> None:
    report = _make_passing_report()
    assert report.vocab_size == 32000
    assert report.tokenizer_version == "hindi_slm_tokenizer_v001"


def test_validation_report_passes_thresholds_true() -> None:
    report = _make_passing_report()
    assert report.passes_thresholds is True


def test_validation_report_passes_thresholds_false() -> None:
    report = ValidationReport(
        variant_name="32k",
        vocab_size=32000,
        tokenizer_version="hindi_slm_tokenizer_v001",
        unk_rate=0.05,  # above 0.001 threshold
        chars_per_token=4.2,
        tokens_per_word=1.8,
        roundtrip_success_rate=0.999,
        devanagari_char_coverage=0.997,
        special_token_split_failures=0,
        thresholds=_make_thresholds(),
    )
    assert report.passes_thresholds is False


def test_validation_report_roundtrip_field() -> None:
    report = _make_passing_report()
    assert 0.0 <= report.roundtrip_success_rate <= 1.0


def test_comparison_result_instantiates() -> None:
    variants = [
        _make_passing_report("24k"),
        _make_passing_report("32k"),
        _make_passing_report("48k"),
    ]
    result = ComparisonResult(variants=variants, recommended_variant="32k")
    assert len(result.variants) == 3


def test_comparison_result_recommended_variant() -> None:
    variants = [
        _make_passing_report("24k"),
        _make_passing_report("32k"),
        _make_passing_report("48k"),
    ]
    result = ComparisonResult(variants=variants, recommended_variant="32k")
    variant_names = [v.variant_name for v in result.variants]
    assert result.recommended_variant in variant_names


def test_records_reject_bare_dict() -> None:
    variants = [
        _make_passing_report("24k"),
        {"variant_name": "32k", "vocab_size": 32000},  # bare dict — not a ValidationReport
        _make_passing_report("48k"),
    ]
    with pytest.raises(TypeError):
        ComparisonResult(variants=variants, recommended_variant="32k")  # type: ignore[arg-type]
