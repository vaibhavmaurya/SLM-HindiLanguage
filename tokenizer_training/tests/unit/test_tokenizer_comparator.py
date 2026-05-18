"""Tests for TokenizerComparator — Phase 5."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hindi_tokenizer.schema.records import ComparisonResult, ValidationReport, ValidationThresholdValues
from hindi_tokenizer.validation.tokenizer_comparator import TokenizerComparator

_VARIANT_NAMES = ["hindi_unigram_24k_v001", "hindi_unigram_32k_v001", "hindi_unigram_48k_v001"]
_VOCAB_SIZES = {
    "hindi_unigram_24k_v001": 24000,
    "hindi_unigram_32k_v001": 32000,
    "hindi_unigram_48k_v001": 48000,
}


def _write_report(
    tmp_path: Path,
    variant_name: str,
    *,
    unk_rate: float = 0.0001,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 1.5,
    roundtrip_success_rate: float = 0.999,
    devanagari_char_coverage: float = 0.999,
) -> Path:
    thresholds = ValidationThresholdValues()
    report = ValidationReport(
        variant_name=variant_name,
        vocab_size=_VOCAB_SIZES[variant_name],
        tokenizer_version="v001",
        unk_rate=unk_rate,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
        roundtrip_success_rate=roundtrip_success_rate,
        devanagari_char_coverage=devanagari_char_coverage,
        special_token_split_failures=0,
        thresholds=thresholds,
    )
    report_dict = report.model_dump()
    report_dict.update({"special_token_failures": [], "pad_token_id": 0, "unk_token_id": 1,
                        "bos_token_id": 2, "eos_token_id": 3})
    path = tmp_path / f"{variant_name}.json"
    path.write_text(json.dumps(report_dict, ensure_ascii=False), encoding="utf-8")
    return path


def _write_failing_report(tmp_path: Path, variant_name: str) -> Path:
    return _write_report(
        tmp_path, variant_name,
        unk_rate=0.5,
        chars_per_token=0.5,
        tokens_per_word=10.0,
        roundtrip_success_rate=0.5,
        devanagari_char_coverage=0.5,
    )


@pytest.fixture()
def report_paths(tmp_path):
    return {name: _write_report(tmp_path, name) for name in _VARIANT_NAMES}


@pytest.fixture()
def output_path(tmp_path):
    return tmp_path / "comparison_report.md"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_comparator_loads_three_reports(report_paths, output_path):
    result = TokenizerComparator(list(report_paths.values())).compare(output_path)
    assert len(result.variants) == 3


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def test_comparator_produces_markdown_file(report_paths, output_path):
    TokenizerComparator(list(report_paths.values())).compare(output_path)
    assert output_path.exists()


def test_comparison_markdown_has_header_row(report_paths, output_path):
    TokenizerComparator(list(report_paths.values())).compare(output_path)
    content = output_path.read_text(encoding="utf-8")
    for col in ["unk_rate", "chars_per_token", "tokens_per_word", "roundtrip_success_rate"]:
        assert col in content, f"Column {col!r} missing from markdown"


def test_comparison_markdown_has_row_per_variant(report_paths, output_path):
    TokenizerComparator(list(report_paths.values())).compare(output_path)
    content = output_path.read_text(encoding="utf-8")
    for name in _VARIANT_NAMES:
        assert name in content, f"Variant {name!r} missing from markdown"


def test_comparison_has_recommendation_section(report_paths, output_path):
    TokenizerComparator(list(report_paths.values())).compare(output_path)
    content = output_path.read_text(encoding="utf-8")
    assert "Recommended" in content


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def test_recommended_variant_is_one_of_three(report_paths, output_path):
    result = TokenizerComparator(list(report_paths.values())).compare(output_path)
    assert result.recommended_variant in _VARIANT_NAMES


def test_recommended_variant_has_lowest_unk_rate(tmp_path, output_path):
    paths = [
        _write_report(tmp_path, "hindi_unigram_24k_v001", unk_rate=0.0005),
        _write_report(tmp_path, "hindi_unigram_32k_v001", unk_rate=0.0003),
        _write_report(tmp_path, "hindi_unigram_48k_v001", unk_rate=0.0001),
    ]
    result = TokenizerComparator(paths).compare(output_path)
    assert result.recommended_variant == "hindi_unigram_48k_v001"


def test_recommended_variant_prefers_32k_on_tie(tmp_path, output_path):
    paths = [
        _write_report(tmp_path, "hindi_unigram_24k_v001", unk_rate=0.0005),
        _write_report(tmp_path, "hindi_unigram_32k_v001", unk_rate=0.0001),
        _write_report(tmp_path, "hindi_unigram_48k_v001", unk_rate=0.0001),
    ]
    result = TokenizerComparator(paths).compare(output_path)
    assert result.recommended_variant == "hindi_unigram_32k_v001"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_comparator_raises_on_missing_report(tmp_path, output_path):
    paths = [
        _write_report(tmp_path, "hindi_unigram_24k_v001"),
        _write_report(tmp_path, "hindi_unigram_32k_v001"),
        tmp_path / "nonexistent.json",
    ]
    with pytest.raises(FileNotFoundError):
        TokenizerComparator(paths).compare(output_path)


def test_comparator_raises_on_all_variants_failing_thresholds(tmp_path, output_path):
    paths = [_write_failing_report(tmp_path, name) for name in _VARIANT_NAMES]
    with pytest.raises(ValueError):
        TokenizerComparator(paths).compare(output_path)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_comparator_returns_comparison_result(report_paths, output_path):
    result = TokenizerComparator(list(report_paths.values())).compare(output_path)
    assert isinstance(result, ComparisonResult)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_comparator_logs_event(report_paths, output_path, mocker):
    mock_logger = mocker.Mock()
    TokenizerComparator(list(report_paths.values())).compare(output_path, run_logger=mock_logger)
    phases = [c.kwargs.get("phase") for c in mock_logger.log_event.call_args_list]
    assert "comparison" in phases


def test_comparator_registers_report_file(report_paths, output_path, mocker):
    mock_registry = mocker.Mock()
    TokenizerComparator(list(report_paths.values())).compare(output_path, file_registry=mock_registry)
    mock_registry.register_file.assert_called()
    registered = [str(c.kwargs.get("path", "")) for c in mock_registry.register_file.call_args_list]
    assert any(str(output_path) in p for p in registered)
