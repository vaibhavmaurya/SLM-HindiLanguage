"""Tests for CorpusSampler and its helper functions — Phase 3."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from hindi_tokenizer.corpus.corpus_sampler import (
    CorpusSampler,
    devanagari_ratio,
    is_valid_hindi_text,
    normalize_unicode,
)
from hindi_tokenizer.schema.records import SampleManifest

# ---------------------------------------------------------------------------
# devanagari_ratio helpers
# ---------------------------------------------------------------------------


def test_devanagari_ratio_pure_hindi():
    ratio = devanagari_ratio("भारत")
    assert ratio == pytest.approx(1.0)


def test_devanagari_ratio_pure_ascii():
    ratio = devanagari_ratio("hello world")
    assert ratio == pytest.approx(0.0)


def test_devanagari_ratio_mixed():
    ratio = devanagari_ratio("भारत India")
    assert 0.3 < ratio < 0.7


def test_devanagari_ratio_only_spaces():
    ratio = devanagari_ratio("   ")
    assert ratio == pytest.approx(0.0)


def test_devanagari_ratio_empty_string():
    ratio = devanagari_ratio("")
    assert ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# is_valid_hindi_text
# ---------------------------------------------------------------------------


def test_is_valid_passes_clean_hindi():
    text = "भारत एक महान देश है जहाँ अनेक भाषाएँ बोली जाती हैं।"  # 52 chars, all Devanagari
    assert is_valid_hindi_text(text, min_char_count=30, max_char_count=5000, min_devanagari_ratio=0.60) is True


def test_is_valid_fails_too_short():
    text = "भारत"  # 4 chars
    assert is_valid_hindi_text(text, min_char_count=30, max_char_count=5000, min_devanagari_ratio=0.60) is False


def test_is_valid_fails_too_long():
    text = "क" * 6000
    assert is_valid_hindi_text(text, min_char_count=30, max_char_count=5000, min_devanagari_ratio=0.60) is False


def test_is_valid_fails_low_devanagari():
    text = "a" * 50 + "भ" * 10  # ~17% Devanagari
    assert is_valid_hindi_text(text, min_char_count=30, max_char_count=5000, min_devanagari_ratio=0.60) is False


def test_is_valid_fails_empty_string():
    assert is_valid_hindi_text("", min_char_count=30, max_char_count=5000, min_devanagari_ratio=0.60) is False


# ---------------------------------------------------------------------------
# normalize_unicode
# ---------------------------------------------------------------------------


def test_normalize_unicode_nfkc():
    composed = unicodedata.normalize("NFC", "क")
    decomposed = unicodedata.normalize("NFD", "क")
    assert normalize_unicode(composed) == normalize_unicode(decomposed)


def test_normalize_collapses_whitespace():
    result = normalize_unicode("भारत   है")
    assert result == "भारत है"


def test_normalize_strips_leading_trailing():
    result = normalize_unicode("  भारत  ")
    assert result == "भारत"


# ---------------------------------------------------------------------------
# CorpusSampler.sample()
# ---------------------------------------------------------------------------


def _make_sampler(sample_parquet_dir: Path, **kwargs) -> CorpusSampler:
    defaults = {
        "input_folder": sample_parquet_dir,
        "text_column": "final_text",
        "file_pattern": "*.parquet",
        "min_char_count": 30,
        "max_char_count": 5000,
        "min_devanagari_ratio": 0.60,
        "random_seed": 42,
    }
    defaults.update(kwargs)
    return CorpusSampler(**defaults)


def test_sample_creates_output_file(sample_parquet_dir, tmp_path):
    out = tmp_path / "sample.txt"
    _make_sampler(sample_parquet_dir).sample(target_size_gb=1.0, output_file=out)
    assert out.exists()


def test_sample_one_record_per_line(sample_parquet_dir, tmp_path):
    out = tmp_path / "sample.txt"
    _make_sampler(sample_parquet_dir).sample(target_size_gb=1.0, output_file=out)
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) > 0
    # Every non-empty line is a single record — no embedded newlines (verified by using splitlines)
    assert all("\n" not in ln for ln in lines)


def test_sample_excludes_filtered_records(sample_parquet_dir, tmp_path):
    out = tmp_path / "sample.txt"
    _make_sampler(sample_parquet_dir, min_char_count=30).sample(target_size_gb=1.0, output_file=out)
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert all(len(ln) >= 30 for ln in lines)


def test_sample_excludes_low_devanagari_records(sample_parquet_dir, tmp_path):
    out = tmp_path / "sample.txt"
    _make_sampler(sample_parquet_dir, min_devanagari_ratio=0.60).sample(target_size_gb=1.0, output_file=out)
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert all(devanagari_ratio(ln) >= 0.60 for ln in lines)


def test_sample_is_deterministic(sample_parquet_dir, tmp_path):
    out1 = tmp_path / "out1.txt"
    out2 = tmp_path / "out2.txt"
    _make_sampler(sample_parquet_dir, random_seed=42).sample(target_size_gb=1.0, output_file=out1)
    _make_sampler(sample_parquet_dir, random_seed=42).sample(target_size_gb=1.0, output_file=out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_sample_different_seeds_differ(sample_parquet_dir, tmp_path):
    out1 = tmp_path / "out1.txt"
    out2 = tmp_path / "out2.txt"
    _make_sampler(sample_parquet_dir, random_seed=42).sample(target_size_gb=1.0, output_file=out1)
    _make_sampler(sample_parquet_dir, random_seed=999).sample(target_size_gb=1.0, output_file=out2)
    assert out1.read_bytes() != out2.read_bytes()


def test_sample_respects_target_size(sample_parquet_dir, tmp_path):
    out = tmp_path / "sample.txt"
    target_size_gb = 0.00001  # ≈ 10.7 KB; fixture has ≈ 19.4 KB valid → stops early
    _make_sampler(sample_parquet_dir).sample(target_size_gb=target_size_gb, output_file=out)
    target_bytes = target_size_gb * 1024**3
    file_size = out.stat().st_size
    assert file_size >= target_bytes * 0.9, f"Under-sampled: {file_size} < 90% of {target_bytes:.0f}"
    assert file_size <= target_bytes * 1.1, f"Over-sampled: {file_size} > 110% of {target_bytes:.0f}"


def test_sample_returns_manifest(sample_parquet_dir, tmp_path):
    out = tmp_path / "sample.txt"
    manifest = _make_sampler(sample_parquet_dir).sample(target_size_gb=1.0, output_file=out)
    assert isinstance(manifest, SampleManifest)
    assert manifest.actual_size_gb > 0
    assert manifest.random_seed == 42


def test_sample_manifest_written_records_positive(sample_parquet_dir, tmp_path):
    out = tmp_path / "sample.txt"
    manifest = _make_sampler(sample_parquet_dir).sample(target_size_gb=1.0, output_file=out)
    assert manifest.written_records > 0


def test_sample_handles_no_parquet_files(tmp_path):
    (tmp_path / "input").mkdir()
    sampler = CorpusSampler(input_folder=tmp_path / "input")
    with pytest.raises(FileNotFoundError):
        sampler.sample(target_size_gb=1.0, output_file=tmp_path / "out.txt")


def test_sample_creates_parent_dirs(sample_parquet_dir, tmp_path):
    out = tmp_path / "deep" / "nested" / "sample.txt"
    _make_sampler(sample_parquet_dir).sample(target_size_gb=1.0, output_file=out)
    assert out.exists()


def test_sample_logs_events(sample_parquet_dir, tmp_path, mocker):
    mock_logger = mocker.Mock()
    out = tmp_path / "sample.txt"
    _make_sampler(sample_parquet_dir).sample(target_size_gb=1.0, output_file=out, run_logger=mock_logger)
    calls = mock_logger.log_event.call_args_list
    phases = [c.kwargs.get("phase") for c in calls]
    assert "corpus_sample" in phases
    statuses = [c.kwargs.get("status") for c in calls]
    assert "started" in statuses
    assert "completed" in statuses


def test_sample_registers_output_file(sample_parquet_dir, tmp_path, mocker):
    mock_registry = mocker.Mock()
    out = tmp_path / "sample.txt"
    _make_sampler(sample_parquet_dir).sample(target_size_gb=1.0, output_file=out, file_registry=mock_registry)
    mock_registry.register_file.assert_called()
    call_kwargs = mock_registry.register_file.call_args.kwargs
    assert Path(call_kwargs.get("path", "")) == out or mock_registry.register_file.call_args.args[0] == out
