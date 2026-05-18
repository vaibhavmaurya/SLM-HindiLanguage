"""Integration tests: ParquetReader → CorpusSampler pipeline — Phase 10."""

from __future__ import annotations

import pytest

from hindi_tokenizer.corpus.corpus_sampler import CorpusSampler


@pytest.fixture(scope="module")
def sampled_output(tmp_path_factory, sample_parquet_dir):
    out = tmp_path_factory.mktemp("corpus_pipeline")
    output_file = out / "sample.txt"
    sampler = CorpusSampler(
        input_folder=sample_parquet_dir,
        text_column="final_text",
        file_pattern="*.parquet",
        min_char_count=30,
        max_char_count=5000,
        min_devanagari_ratio=0.60,
        random_seed=42,
    )
    manifest = sampler.sample(target_size_gb=1.0, output_file=output_file)
    return output_file, manifest


def test_parquet_reader_to_sampler_end_to_end(sampled_output):
    output_file, manifest = sampled_output
    assert output_file.exists()
    lines = [line for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) > 0


def test_sampler_output_contains_no_filtered_records(sampled_output):
    output_file, _ = sampled_output
    lines = [line for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        assert len(line) >= 30


def test_sampler_output_is_utf8(sampled_output):
    output_file, _ = sampled_output
    content = output_file.read_bytes()
    content.decode("utf-8")


def test_sampler_manifest_matches_actual_file(sampled_output):
    output_file, manifest = sampled_output
    line_count = sum(1 for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip())
    assert manifest.written_records == line_count
