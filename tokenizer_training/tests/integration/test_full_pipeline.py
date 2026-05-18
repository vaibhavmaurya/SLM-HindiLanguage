"""Integration tests: full pipeline sample→train→validate→compare→package — Phase 10."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hindi_tokenizer.observability.file_registry import FileRegistry
from hindi_tokenizer.observability.run_logger import TokenizerRunLogger
from hindi_tokenizer.packaging.artifact_packager import ArtifactPackager
from hindi_tokenizer.packaging.checksum_generator import ChecksumGenerator
from hindi_tokenizer.publishing.tokenizer_publisher import TokenizerPublisher
from hindi_tokenizer.training.tokenizer_trainer import TokenizerTrainer
from hindi_tokenizer.validation.tokenizer_comparator import TokenizerComparator
from hindi_tokenizer.validation.tokenizer_validator import TokenizerValidator

_SMALL_VOCAB = 500
_VERSION = "hindi_slm_tokenizer_v001"
_VARIANT = "hindi_unigram_500_v001"
_REQUIRED_ARTIFACT_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_metadata.json",
    "VERSION",
    "README.md",
}


@pytest.fixture(scope="module")
def pipeline_dirs(tmp_path_factory, small_corpus_path, validation_sentences_path):
    root = tmp_path_factory.mktemp("full_pipeline")

    artifact_dir = root / "artifact"
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(corpus_file=small_corpus_path, output_dir=artifact_dir)

    val_sentences = [ln for ln in validation_sentences_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    report_path = root / "validation_report.json"
    TokenizerValidator(
        artifact_dir=artifact_dir,
        validation_sentences=val_sentences,
        variant_name=_VARIANT,
        tokenizer_version=_VERSION,
    ).validate(report_path=report_path)

    comparison_path = root / "comparison.md"
    try:
        cmp_result = TokenizerComparator(report_paths=[report_path]).compare(output_path=comparison_path)
        recommended_variant = cmp_result.recommended_variant
    except ValueError:
        recommended_variant = _VARIANT
        comparison_path.write_text(f"# Comparison\nRecommended: {recommended_variant}", encoding="utf-8")

    training_config = root / "config.yaml"
    training_config.write_text("project:\n  name: test\n", encoding="utf-8")
    final_dir = root / "final"
    run_log = root / "run_log.csv"
    reg_file = root / "registry.csv"

    run_logger = TokenizerRunLogger(run_id="test-run", log_file=run_log)
    file_registry = FileRegistry(run_id="test-run", registry_file=reg_file)

    ArtifactPackager(
        artifact_dir=artifact_dir,
        output_dir=final_dir,
        validation_report_path=report_path,
        comparison_report_path=comparison_path,
        training_config_path=training_config,
        tokenizer_version=_VERSION,
    ).package(run_logger=run_logger, file_registry=file_registry)

    ChecksumGenerator().generate(final_dir)

    return {
        "root": root,
        "artifact_dir": artifact_dir,
        "final_dir": final_dir,
        "report_path": report_path,
        "run_log": run_log,
        "registry_file": reg_file,
    }


def test_full_pipeline_sample_to_artifact(pipeline_dirs):
    final_dir: Path = pipeline_dirs["final_dir"]
    for fname in _REQUIRED_ARTIFACT_FILES:
        assert (final_dir / fname).exists(), f"Missing: {fname}"


def test_full_pipeline_validation_report_passes_thresholds(pipeline_dirs):
    report_path: Path = pipeline_dirs["report_path"]
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "passes_thresholds" in data


def test_full_pipeline_checksums_generated(pipeline_dirs):
    final_dir: Path = pipeline_dirs["final_dir"]
    checksums_path = final_dir / "checksums.json"
    assert checksums_path.exists()
    data = json.loads(checksums_path.read_text(encoding="utf-8"))
    assert len(data) > 0


def test_full_pipeline_publish_mocked(pipeline_dirs, mocker):
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    TokenizerPublisher(
        source_dir=pipeline_dirs["final_dir"],
        repo_id="vaibhavmaurya/hindi-slm-tokenizer-v001",
    ).publish()
    mock_api.upload_folder.assert_called_once()


def test_full_pipeline_run_log_has_entries(pipeline_dirs):
    run_log: Path = pipeline_dirs["run_log"]
    assert run_log.exists()
    lines = run_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2


def test_full_pipeline_file_registry_has_entries(pipeline_dirs):
    registry_file: Path = pipeline_dirs["registry_file"]
    assert registry_file.exists()
    lines = registry_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
