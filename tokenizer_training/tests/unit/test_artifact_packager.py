"""Tests for ArtifactPackager — Phase 7."""

from __future__ import annotations

import json

import pytest

from hindi_tokenizer.packaging.artifact_packager import ArtifactPackager

_VERSION = "hindi_slm_tokenizer_v001"
_TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer_metadata.json"]


@pytest.fixture()
def source_artifact_dir(tmp_path):
    d = tmp_path / "source_artifact"
    d.mkdir()
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        (d / f).write_text(f'{{"file": "{f}"}}', encoding="utf-8")
    (d / "tokenizer_metadata.json").write_text(
        json.dumps({"algorithm": "unigram", "vocab_size": 32000, "normalizer": "NFKC", "pre_tokenizer": "Metaspace"}),
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def validation_report_path(tmp_path):
    p = tmp_path / "validation_report.json"
    p.write_text('{"unk_rate": 0.0001}', encoding="utf-8")
    return p


@pytest.fixture()
def comparison_report_path(tmp_path):
    p = tmp_path / "comparison_report.md"
    p.write_text("# Comparison\nRecommended: v001", encoding="utf-8")
    return p


@pytest.fixture()
def training_config_path(tmp_path):
    p = tmp_path / "tokenizer_training_config.yaml"
    p.write_text("project:\n  name: hindi-slm-tokenizer\n", encoding="utf-8")
    return p


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path / "packaged_artifact"


@pytest.fixture()
def packager(source_artifact_dir, output_dir, validation_report_path, comparison_report_path, training_config_path):
    return ArtifactPackager(
        artifact_dir=source_artifact_dir,
        output_dir=output_dir,
        validation_report_path=validation_report_path,
        comparison_report_path=comparison_report_path,
        training_config_path=training_config_path,
        tokenizer_version=_VERSION,
    )


# ---------------------------------------------------------------------------
# File copying
# ---------------------------------------------------------------------------


def test_packager_copies_tokenizer_json(packager, output_dir):
    packager.package()
    assert (output_dir / "tokenizer.json").exists()


def test_packager_copies_tokenizer_config_json(packager, output_dir):
    packager.package()
    assert (output_dir / "tokenizer_config.json").exists()


def test_packager_copies_special_tokens_map_json(packager, output_dir):
    packager.package()
    assert (output_dir / "special_tokens_map.json").exists()


def test_packager_copies_tokenizer_metadata_json(packager, output_dir):
    packager.package()
    assert (output_dir / "tokenizer_metadata.json").exists()


def test_packager_copies_training_config(packager, output_dir):
    packager.package()
    assert (output_dir / "tokenizer_training_config.yaml").exists()


def test_packager_copies_validation_report(packager, output_dir):
    packager.package()
    assert (output_dir / "tokenizer_validation_report.json").exists()


def test_packager_copies_comparison_report(packager, output_dir):
    packager.package()
    assert (output_dir / "tokenizer_comparison_report.md").exists()


# ---------------------------------------------------------------------------
# VERSION file
# ---------------------------------------------------------------------------


def test_packager_creates_version_file(packager, output_dir):
    packager.package()
    assert (output_dir / "VERSION").exists()


def test_packager_version_file_content(packager, output_dir):
    packager.package()
    content = (output_dir / "VERSION").read_text(encoding="utf-8").strip()
    assert content == _VERSION


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def test_packager_creates_readme(packager, output_dir):
    packager.package()
    assert (output_dir / "README.md").exists()


def test_packager_readme_contains_algorithm(packager, output_dir):
    packager.package()
    content = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Unigram" in content


def test_packager_readme_contains_vocab_size(packager, output_dir):
    packager.package()
    content = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "32000" in content


# ---------------------------------------------------------------------------
# Directory & error handling
# ---------------------------------------------------------------------------


def test_packager_creates_output_dir(packager, output_dir):
    assert not output_dir.exists()
    packager.package()
    assert output_dir.exists()


def test_packager_raises_on_missing_source_artifact(
    tmp_path, output_dir, validation_report_path, comparison_report_path, training_config_path
):
    p = ArtifactPackager(
        artifact_dir=tmp_path / "nonexistent_dir",
        output_dir=output_dir,
        validation_report_path=validation_report_path,
        comparison_report_path=comparison_report_path,
        training_config_path=training_config_path,
        tokenizer_version=_VERSION,
    )
    with pytest.raises(FileNotFoundError):
        p.package()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_packager_logs_event(packager, mocker):
    mock_logger = mocker.Mock()
    packager.package(run_logger=mock_logger)
    phases = [c.kwargs.get("phase") for c in mock_logger.log_event.call_args_list]
    assert "packaging" in phases


def test_packager_registers_all_files(packager, output_dir, mocker):
    mock_registry = mocker.Mock()
    packager.package(file_registry=mock_registry)
    assert mock_registry.register_file.call_count >= len(_TOKENIZER_FILES)
