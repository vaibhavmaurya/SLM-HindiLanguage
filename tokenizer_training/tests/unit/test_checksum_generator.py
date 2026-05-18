"""Tests for ChecksumGenerator — Phase 7."""

from __future__ import annotations

import json

import pytest

from hindi_tokenizer.packaging.checksum_generator import ChecksumGenerator


@pytest.fixture()
def artifact_dir(tmp_path):
    d = tmp_path / "artifact"
    d.mkdir()
    (d / "tokenizer.json").write_text('{"model": "unigram"}', encoding="utf-8")
    (d / "tokenizer_config.json").write_text('{"vocab_size": 32000}', encoding="utf-8")
    (d / "VERSION").write_text("hindi_slm_tokenizer_v001", encoding="utf-8")
    return d


def test_checksums_json_created(artifact_dir):
    ChecksumGenerator().generate(artifact_dir)
    assert (artifact_dir / "checksums.json").exists()


def test_checksums_covers_all_files(artifact_dir):
    ChecksumGenerator().generate(artifact_dir)
    data = json.loads((artifact_dir / "checksums.json").read_text(encoding="utf-8"))
    expected = {p.name for p in artifact_dir.iterdir() if p.is_file() and p.name != "checksums.json"}
    assert expected <= set(data.keys())


def test_checksum_values_are_sha256(artifact_dir):
    ChecksumGenerator().generate(artifact_dir)
    data = json.loads((artifact_dir / "checksums.json").read_text(encoding="utf-8"))
    for val in data.values():
        assert len(val) == 64
        assert all(c in "0123456789abcdef" for c in val)


def test_checksum_is_reproducible(artifact_dir):
    gen = ChecksumGenerator()
    gen.generate(artifact_dir)
    first = (artifact_dir / "checksums.json").read_text(encoding="utf-8")
    (artifact_dir / "checksums.json").unlink()
    gen.generate(artifact_dir)
    second = (artifact_dir / "checksums.json").read_text(encoding="utf-8")
    assert first == second


def test_checksum_changes_on_file_modification(artifact_dir):
    gen = ChecksumGenerator()
    gen.generate(artifact_dir)
    data_before = json.loads((artifact_dir / "checksums.json").read_text(encoding="utf-8"))
    (artifact_dir / "tokenizer.json").write_text('{"model": "bpe"}', encoding="utf-8")
    (artifact_dir / "checksums.json").unlink()
    gen.generate(artifact_dir)
    data_after = json.loads((artifact_dir / "checksums.json").read_text(encoding="utf-8"))
    assert data_before["tokenizer.json"] != data_after["tokenizer.json"]


def test_checksum_is_sha256_not_md5(artifact_dir):
    ChecksumGenerator().generate(artifact_dir)
    data = json.loads((artifact_dir / "checksums.json").read_text(encoding="utf-8"))
    for val in data.values():
        assert len(val) == 64


def test_checksum_handles_subdirectories(artifact_dir):
    sub = artifact_dir / "subdir"
    sub.mkdir()
    (sub / "extra.json").write_text('{"key": "value"}', encoding="utf-8")
    ChecksumGenerator().generate(artifact_dir)
    data = json.loads((artifact_dir / "checksums.json").read_text(encoding="utf-8"))
    assert any("extra.json" in k for k in data.keys())


def test_checksum_excludes_itself(artifact_dir):
    ChecksumGenerator().generate(artifact_dir)
    data = json.loads((artifact_dir / "checksums.json").read_text(encoding="utf-8"))
    assert "checksums.json" not in data
