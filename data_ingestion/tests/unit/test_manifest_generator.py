"""Tests for ingestion/manifest_generator.py — ManifestGenerator."""

import gzip
import json
from pathlib import Path

import pytest

from slm_hindi.config.settings import ExportConfig
from slm_hindi.ingestion.corpus_exporter import CorpusExporter
from slm_hindi.ingestion.manifest_generator import ManifestGenerator, _sha256_file
from slm_hindi.schema.corpus_record import CorpusRecord

_HINDI_TEXT = "भारत एक विविधताओं से भरा देश है और यहाँ की संस्कृति बहुत समृद्ध है।"


def _records(n: int = 5) -> list[CorpusRecord]:
    return [
        CorpusRecord(
            record_id=f"r{i:04d}", document_id=f"d{i:04d}",
            source_type="huggingface_dataset", source_name="test",
            raw_text=_HINDI_TEXT, final_text=_HINDI_TEXT,
            char_count=len(_HINDI_TEXT), word_count=len(_HINDI_TEXT.split()),
            estimated_token_count=15,
        )
        for i in range(n)
    ]


def _setup_corpus(tmp_path: Path) -> tuple[Path, dict[str, list[CorpusRecord]]]:
    config = ExportConfig()
    exporter = CorpusExporter(config, data_root=tmp_path)
    splits = {"train": _records(5), "validation": _records(1), "test": _records(1)}
    exporter.export(splits)
    return tmp_path, splits


def test_manifest_file_written(tmp_path: Path):
    data_root, splits = _setup_corpus(tmp_path)
    gen = ManifestGenerator(ExportConfig(), data_root=data_root)
    gen.generate(splits)
    manifest_path = data_root / "reports" / "hindi_corpus_v001_manifest.json"
    assert manifest_path.exists()


def test_manifest_sha256_matches_actual_file(tmp_path: Path):
    data_root, splits = _setup_corpus(tmp_path)
    gen = ManifestGenerator(ExportConfig(), data_root=data_root)
    manifest = gen.generate(splits)
    for entry in manifest["files"]:
        file_path = data_root / entry["path"]
        actual_sha = _sha256_file(file_path)
        assert entry["sha256"] == actual_sha, f"SHA-256 mismatch for {entry['path']}"


def test_manifest_corpus_version_matches_config(tmp_path: Path):
    data_root, splits = _setup_corpus(tmp_path)
    gen = ManifestGenerator(ExportConfig(), data_root=data_root)
    manifest = gen.generate(splits)
    assert manifest["corpus_version"] == "hindi_corpus_v001"


def test_profile_file_written(tmp_path: Path):
    data_root, splits = _setup_corpus(tmp_path)
    gen = ManifestGenerator(ExportConfig(), data_root=data_root)
    gen.generate(splits)
    profile_path = data_root / "reports" / "hindi_corpus_v001_profile.json"
    assert profile_path.exists()


def test_profile_totals_correct(tmp_path: Path):
    data_root, splits = _setup_corpus(tmp_path)
    gen = ManifestGenerator(ExportConfig(), data_root=data_root)
    gen.generate(splits)
    profile_path = data_root / "reports" / "hindi_corpus_v001_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["splits"]["train"]["record_count"] == 5
    assert profile["splits"]["validation"]["record_count"] == 1
    assert profile["splits"]["test"]["record_count"] == 1
