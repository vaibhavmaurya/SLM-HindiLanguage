"""Tests for FileRegistry — Phase 3."""

from __future__ import annotations

import csv
import hashlib

from hindi_tokenizer.observability.file_registry import FileRegistry


def test_register_file_creates_csv(tmp_path):
    registry_file = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="test-uuid", registry_file=registry_file)
    data_file = tmp_path / "test.txt"
    data_file.write_text("hello")
    registry.register_file(path=data_file, role="output", stage="test_stage")
    assert registry_file.exists()


def test_register_file_computes_sha256(tmp_path):
    registry_file = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="test-uuid", registry_file=registry_file)
    data_file = tmp_path / "test.bin"
    content = b"hello world"
    data_file.write_bytes(content)
    registry.register_file(path=data_file, role="output", stage="test_stage")
    rows = list(csv.DictReader(registry_file.open(encoding="utf-8")))
    expected_sha256 = hashlib.sha256(content).hexdigest()
    assert rows[0]["sha256"] == expected_sha256


def test_register_file_records_size(tmp_path):
    registry_file = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="test-uuid", registry_file=registry_file)
    data_file = tmp_path / "test.bin"
    content = b"hello world"
    data_file.write_bytes(content)
    registry.register_file(path=data_file, role="output", stage="test_stage")
    rows = list(csv.DictReader(registry_file.open(encoding="utf-8")))
    assert int(rows[0]["size_bytes"]) == len(content)


def test_register_file_appends_multiple(tmp_path):
    registry_file = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="test-uuid", registry_file=registry_file)
    for i in range(3):
        f = tmp_path / f"file_{i}.txt"
        f.write_text(f"content {i}")
        registry.register_file(path=f, role="output", stage="test_stage")
    rows = list(csv.DictReader(registry_file.open(encoding="utf-8")))
    assert len(rows) == 3


def test_register_file_contains_correct_fields(tmp_path):
    registry_file = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="my-run", registry_file=registry_file)
    data_file = tmp_path / "sample.txt"
    data_file.write_text("some text")
    registry.register_file(path=data_file, role="intermediate", stage="corpus_sample", source_id="sangraha")
    rows = list(csv.DictReader(registry_file.open(encoding="utf-8")))
    assert rows[0]["run_id"] == "my-run"
    assert rows[0]["role"] == "intermediate"
    assert rows[0]["stage"] == "corpus_sample"
    assert rows[0]["source_id"] == "sangraha"
    assert rows[0]["file_name"] == "sample.txt"


def test_register_file_appends_across_instances(tmp_path):
    registry_file = tmp_path / "registry.csv"
    f1 = tmp_path / "a.txt"
    f1.write_text("aaa")
    f2 = tmp_path / "b.txt"
    f2.write_text("bbb")
    FileRegistry(run_id="run-1", registry_file=registry_file).register_file(path=f1, role="output", stage="s1")
    FileRegistry(run_id="run-2", registry_file=registry_file).register_file(path=f2, role="output", stage="s2")
    rows = list(csv.DictReader(registry_file.open(encoding="utf-8")))
    assert len(rows) == 2
