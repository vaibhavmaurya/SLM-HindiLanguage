"""Tests for observability/file_registry.py — FileRegistry."""

import csv
from pathlib import Path

import pytest

from slm_hindi.observability.file_registry import FileRegistry, compute_sha256


def _write_test_file(path: Path, content: str = "test content") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_registry_file_created_on_init(tmp_path: Path):
    reg_path = tmp_path / "registry.csv"
    FileRegistry(run_id="r-001", registry_path=reg_path)
    assert reg_path.exists()


def test_header_written(tmp_path: Path):
    reg_path = tmp_path / "registry.csv"
    FileRegistry(run_id="r-001", registry_path=reg_path)
    with open(reg_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert "sha256" in (reader.fieldnames or [])
        assert "file_path" in (reader.fieldnames or [])


def test_register_file_appends_row(tmp_path: Path):
    reg_path = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="r-001", registry_path=reg_path)
    f = _write_test_file(tmp_path / "output.parquet", "fake parquet data")
    registry.register_file(f, role="output", stage="export", file_format="parquet")
    with open(reg_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["file_name"] == "output.parquet"
    assert rows[0]["role"] == "output"


def test_sha256_populated(tmp_path: Path):
    reg_path = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="r-001", registry_path=reg_path)
    content = "हिंदी परीक्षण सामग्री"
    f = _write_test_file(tmp_path / "test.txt", content)
    registry.register_file(f, role="input", stage="load")
    with open(reg_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows[0]["sha256"]) == 64  # SHA-256 hex digest length


def test_register_nonexistent_file_raises(tmp_path: Path):
    reg_path = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="r-001", registry_path=reg_path)
    with pytest.raises(FileNotFoundError):
        registry.register_file(tmp_path / "does_not_exist.txt", role="input", stage="load")


def test_compute_sha256_is_deterministic(tmp_path: Path):
    f = _write_test_file(tmp_path / "deterministic.txt", "constant content")
    h1 = compute_sha256(f)
    h2 = compute_sha256(f)
    assert h1 == h2
    assert len(h1) == 64


def test_multiple_files_append_multiple_rows(tmp_path: Path):
    reg_path = tmp_path / "registry.csv"
    registry = FileRegistry(run_id="r-001", registry_path=reg_path)
    for i in range(3):
        f = _write_test_file(tmp_path / f"file_{i}.txt", f"content {i}")
        registry.register_file(f, role="output", stage="export")
    with open(reg_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
