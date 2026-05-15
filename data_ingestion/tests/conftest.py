"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_pdfs" / "sample_hindi_2page.pdf"
SAMPLE_SANGRAHA = FIXTURES_DIR / "sample_sangraha" / "sample_rows.jsonl"
SAMPLE_METADATA = FIXTURES_DIR / "sample_metadata" / "metadata.json"
SAMPLE_CONFIG = FIXTURES_DIR / "sample_configs" / "test_ingestion_config.yaml"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_ollama: test needs a live Ollama instance")


@pytest.fixture()
def tmp_data_root(tmp_path: Path) -> Path:
    """Temporary data root directory for pipeline output."""
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture()
def sample_pdf_path() -> Path:
    assert SAMPLE_PDF.exists(), f"Sample PDF not found: {SAMPLE_PDF}"
    return SAMPLE_PDF


@pytest.fixture()
def sample_sangraha_rows() -> list[dict]:
    rows = []
    with open(SAMPLE_SANGRAHA, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture()
def sample_metadata() -> dict:
    return json.loads(SAMPLE_METADATA.read_text(encoding="utf-8"))


@pytest.fixture()
def pdf_source_dir(tmp_path: Path, sample_pdf_path: Path, sample_metadata: dict) -> Path:
    """A valid PDF source directory with original.pdf and metadata.json."""
    src_dir = tmp_path / "pdf_sources" / "pdf_test_001"
    src_dir.mkdir(parents=True)
    shutil.copy(sample_pdf_path, src_dir / "original.pdf")
    (src_dir / "metadata.json").write_text(
        json.dumps(sample_metadata), encoding="utf-8"
    )
    return src_dir.parent


@pytest.fixture()
def run_logger(tmp_data_root: Path):
    from slm_hindi.observability.run_logger import IngestionRunLogger
    return IngestionRunLogger(run_id="test-run-001", log_path=tmp_data_root / "pipeline_run_log.csv")


@pytest.fixture()
def file_registry(tmp_data_root: Path):
    from slm_hindi.observability.file_registry import FileRegistry
    return FileRegistry(run_id="test-run-001", registry_path=tmp_data_root / "data_file_registry.csv")
