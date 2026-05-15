"""Integration test: Full end-to-end pipeline on sample data (Ollama mocked)."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

from slm_hindi.config.settings import load_settings
from slm_hindi.ingestion.corpus_exporter import CorpusExporter
from slm_hindi.ingestion.corpus_splitter import CorpusSplitter
from slm_hindi.ingestion.deduplicator import Deduplicator
from slm_hindi.ingestion.manifest_generator import ManifestGenerator
from slm_hindi.ingestion.quality_filter import QualityFilter
from slm_hindi.ingestion.sangraha_loader import SangrahaLoader
from slm_hindi.ingestion.text_normalizer import TextNormalizer
from slm_hindi.observability.file_registry import FileRegistry
from slm_hindi.observability.run_logger import IngestionRunLogger

SAMPLE_ROWS_PATH = Path(__file__).parents[1] / "fixtures" / "sample_sangraha" / "sample_rows.jsonl"
SAMPLE_CONFIG = Path(__file__).parents[1] / "fixtures" / "sample_configs" / "test_ingestion_config.yaml"

_CLEAN_HINDI = "यह एक साफ हिंदी पाठ है जो परीक्षण के लिए उपयोग किया जाता है और पर्याप्त लंबा है।" * 2


def _load_sample_rows() -> list[dict]:
    rows = []
    with open(SAMPLE_ROWS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mock_ollama_response():
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.json.return_value = {"response": _CLEAN_HINDI}
    resp.raise_for_status.return_value = None
    return resp


def test_full_pipeline_produces_parquet_output(tmp_path: Path):
    run_id = "integration-full-001"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    run_logger = IngestionRunLogger(run_id=run_id, log_path=reports_dir / "pipeline_run_log.csv")
    file_registry = FileRegistry(run_id=run_id, registry_path=reports_dir / "data_file_registry.csv")

    settings = load_settings(SAMPLE_CONFIG)

    # Sangraha source
    sample_rows = _load_sample_rows()
    sangraha_loader = SangrahaLoader(settings.sources.sangraha, run_id=run_id)
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = sangraha_loader.load(run_logger=run_logger)

    normalizer = TextNormalizer(settings.quality_filter)
    records = normalizer.normalize_records(records, run_logger=run_logger)

    # Quality filter
    qfilter = QualityFilter(settings.quality_filter)
    records, _ = qfilter.filter(records, run_logger=run_logger)

    # Dedup
    dedup = Deduplicator()
    records = dedup.deduplicate(records, run_logger=run_logger)

    # Split
    splitter = CorpusSplitter(settings.export.splits)
    splits = splitter.split(records, run_logger=run_logger)

    # Export
    exporter = CorpusExporter(settings.export, data_root=tmp_path)
    exporter.export(splits, run_logger=run_logger, file_registry=file_registry)

    # Manifest
    gen = ManifestGenerator(settings.export, data_root=tmp_path)
    gen.generate(splits, run_logger=run_logger, file_registry=file_registry)

    # Assertions
    parquet_files = list((tmp_path / "final" / "parquet").rglob("*.parquet"))
    assert len(parquet_files) >= 1

    # corpus_version comes from export.naming.corpus_version (defaults to hindi_corpus_v001)
    corpus_version = settings.export.naming.corpus_version
    manifest_path = tmp_path / "reports" / f"{corpus_version}_manifest.json"
    assert manifest_path.exists()


def test_pipeline_run_log_has_completed_events(tmp_path: Path):
    run_id = "integration-log-001"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    run_logger = IngestionRunLogger(run_id=run_id, log_path=reports_dir / "pipeline_run_log.csv")

    run_logger.log_event(phase="sangraha_load", component="sangraha_loader", status="started")
    run_logger.log_event(phase="sangraha_load", component="sangraha_loader", status="completed", records_out=5)
    run_logger.log_event(phase="normalize", component="text_normalizer", status="completed", records_out=5)

    with open(reports_dir / "pipeline_run_log.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    phases = [r["phase"] for r in rows]
    statuses = [r["status"] for r in rows]

    assert "sangraha_load" in phases
    assert "completed" in statuses


def test_data_file_registry_populated(tmp_path: Path):
    run_id = "integration-registry-001"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    file_registry = FileRegistry(run_id=run_id, registry_path=reports_dir / "data_file_registry.csv")

    # Create a test file and register it
    test_file = tmp_path / "test_output.parquet"
    test_file.write_bytes(b"fake parquet data")
    file_registry.register_file(test_file, role="output", stage="export", file_format="parquet")

    with open(reports_dir / "data_file_registry.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    assert rows[0]["file_name"] == "test_output.parquet"
    assert len(rows[0]["sha256"]) == 64
