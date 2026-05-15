"""Integration test: Sangraha load (mocked) → normalize → filter → dedup."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from slm_hindi.config.settings import QualityFilterConfig, SangrahaSourceConfig
from slm_hindi.ingestion.deduplicator import Deduplicator
from slm_hindi.ingestion.quality_filter import QualityFilter
from slm_hindi.ingestion.sangraha_loader import SangrahaLoader
from slm_hindi.ingestion.text_normalizer import TextNormalizer

SAMPLE_ROWS_PATH = Path(__file__).parents[1] / "fixtures" / "sample_sangraha" / "sample_rows.jsonl"


def _load_sample() -> list[dict]:
    rows = []
    with open(SAMPLE_ROWS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_sangraha_pipeline_end_to_end():
    sample_rows = _load_sample()
    assert len(sample_rows) == 5

    # 1. Load (mocked HuggingFace)
    config = SangrahaSourceConfig(max_records=5)
    loader = SangrahaLoader(config, run_id="integration-test")
    with patch("datasets.load_dataset", return_value=sample_rows):
        records = loader.load()
    assert len(records) == 5

    # 2. Normalize
    normalizer = TextNormalizer(QualityFilterConfig())
    records = normalizer.normalize_records(records)
    assert all(r.final_text for r in records)

    # 3. Quality filter
    qfilter = QualityFilter(QualityFilterConfig())
    passed, rejected = qfilter.filter(records)
    assert len(passed) > 0  # sample rows are valid Hindi
    assert all(r.devanagari_ratio > 0.5 for r in passed)

    # 4. Dedup — all 5 rows are distinct
    dedup = Deduplicator()
    deduped = dedup.deduplicate(passed)
    assert len(deduped) == len(passed)


def test_duplicate_sangraha_rows_deduped():
    sample_rows = _load_sample()
    # Inject exact duplicate of first row
    duplicate = dict(sample_rows[0])
    duplicate["id"] = "dupe_001"
    all_rows = sample_rows + [duplicate]

    config = SangrahaSourceConfig(max_records=len(all_rows))
    loader = SangrahaLoader(config)
    with patch("datasets.load_dataset", return_value=all_rows):
        records = loader.load()

    dedup = Deduplicator()
    deduped = dedup.deduplicate(records)
    # The duplicate should be removed
    assert len(deduped) < len(records)
