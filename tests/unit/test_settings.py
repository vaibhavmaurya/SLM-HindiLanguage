"""Tests for config/settings.py — IngestionSettings loading from YAML."""

from pathlib import Path

import pytest

from slm_hindi.config.settings import IngestionSettings, load_settings

SAMPLE_CONFIG = Path(__file__).parents[1] / "fixtures" / "sample_configs" / "test_ingestion_config.yaml"


def test_load_settings_from_yaml():
    settings = load_settings(SAMPLE_CONFIG)
    assert isinstance(settings, IngestionSettings)
    assert settings.project.name == "hindi-slm-test"
    assert settings.project.corpus_version == "hindi_corpus_test_v001"


def test_sources_sangraha_enabled():
    settings = load_settings(SAMPLE_CONFIG)
    assert settings.sources.sangraha.enabled is True
    assert settings.sources.sangraha.dataset_name == "ai4bharat/sangraha"
    assert settings.sources.sangraha.max_records == 5


def test_sources_pdf_enabled():
    settings = load_settings(SAMPLE_CONFIG)
    assert settings.sources.pdf.enabled is True


def test_runtime_defaults():
    settings = load_settings(SAMPLE_CONFIG)
    assert settings.runtime.batch_size == 10
    assert settings.runtime.random_seed == 42


def test_defaults_applied_for_missing_sub_configs():
    """When sub-config YAML files are absent, pydantic defaults should apply."""
    settings = load_settings(SAMPLE_CONFIG)
    # pdf_extraction comes from pdf_extraction_config.yaml which exists in configs/
    # but test config points to a dir without it — defaults kick in
    assert settings.pdf_extraction.primary_engine in ("pymupdf", "pdfplumber")
    assert settings.model_cleaning.model == "qwen3"
