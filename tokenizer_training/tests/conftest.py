"""Shared fixtures for all tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_parquet_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_parquet"


@pytest.fixture(scope="session")
def sample_text_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_text"


@pytest.fixture(scope="session")
def sample_configs_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_configs"


@pytest.fixture(scope="session")
def validation_sentences_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "validation_sentences.txt"


@pytest.fixture(scope="session")
def test_config_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_configs" / "test_tokenizer_training_config.yaml"


@pytest.fixture(scope="session")
def small_corpus_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_text" / "small_corpus.txt"
