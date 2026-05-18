"""Tests for TokenizerSettings config loading — Phase 2."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hindi_tokenizer.config.settings import load_settings


def test_load_config_from_yaml(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.project.name == "hindi-slm-tokenizer"


def test_default_vocab_size_is_32k(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.tokenizer.default_vocab_size == 32000


def test_vocab_sizes_contains_three_variants(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.tokenizer.vocab_sizes == [24000, 32000, 48000]


def test_min_char_count_default(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.text_filters.min_char_count == 30


def test_max_char_count_default(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.text_filters.max_char_count == 5000


def test_min_devanagari_ratio_default(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.text_filters.min_devanagari_ratio == pytest.approx(0.60)


def test_special_tokens_count(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert len(settings.special_tokens.all_tokens()) == 8


def test_pad_token_value(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.special_tokens.pad_token == "<pad>"


def test_random_seed_default(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.sampling.random_seed == 42


def test_algorithm_is_unigram(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.tokenizer.algorithm == "unigram"


def test_normalizer_is_nfkc(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.tokenizer.normalizer == "nfkc"


def test_pre_tokenizer_is_metaspace(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.tokenizer.pre_tokenizer == "metaspace"


def test_validation_max_unk_rate(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.validation.thresholds.max_unk_rate == pytest.approx(0.001)


def test_validation_min_roundtrip(test_config_path: Path) -> None:
    settings = load_settings(test_config_path)
    assert settings.validation.thresholds.min_roundtrip_success_rate == pytest.approx(0.99)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    bad_config: dict = {
        # project section omitted — name is required
        "input": {"parquet_train_folder": "data/train"},
        "sampling": {
            "random_seed": 42,
            "smoke_test": {"target_size_gb": 0.001, "output_file": "data/a.txt"},
            "experiment": {"target_size_gb": 0.001, "output_file": "data/b.txt"},
            "final": {"target_size_gb": 0.001, "output_file": "data/c.txt"},
        },
        "text_filters": {},
        "tokenizer": {},
        "special_tokens": {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
        "artifacts": {"artifact_dir": "data/artifacts", "final_dir": "data/final"},
    }
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text(yaml.dump(bad_config), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config_file)


def test_env_var_overrides_yaml(test_config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENIZER_VERSION", "test_v999")
    settings = load_settings(test_config_path)
    assert settings.project.tokenizer_version == "test_v999"
