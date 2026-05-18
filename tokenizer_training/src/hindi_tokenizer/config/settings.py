"""TokenizerSettings: typed config loaded from YAML with env-var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectSettings(BaseModel):
    name: str
    tokenizer_version: str
    hf_repo_id: str = ""
    data_root: str = "data"
    log_level: str = "INFO"


class InputSettings(BaseModel):
    parquet_train_folder: str
    text_column: str = "final_text"
    file_pattern: str = "*.parquet"


class SamplingProfile(BaseModel):
    target_size_gb: float
    output_file: str


class SamplingSettings(BaseModel):
    random_seed: int = 42
    smoke_test: SamplingProfile
    experiment: SamplingProfile
    final: SamplingProfile


class TextFilterSettings(BaseModel):
    min_char_count: int = 30
    max_char_count: int = 5000
    min_devanagari_ratio: float = 0.60
    remove_empty: bool = True
    normalize_unicode: bool = True


class TokenizerConfig(BaseModel):
    algorithm: str = "unigram"
    vocab_sizes: list[int] = Field(default_factory=lambda: [24000, 32000, 48000])
    default_vocab_size: int = 32000
    normalizer: str = "nfkc"
    pre_tokenizer: str = "metaspace"
    decoder: str = "metaspace"
    max_piece_length: int = 24
    n_sub_iterations: int = 2
    model_max_length: int = 2048


class SpecialTokenSettings(BaseModel):
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    additional_special_tokens: list[str] = Field(
        default_factory=lambda: ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    )

    def all_tokens(self) -> list[str]:
        return [self.pad_token, self.unk_token, self.bos_token, self.eos_token] + self.additional_special_tokens


class ArtifactSettings(BaseModel):
    artifact_dir: str = "data/artifacts"
    final_dir: str


class ValidationThresholds(BaseModel):
    max_unk_rate: float = 0.001
    min_chars_per_token: float = 3.0
    max_tokens_per_word: float = 2.5
    min_roundtrip_success_rate: float = 0.99
    min_devanagari_coverage: float = 0.995
    max_special_token_split_failures: int = 0


class ValidationSettings(BaseModel):
    thresholds: ValidationThresholds = Field(default_factory=ValidationThresholds)


class TokenizerSettings(BaseModel):
    project: ProjectSettings
    input: InputSettings
    sampling: SamplingSettings
    text_filters: TextFilterSettings = Field(default_factory=TextFilterSettings)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    special_tokens: SpecialTokenSettings = Field(default_factory=SpecialTokenSettings)
    artifacts: ArtifactSettings
    validation: ValidationSettings = Field(default_factory=ValidationSettings)


def load_settings(config_path: str | Path) -> TokenizerSettings:
    path = Path(config_path)
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    tokenizer_version = os.environ.get("TOKENIZER_VERSION")
    if tokenizer_version is not None:
        data.setdefault("project", {})["tokenizer_version"] = tokenizer_version

    return TokenizerSettings(**data)
