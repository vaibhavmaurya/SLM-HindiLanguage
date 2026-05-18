"""Pydantic v2 schemas for tokenizer pipeline structured data."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SampleManifest(BaseModel):
    version: str
    source_folder: str
    text_column: str
    target_size_gb: float
    actual_size_gb: float
    written_records: int
    random_seed: int
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ValidationThresholdValues(BaseModel):
    max_unk_rate: float = 0.001
    min_chars_per_token: float = 3.0
    max_tokens_per_word: float = 2.5
    min_roundtrip_success_rate: float = 0.99
    min_devanagari_coverage: float = 0.995
    max_special_token_split_failures: int = 0


class ValidationReport(BaseModel):
    variant_name: str
    vocab_size: int
    tokenizer_version: str
    unk_rate: float
    chars_per_token: float
    tokens_per_word: float
    roundtrip_success_rate: float
    devanagari_char_coverage: float
    special_token_split_failures: int
    thresholds: ValidationThresholdValues
    passes_thresholds: bool = False

    def model_post_init(self, __context: Any) -> None:
        t = self.thresholds
        self.passes_thresholds = (
            self.unk_rate < t.max_unk_rate
            and self.chars_per_token > t.min_chars_per_token
            and self.tokens_per_word < t.max_tokens_per_word
            and self.roundtrip_success_rate >= t.min_roundtrip_success_rate
            and self.devanagari_char_coverage > t.min_devanagari_coverage
            and self.special_token_split_failures <= t.max_special_token_split_failures
        )


class ComparisonResult(BaseModel):
    variants: list[ValidationReport]
    recommended_variant: str

    @field_validator("variants", mode="before")
    @classmethod
    def reject_bare_dicts(cls, v: Any) -> Any:
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    raise TypeError(f"Expected ValidationReport instance, got dict: {item!r}")
        return v

    @model_validator(mode="after")
    def check_recommended_in_variants(self) -> ComparisonResult:
        names = [r.variant_name for r in self.variants]
        if self.recommended_variant not in names:
            raise ValueError(
                f"recommended_variant {self.recommended_variant!r} not in variant names {names}"
            )
        return self
