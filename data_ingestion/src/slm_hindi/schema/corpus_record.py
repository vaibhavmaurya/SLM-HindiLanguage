"""Unified corpus record schema for all ingestion sources."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CorpusRecord(BaseModel):
    """Single unit of text in the Hindi corpus, regardless of source."""

    record_id: str
    document_id: str
    paragraph_id: str = ""

    source_type: Literal["huggingface_dataset", "pdf", "wiki"]
    source_name: str
    source_dataset: str | None = None
    source_file_name: str | None = None
    source_url_or_path: str | None = None
    page_number: int | None = None

    raw_text: str
    cleaned_text: str | None = None
    final_text: str = ""

    language: str = "hi"
    script: str = "Devanagari"

    char_count: int = 0
    word_count: int = 0
    estimated_token_count: int = 0
    devanagari_ratio: float = 0.0
    latin_ratio: float = 0.0
    digit_ratio: float = 0.0
    symbol_ratio: float = 0.0
    quality_score: float = 0.0

    cleaning_method: Literal["deterministic_normalization", "ollama_model_assisted", "none"] = (
        "none"
    )
    cleaning_model: str | None = None
    cleaning_model_version: str | None = None
    cleaning_status: Literal["pending", "clean", "quarantined", "skipped"] = "pending"

    dedup_hash: str = ""
    near_dedup_cluster_id: str | None = None

    split_name: Literal["train", "validation", "test"] | None = None

    created_at: str = Field(default_factory=_utcnow)
    ingestion_run_id: str = ""

    @field_validator("char_count", "word_count", "estimated_token_count", mode="before")
    @classmethod
    def _non_negative_int(cls, v: object) -> int:
        v = int(v)  # type: ignore[arg-type]
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("devanagari_ratio", "latin_ratio", "digit_ratio", "symbol_ratio", "quality_score", mode="before")
    @classmethod
    def _ratio_range(cls, v: object) -> float:
        v = float(v)  # type: ignore[arg-type]
        if not (0.0 <= v <= 1.0):
            raise ValueError("ratio must be between 0.0 and 1.0")
        return v
