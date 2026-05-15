"""Discover and validate user-provided PDF sources from the input directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class PdfMetadata(BaseModel):
    source_id: str
    source_name: str
    file_name: str
    source_category: str = ""
    provided_by: str = "user"
    license_or_usage_note: str = ""
    language: str = "hi"
    script: str = "Devanagari"
    notes: str = ""


class PdfSource(BaseModel):
    source_id: str
    pdf_path: Path
    metadata: PdfMetadata

    model_config = {"arbitrary_types_allowed": True}


class PdfRegistry:
    def __init__(self, input_dir: str | Path, require_metadata: bool = True) -> None:
        self._input_dir = Path(input_dir)
        self._require_metadata = require_metadata

    def discover(self) -> list[PdfSource]:
        sources: list[PdfSource] = []
        if not self._input_dir.exists():
            logger.warning("PDF input directory does not exist: %s", self._input_dir)
            return sources

        for source_dir in sorted(self._input_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            source = self._validate_source(source_dir)
            if source:
                sources.append(source)

        logger.info("Discovered %d valid PDF sources", len(sources))
        return sources

    def _validate_source(self, source_dir: Path) -> PdfSource | None:
        pdf_path = source_dir / "original.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing original.pdf in {source_dir}")

        metadata_path = source_dir / "metadata.json"
        if self._require_metadata and not metadata_path.exists():
            raise ValueError(f"Missing metadata.json in {source_dir}")

        if metadata_path.exists():
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            try:
                metadata = PdfMetadata(**raw)
            except ValidationError as exc:
                raise ValueError(f"Invalid metadata.json in {source_dir}: {exc}") from exc
        else:
            metadata = PdfMetadata(
                source_id=source_dir.name,
                source_name=source_dir.name,
                file_name="original.pdf",
            )

        return PdfSource(
            source_id=metadata.source_id,
            pdf_path=pdf_path,
            metadata=metadata,
        )
