"""Extract text from PDF files using PyMuPDF with pdfplumber as fallback."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from slm_hindi.config.settings import PdfExtractionConfig
from slm_hindi.ingestion.pdf_registry import PdfSource
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "pdf_extract"
_COMPONENT = "pdf_extractor"


class PdfExtractor:
    def __init__(self, config: PdfExtractionConfig, run_id: str = "") -> None:
        self._config = config
        self._run_id = run_id or str(uuid.uuid4())

    def extract(
        self,
        source: PdfSource,
        run_logger: IngestionRunLogger | None = None,
    ) -> list[CorpusRecord]:
        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="started",
                source_id=source.source_id,
            )

        try:
            records = self._extract_with_pymupdf(source)
            method = "pymupdf"
        except Exception as exc:
            logger.warning("PyMuPDF failed for %s: %s — falling back to pdfplumber", source.pdf_path, exc)
            records = self._extract_with_pdfplumber(source)
            method = "pdfplumber"

        for record in records:
            record.source_file_name = source.metadata.file_name
            record.source_url_or_path = str(source.pdf_path)
            # Store extraction method in notes via cleaning_model_version field
            record.cleaning_model_version = method

        logger.info("Extracted %d pages from %s using %s", len(records), source.pdf_path, method)
        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                source_id=source.source_id,
                records_out=len(records),
                notes=f"engine={method}",
            )
        return records

    def _extract_with_pymupdf(self, source: PdfSource) -> list[CorpusRecord]:
        import fitz  # noqa: PLC0415

        records: list[CorpusRecord] = []
        with fitz.open(str(source.pdf_path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if len(text) < self._config.min_page_text_chars:
                    continue
                text = text[: self._config.max_page_text_chars]
                records.append(self._make_record(source, page_num, text, "pymupdf"))
        return records

    def _extract_with_pdfplumber(self, source: PdfSource) -> list[CorpusRecord]:
        import pdfplumber  # noqa: PLC0415

        records: list[CorpusRecord] = []
        with pdfplumber.open(str(source.pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if len(text) < self._config.min_page_text_chars:
                    continue
                text = text[: self._config.max_page_text_chars]
                records.append(self._make_record(source, page_num, text, "pdfplumber"))
        return records

    def _make_record(
        self, source: PdfSource, page_num: int, text: str, method: str
    ) -> CorpusRecord:
        doc_id = source.source_id
        rec_id = f"{doc_id}_p{page_num:04d}"
        return CorpusRecord(
            record_id=rec_id,
            document_id=doc_id,
            paragraph_id=f"p{page_num:04d}",
            source_type="pdf",
            source_name="user_provided_pdfs",
            source_file_name=source.metadata.file_name,
            source_url_or_path=str(source.pdf_path),
            page_number=page_num,
            raw_text=text,
            final_text=text,
            char_count=len(text),
            word_count=len(text.split()),
            estimated_token_count=max(1, int(len(text) / 4.5)),
            cleaning_method="ollama_model_assisted",
            cleaning_model="qwen3",
            cleaning_status="pending",
            cleaning_model_version=method,
            ingestion_run_id=self._run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
