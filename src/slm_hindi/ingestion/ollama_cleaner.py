"""Clean extracted PDF text using Ollama + Qwen3 via the local REST API."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

import requests

from slm_hindi.config.settings import ModelCleaningConfig
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "model_clean"
_COMPONENT = "ollama_cleaner"

_PROMPT_TEMPLATE = """\
You are cleaning Hindi text extracted from a PDF for language model pretraining.

Your task:
- Clean extraction noise only.
- Preserve the original meaning.
- Preserve the original Hindi wording as much as possible.
- Remove page numbers, headers, footers, broken line breaks, repeated boilerplate, and OCR artifacts.
- Do not summarize.
- Do not translate.
- Do not add new information.
- Do not rewrite creatively.
- Return only the cleaned Hindi text.

Input text:
{raw_text}

Cleaned Hindi text:
"""


class OllamaError(RuntimeError):
    pass


class OllamaCleaner:
    def __init__(self, config: ModelCleaningConfig, run_id: str = "") -> None:
        self._config = config
        self._run_id = run_id or str(uuid.uuid4())

    def clean(
        self,
        records: list[CorpusRecord],
        run_logger: IngestionRunLogger | None = None,
    ) -> list[CorpusRecord]:
        if not self._config.enabled:
            for r in records:
                r.cleaned_text = r.raw_text
            return records

        if run_logger:
            run_logger.log_event(
                phase=_PHASE, component=_COMPONENT, status="started", records_in=len(records)
            )

        cleaned: list[CorpusRecord] = []
        for record in records:
            chunks = self._chunk_text(record.raw_text)
            cleaned_parts: list[str] = []
            for chunk in chunks:
                result = self._call_ollama(chunk)
                cleaned_parts.append(result)
            record.cleaned_text = "\n\n".join(cleaned_parts).strip()
            record.cleaning_model = self._config.model
            cleaned.append(record)

        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                records_in=len(records),
                records_out=len(cleaned),
            )
        return cleaned

    def _chunk_text(self, text: str) -> list[str]:
        max_chars = self._config.chunking.max_input_chars
        overlap = self._config.chunking.overlap_chars
        split_on_para = self._config.chunking.split_on_paragraph

        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to split at paragraph boundary
            if split_on_para:
                para_break = text.rfind("\n\n", start, end)
                if para_break > start:
                    end = para_break

            chunks.append(text[start:end])
            start = end - overlap

        return chunks

    def _call_ollama(self, text: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(raw_text=text)
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._config.generation_options.temperature,
                "top_p": self._config.generation_options.top_p,
                "repeat_penalty": self._config.generation_options.repeat_penalty,
            },
        }

        backoff = self._config.retry_backoff_base_seconds
        last_exc: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                resp = requests.post(
                    self._config.endpoint,
                    json=payload,
                    timeout=self._config.request_timeout_seconds,
                )
                resp.raise_for_status()
                return resp.json()["response"].strip()
            except requests.Timeout as exc:
                last_exc = exc
                logger.warning("Ollama timeout on attempt %d/%d", attempt, self._config.max_retries)
                if attempt < self._config.max_retries:
                    time.sleep(backoff * (2 ** (attempt - 1)))
            except requests.HTTPError as exc:
                raise OllamaError(f"Ollama HTTP error: {exc}") from exc

        raise OllamaError(f"Ollama timed out after {self._config.max_retries} attempts") from last_exc
