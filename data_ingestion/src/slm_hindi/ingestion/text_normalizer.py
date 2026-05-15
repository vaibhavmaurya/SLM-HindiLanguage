"""Deterministic text normalization: Unicode NFC, whitespace, danda, quotes, URLs."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING, Callable

from slm_hindi.config.settings import QualityFilterConfig

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_DECORATIVE = re.compile(r"[★☆✓✗✔✘•◦▪▫►◄→←↑↓]")


def _remove_repeated_lines(text: str, min_repeats: int = 3) -> str:
    lines = text.splitlines()
    result: list[str] = []
    i = 0
    while i < len(lines):
        # Count consecutive identical non-empty lines
        if lines[i].strip():
            j = i
            while j < len(lines) and lines[j] == lines[i]:
                j += 1
            if j - i >= min_repeats:
                result.append(lines[i])  # keep one copy
                i = j
                continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


class TextNormalizer:
    def __init__(self, config: QualityFilterConfig) -> None:
        self._remove_urls = config.remove_urls

    def normalize(self, text: str) -> str:
        # 1. Unicode NFC
        text = unicodedata.normalize("NFC", text)
        # 2. Remove URLs (configurable)
        if self._remove_urls:
            text = _URL_RE.sub(" ", text)
        # 3. Remove decorative symbols
        text = _DECORATIVE.sub("", text)
        # 4. Collapse multiple spaces/tabs (preserve newlines)
        text = _MULTI_SPACE.sub(" ", text)
        # 5. Limit consecutive newlines to 2
        text = _MULTI_NEWLINE.sub("\n\n", text)
        # 6. Remove repeated header/footer lines
        text = _remove_repeated_lines(text)
        # 7. Strip leading/trailing whitespace per line, then globally
        lines = [ln.rstrip() for ln in text.splitlines()]
        text = "\n".join(lines).strip()
        return text

    def normalize_records(
        self,
        records: list,  # list[CorpusRecord] — avoid circular import
        run_logger: IngestionRunLogger | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list:
        from slm_hindi.schema.corpus_record import CorpusRecord  # noqa: PLC0415

        if run_logger:
            run_logger.log_event(
                phase="normalize", component="text_normalizer", status="started", records_in=len(records)
            )

        for record in records:
            source = record.cleaned_text if record.cleaned_text else record.raw_text
            record.final_text = self.normalize(source)
            record.char_count = len(record.final_text)
            record.word_count = len(record.final_text.split())
            record.estimated_token_count = max(1, int(record.char_count / 4.5))
            if progress_callback:
                progress_callback(1)

        if run_logger:
            run_logger.log_event(
                phase="normalize", component="text_normalizer", status="completed", records_out=len(records)
            )
        return records
