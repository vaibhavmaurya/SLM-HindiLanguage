"""Validate model-cleaned text against 7 quality checks; quarantine failures."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

from slm_hindi.config.settings import CleaningValidationConfig
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "cleaning_validate"
_COMPONENT = "cleaning_validator"

_DEVANAGARI_RANGE = (0x0900, 0x097F)
_LATIN_RANGE = (0x0041, 0x007A)


def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    count = sum(1 for ch in text if _DEVANAGARI_RANGE[0] <= ord(ch) <= _DEVANAGARI_RANGE[1])
    return count / len(text)


def _has_prompt_echo(output: str, prompt_marker: str = "Cleaned Hindi text:") -> bool:
    return prompt_marker.lower() in output.lower()


def _has_repeated_lines(text: str, min_repeats: int = 3) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i in range(len(lines) - min_repeats + 1):
        if len(set(lines[i : i + min_repeats])) == 1:
            return True
    return False


def _is_mostly_english(text: str, threshold: float = 0.70) -> bool:
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total_alpha = sum(1 for ch in text if ch.isalpha())
    if total_alpha == 0:
        return False
    return (latin / total_alpha) >= threshold


class CleaningValidator:
    def __init__(self, config: CleaningValidationConfig) -> None:
        self._cfg = config

    def validate(self, record: CorpusRecord) -> tuple[CorpusRecord, bool]:
        """Return (record, passed). Sets cleaning_status accordingly."""
        output = record.cleaned_text or ""
        raw = record.raw_text

        rejection_reason = self._find_rejection_reason(output, raw)
        if rejection_reason:
            record.cleaning_status = "quarantined"
            logger.debug("Quarantined %s: %s", record.record_id, rejection_reason)
            return record, False

        record.cleaning_status = "clean"
        return record, True

    def _find_rejection_reason(self, output: str, raw: str) -> str | None:
        if self._cfg.reject_if_output_empty and not output.strip():
            return "empty_output"

        if len(output) < self._cfg.min_output_char_count:
            return f"output_too_short ({len(output)} < {self._cfg.min_output_char_count})"

        if raw:
            ratio = len(output) / len(raw)
            if ratio < self._cfg.min_output_to_input_length_ratio:
                return f"output_compressed ({ratio:.2f} < {self._cfg.min_output_to_input_length_ratio})"
            if ratio > self._cfg.max_output_to_input_length_ratio:
                return f"output_expanded ({ratio:.2f} > {self._cfg.max_output_to_input_length_ratio})"

        if _devanagari_ratio(output) < self._cfg.min_devanagari_ratio:
            return f"devanagari_ratio_low ({_devanagari_ratio(output):.2f})"

        if self._cfg.reject_if_prompt_echo and _has_prompt_echo(output):
            return "prompt_echo_detected"

        if self._cfg.reject_if_repeated_lines and _has_repeated_lines(output):
            return "repeated_lines"

        if _is_mostly_english(output):
            return "output_is_english"

        return None

    def validate_batch(
        self,
        records: list[CorpusRecord],
        run_logger: IngestionRunLogger | None = None,
    ) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
        if run_logger:
            run_logger.log_event(
                phase=_PHASE, component=_COMPONENT, status="started", records_in=len(records)
            )

        passed: list[CorpusRecord] = []
        quarantined: list[CorpusRecord] = []
        for record in records:
            _, ok = self.validate(record)
            (passed if ok else quarantined).append(record)

        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                records_in=len(records),
                records_out=len(passed),
                records_rejected=len(quarantined),
            )
        return passed, quarantined

    def save_quarantine(self, records: list[CorpusRecord], output_path: str | Path) -> None:
        if not records:
            return
        import pandas as pd  # noqa: PLC0415

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([r.model_dump() for r in records])
        df.to_parquet(output_path, index=False, compression="zstd")
        logger.info("Saved %d quarantined records to %s", len(records), output_path)
