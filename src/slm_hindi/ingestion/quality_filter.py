"""Filter corpus records by Hindi language quality thresholds."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from slm_hindi.config.settings import QualityFilterConfig
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "quality_filter"
_COMPONENT = "quality_filter"

_DEVANAGARI_START = 0x0900
_DEVANAGARI_END = 0x097F


def _char_ratios(text: str) -> dict[str, float]:
    if not text:
        return {"devanagari": 0.0, "latin": 0.0, "digit": 0.0, "symbol": 0.0}
    n = len(text)
    deva = sum(1 for ch in text if _DEVANAGARI_START <= ord(ch) <= _DEVANAGARI_END)
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    digit = sum(1 for ch in text if ch.isdigit())
    # Exclude Devanagari block chars from symbol count — combining marks (matras) are not symbols
    symbol = sum(
        1 for ch in text
        if not ch.isalnum() and not ch.isspace()
        and not (_DEVANAGARI_START <= ord(ch) <= _DEVANAGARI_END)
    )
    return {
        "devanagari": deva / n,
        "latin": latin / n,
        "digit": digit / n,
        "symbol": symbol / n,
    }


class QualityFilter:
    def __init__(self, config: QualityFilterConfig) -> None:
        self._cfg = config

    def filter(
        self,
        records: list[CorpusRecord],
        run_logger: IngestionRunLogger | None = None,
    ) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
        if run_logger:
            run_logger.log_event(
                phase=_PHASE, component=_COMPONENT, status="started", records_in=len(records)
            )

        passed: list[CorpusRecord] = []
        rejected: list[CorpusRecord] = []

        for record in records:
            self._annotate(record)
            if self._passes(record):
                passed.append(record)
            else:
                rejected.append(record)

        logger.info("Quality filter: %d passed, %d rejected", len(passed), len(rejected))
        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                records_in=len(records),
                records_out=len(passed),
                records_rejected=len(rejected),
            )
        return passed, rejected

    def _annotate(self, record: CorpusRecord) -> None:
        ratios = _char_ratios(record.final_text)
        record.devanagari_ratio = ratios["devanagari"]
        record.latin_ratio = ratios["latin"]
        record.digit_ratio = ratios["digit"]
        record.symbol_ratio = ratios["symbol"]

        w = self._cfg.quality_score_weights
        # Normalise char count to a 0-1 length score
        length_score = min(1.0, record.char_count / 1000)
        record.quality_score = round(
            record.devanagari_ratio * w.devanagari_ratio + length_score * w.length_score, 4
        )

    def _passes(self, record: CorpusRecord) -> bool:
        if record.devanagari_ratio < self._cfg.min_devanagari_ratio:
            return False
        if record.char_count < self._cfg.min_char_count:
            return False
        if record.char_count > self._cfg.max_char_count:
            return False
        if self._cfg.reject_table_fragments:
            if record.digit_ratio > self._cfg.max_digit_ratio:
                return False
            if record.symbol_ratio > self._cfg.max_symbol_ratio:
                return False
        return True
