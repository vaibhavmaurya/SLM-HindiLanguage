"""Load AI4Bharat Sangraha Hindi corpus and map to unified CorpusRecord schema."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from slm_hindi.config.settings import SangrahaSourceConfig
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_SOURCE_NAME = "ai4bharat/sangraha"
_PHASE = "sangraha_load"
_COMPONENT = "sangraha_loader"


class SangrahaLoader:
    def __init__(self, config: SangrahaSourceConfig, run_id: str = "") -> None:
        self._config = config
        self._run_id = run_id or str(uuid.uuid4())

    def load(
        self,
        run_logger: IngestionRunLogger | None = None,
    ) -> list[CorpusRecord]:
        from datasets import load_dataset  # noqa: PLC0415  (lazy import — heavy dependency)

        if run_logger:
            run_logger.log_event(
                phase=_PHASE, component=_COMPONENT, status="started", source_id=_SOURCE_NAME
            )

        dataset = load_dataset(
            self._config.dataset_name,
            data_dir=self._config.data_dir,
            split=self._config.split,
            streaming=self._config.streaming,
            cache_dir=self._config.cache_dir,
        )

        records: list[CorpusRecord] = []
        for i, row in enumerate(dataset):
            if self._config.max_records is not None and i >= self._config.max_records:
                break
            record = self._map_row(row, i)
            records.append(record)

        logger.info("Loaded %d Sangraha records", len(records))
        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                source_id=_SOURCE_NAME,
                records_in=len(records),
                records_out=len(records),
            )
        return records

    def _map_row(self, row: dict, index: int) -> CorpusRecord:
        text = row.get("text", "")
        doc_id = row.get("id", f"sangraha_{index:08d}")
        return CorpusRecord(
            record_id=f"{doc_id}_p0000",
            document_id=str(doc_id),
            paragraph_id="p0000",
            source_type="huggingface_dataset",
            source_name=_SOURCE_NAME,
            source_dataset=f"{self._config.dataset_name}/{self._config.data_dir}",
            raw_text=text,
            final_text=text,
            char_count=len(text),
            word_count=len(text.split()),
            estimated_token_count=max(1, int(len(text) / 4.5)),
            cleaning_method="deterministic_normalization",
            cleaning_model=None,
            cleaning_status="pending",
            ingestion_run_id=self._run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
