"""Document-level stratified train/validation/test split."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from slm_hindi.config.settings import SplitsConfig
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "split"
_COMPONENT = "corpus_splitter"


class CorpusSplitter:
    def __init__(self, config: SplitsConfig) -> None:
        self._cfg = config

    def split(
        self,
        records: list[CorpusRecord],
        run_logger: IngestionRunLogger | None = None,
    ) -> dict[str, list[CorpusRecord]]:
        if run_logger:
            run_logger.log_event(
                phase=_PHASE, component=_COMPONENT, status="started", records_in=len(records)
            )

        # Group records by document_id (document-level split)
        doc_map: dict[str, list[CorpusRecord]] = {}
        for record in records:
            doc_map.setdefault(record.document_id, []).append(record)

        doc_ids = sorted(doc_map.keys())
        rng = random.Random(self._cfg.random_seed)
        rng.shuffle(doc_ids)

        n = len(doc_ids)
        n_val = max(1, round(n * self._cfg.validation))
        n_test = max(1, round(n * self._cfg.test))
        n_train = n - n_val - n_test

        val_ids = set(doc_ids[n_train : n_train + n_val])
        test_ids = set(doc_ids[n_train + n_val :])

        splits: dict[str, list[CorpusRecord]] = {"train": [], "validation": [], "test": []}
        for doc_id, doc_records in doc_map.items():
            if doc_id in val_ids:
                split_name = "validation"
            elif doc_id in test_ids:
                split_name = "test"
            else:
                split_name = "train"
            for record in doc_records:
                record.split_name = split_name  # type: ignore[assignment]
            splits[split_name].extend(doc_records)

        logger.info(
            "Split: train=%d val=%d test=%d records",
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        )
        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                records_in=len(records),
                records_out=len(records),
                notes=f"train={len(splits['train'])},val={len(splits['validation'])},test={len(splits['test'])}",
            )
        return splits
