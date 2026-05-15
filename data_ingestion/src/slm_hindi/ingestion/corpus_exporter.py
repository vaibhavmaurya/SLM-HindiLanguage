"""Export the final corpus as sharded Parquet, JSONL.gz, and TXT.gz."""

from __future__ import annotations

import gzip
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from slm_hindi.config.settings import ExportConfig
from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.file_registry import FileRegistry
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "export"
_COMPONENT = "corpus_exporter"


class CorpusExporter:
    def __init__(self, config: ExportConfig, data_root: str | Path = "data") -> None:
        self._cfg = config
        self._data_root = Path(data_root)

    def export(
        self,
        split_records: dict[str, list[CorpusRecord]],
        run_logger: IngestionRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        corpus_version = self._cfg.naming.corpus_version
        if run_logger:
            run_logger.log_event(phase=_PHASE, component=_COMPONENT, status="started")

        for split_name, records in split_records.items():
            if not records:
                continue
            if self._cfg.exports.parquet.enabled:
                self._write_parquet(records, split_name, corpus_version, file_registry)
            if self._cfg.exports.jsonl.enabled:
                self._write_jsonl(records, split_name, corpus_version, file_registry)
            if self._cfg.exports.text.enabled:
                self._write_text(records, split_name, corpus_version, file_registry)
            if progress_callback:
                progress_callback(len(records))

        if run_logger:
            run_logger.log_event(phase=_PHASE, component=_COMPONENT, status="completed")

    def _shard(self, records: list[CorpusRecord], shard_size_mb: int) -> list[list[CorpusRecord]]:
        # Rough size estimate: 4 bytes per character of final_text
        target_bytes = shard_size_mb * 1024 * 1024
        shards: list[list[CorpusRecord]] = []
        current: list[CorpusRecord] = []
        current_size = 0
        for record in records:
            size = len(record.final_text.encode("utf-8"))
            if current and current_size + size > target_bytes:
                shards.append(current)
                current = []
                current_size = 0
            current.append(record)
            current_size += size
        if current:
            shards.append(current)
        return shards

    def _write_parquet(
        self,
        records: list[CorpusRecord],
        split_name: str,
        corpus_version: str,
        file_registry: FileRegistry | None,
    ) -> None:
        out_dir = self._data_root / "final" / "parquet" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        shards = self._shard(records, self._cfg.exports.parquet.shard_size_mb)
        for i, shard in enumerate(shards):
            fname = f"{corpus_version}_{split_name}_{i:05d}.parquet"
            out_path = out_dir / fname
            df = pd.DataFrame([r.model_dump() for r in shard])
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, str(out_path), compression=self._cfg.exports.parquet.compression)
            logger.info("Wrote %s (%d records)", out_path, len(shard))
            if file_registry:
                file_registry.register_file(
                    out_path, role="output", stage="export", file_format="parquet",
                    row_count=len(shard), compression=self._cfg.exports.parquet.compression,
                    notes=f"split={split_name},shard_index={i}",
                )

    def _write_jsonl(
        self,
        records: list[CorpusRecord],
        split_name: str,
        corpus_version: str,
        file_registry: FileRegistry | None,
    ) -> None:
        out_dir = self._data_root / "final" / "training_jsonl" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        shards = self._shard(records, self._cfg.exports.parquet.shard_size_mb)
        for i, shard in enumerate(shards):
            fname = f"{corpus_version}_{split_name}_{i:05d}.jsonl.gz"
            out_path = out_dir / fname
            with gzip.open(str(out_path), "wt", encoding="utf-8") as fh:
                for record in shard:
                    obj: dict = {"text": record.final_text}
                    if self._cfg.exports.jsonl.include_metadata:
                        obj.update({
                            "source_type": record.source_type,
                            "document_id": record.document_id,
                            "record_id": record.record_id,
                        })
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if file_registry:
                file_registry.register_file(
                    out_path, role="output", stage="export", file_format="jsonl.gz",
                    row_count=len(shard), compression="gzip",
                    notes=f"split={split_name},shard_index={i}",
                )

    def _write_text(
        self,
        records: list[CorpusRecord],
        split_name: str,
        corpus_version: str,
        file_registry: FileRegistry | None,
    ) -> None:
        out_dir = self._data_root / "final" / "training_text" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        shards = self._shard(records, self._cfg.exports.parquet.shard_size_mb)
        sep = self._cfg.exports.text.separator
        for i, shard in enumerate(shards):
            fname = f"{corpus_version}_{split_name}_{i:05d}.txt.gz"
            out_path = out_dir / fname
            with gzip.open(str(out_path), "wt", encoding="utf-8") as fh:
                for record in shard:
                    fh.write(record.final_text)
                    fh.write(sep)
            if file_registry:
                file_registry.register_file(
                    out_path, role="output", stage="export", file_format="txt.gz",
                    row_count=len(shard), compression="gzip",
                    notes=f"split={split_name},shard_index={i}",
                )
