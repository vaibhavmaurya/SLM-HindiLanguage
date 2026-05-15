"""Exact and near-duplicate removal using SHA-256 hash and MinHash LSH."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import TYPE_CHECKING, Callable

from slm_hindi.schema.corpus_record import CorpusRecord

if TYPE_CHECKING:
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "dedup"
_COMPONENT = "deduplicator"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_shingles(text: str, n: int = 3) -> set[str]:
    words = text.split()
    return {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


class Deduplicator:
    def __init__(
        self,
        num_perm: int = 128,
        jaccard_threshold: float = 0.85,
        shingle_size: int = 3,
    ) -> None:
        self._num_perm = num_perm
        self._threshold = jaccard_threshold
        self._shingle_size = shingle_size

    def deduplicate(
        self,
        records: list[CorpusRecord],
        run_logger: IngestionRunLogger | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[CorpusRecord]:
        if run_logger:
            run_logger.log_event(
                phase=_PHASE, component=_COMPONENT, status="started", records_in=len(records)
            )

        # Pass 1: exact dedup by SHA-256
        records = self._exact_dedup(records)
        # Pass 2: near-dedup by MinHash LSH
        records = self._near_dedup(records)

        if progress_callback:
            progress_callback(len(records))
        logger.info("After dedup: %d records remain", len(records))
        if run_logger:
            run_logger.log_event(
                phase=_PHASE,
                component=_COMPONENT,
                status="completed",
                records_out=len(records),
            )
        return records

    def _exact_dedup(self, records: list[CorpusRecord]) -> list[CorpusRecord]:
        seen: set[str] = set()
        unique: list[CorpusRecord] = []
        for record in records:
            h = _sha256(record.final_text)
            record.dedup_hash = h
            if h not in seen:
                seen.add(h)
                unique.append(record)
        removed = len(records) - len(unique)
        if removed:
            logger.info("Exact dedup removed %d records", removed)
        return unique

    def _near_dedup(self, records: list[CorpusRecord]) -> list[CorpusRecord]:
        try:
            from datasketch import MinHash, MinHashLSH  # noqa: PLC0415
        except ImportError:
            logger.warning("datasketch not installed — skipping near-dedup")
            return records

        lsh = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
        minhashes: dict[str, MinHash] = {}

        for record in records:
            shingles = _word_shingles(record.final_text, self._shingle_size)
            m = MinHash(num_perm=self._num_perm)
            for s in shingles:
                m.update(s.encode("utf-8"))
            minhashes[record.record_id] = m
            try:
                lsh.insert(record.record_id, m)
            except ValueError:
                pass  # duplicate key — already inserted

        # Assign cluster IDs
        cluster_map: dict[str, str] = {}
        for record in records:
            m = minhashes[record.record_id]
            neighbours = lsh.query(m)
            # Canonical representative is the lexicographically smallest ID in the cluster
            canonical = min(neighbours)
            cluster_id = cluster_map.get(canonical)
            if cluster_id is None:
                cluster_id = str(uuid.uuid4())
                cluster_map[canonical] = cluster_id
            cluster_map[record.record_id] = cluster_id
            record.near_dedup_cluster_id = cluster_id

        # Keep only the canonical record per cluster
        kept_clusters: set[str] = set()
        unique: list[CorpusRecord] = []
        for record in records:
            canonical = min(lsh.query(minhashes[record.record_id]))
            if canonical not in kept_clusters:
                kept_clusters.add(canonical)
                unique.append(record)

        removed = len(records) - len(unique)
        if removed:
            logger.info("Near-dedup removed %d records", removed)
        return unique
