"""Generate SHA-256 manifest and corpus profile JSON for the final corpus."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from slm_hindi.config.settings import ExportConfig

if TYPE_CHECKING:
    from slm_hindi.observability.file_registry import FileRegistry
    from slm_hindi.observability.run_logger import IngestionRunLogger

logger = logging.getLogger(__name__)

_PHASE = "manifest"
_COMPONENT = "manifest_generator"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ManifestGenerator:
    def __init__(self, config: ExportConfig, data_root: str | Path = "data") -> None:
        self._cfg = config
        self._data_root = Path(data_root)

    def generate(
        self,
        split_records: dict[str, list],  # list[CorpusRecord]
        run_logger: IngestionRunLogger | None = None,
        file_registry: FileRegistry | None = None,
    ) -> dict:
        corpus_version = self._cfg.naming.corpus_version
        final_dir = self._data_root / "final"
        reports_dir = self._data_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if run_logger:
            run_logger.log_event(phase=_PHASE, component=_COMPONENT, status="started")

        # Collect all output files
        file_entries: list[dict] = []
        for path in sorted(final_dir.rglob("*")):
            if path.is_file():
                file_entries.append({
                    "path": str(path.relative_to(self._data_root)),
                    "split": self._infer_split(path),
                    "format": self._infer_format(path),
                    "compression": self._infer_compression(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })

        manifest = {
            "corpus_version": corpus_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": "hi",
            "script": "Devanagari",
            "sources": [
                {"source_name": "ai4bharat/sangraha", "source_type": "huggingface_dataset"},
                {"source_name": "user_provided_pdfs", "source_type": "pdf"},
            ],
            "splits": {
                "train": self._cfg.splits.train,
                "validation": self._cfg.splits.validation,
                "test": self._cfg.splits.test,
            },
            "files": file_entries,
        }

        manifest_path = reports_dir / f"{corpus_version}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Written manifest to %s", manifest_path)

        profile = self._build_profile(split_records, corpus_version)
        profile_path = reports_dir / f"{corpus_version}_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Written profile to %s", profile_path)

        if file_registry:
            for p in (manifest_path, profile_path):
                file_registry.register_file(p, role="report", stage="manifest", file_format="json")

        if run_logger:
            run_logger.log_event(phase=_PHASE, component=_COMPONENT, status="completed")

        return manifest

    def _build_profile(self, split_records: dict[str, list], corpus_version: str) -> dict:
        profile: dict = {"corpus_version": corpus_version, "splits": {}}
        for split_name, records in split_records.items():
            profile["splits"][split_name] = {
                "record_count": len(records),
                "char_count": sum(r.char_count for r in records),
                "word_count": sum(r.word_count for r in records),
                "estimated_token_count": sum(r.estimated_token_count for r in records),
            }
        return profile

    @staticmethod
    def _infer_split(path: Path) -> str:
        for part in path.parts:
            if part in ("train", "validation", "test"):
                return part
        return "unknown"

    @staticmethod
    def _infer_format(path: Path) -> str:
        name = path.name
        if name.endswith(".parquet"):
            return "parquet"
        if name.endswith(".jsonl.gz"):
            return "jsonl.gz"
        if name.endswith(".txt.gz"):
            return "txt.gz"
        return path.suffix.lstrip(".")

    @staticmethod
    def _infer_compression(path: Path) -> str:
        if path.name.endswith(".parquet"):
            return "zstd"
        if path.name.endswith(".gz"):
            return "gzip"
        return "none"
