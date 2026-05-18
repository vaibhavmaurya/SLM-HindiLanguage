"""Computes SHA-256 checksums for all files in an artifact directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ChecksumGenerator:
    def generate(self, artifact_dir: str | Path) -> dict[str, str]:
        artifact_dir = Path(artifact_dir)
        checksums: dict[str, str] = {}

        for file_path in sorted(artifact_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name == "checksums.json":
                continue
            rel = file_path.relative_to(artifact_dir).as_posix()
            checksums[rel] = self._sha256(file_path)

        output_path = artifact_dir / "checksums.json"
        output_path.write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")
        return checksums

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
