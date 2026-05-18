"""Assembles the final frozen artifact directory from the selected variant."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindi_tokenizer.observability.file_registry import FileRegistry
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger

_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_metadata.json",
]


class ArtifactPackager:
    def __init__(
        self,
        artifact_dir: str | Path,
        output_dir: str | Path,
        validation_report_path: str | Path,
        comparison_report_path: str | Path,
        training_config_path: str | Path,
        tokenizer_version: str,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.output_dir = Path(output_dir)
        self.validation_report_path = Path(validation_report_path)
        self.comparison_report_path = Path(comparison_report_path)
        self.training_config_path = Path(training_config_path)
        self.tokenizer_version = tokenizer_version

    def package(
        self,
        run_logger: TokenizerRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Path:
        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Source artifact directory not found: {self.artifact_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if run_logger is not None:
            run_logger.log_event(
                phase="packaging",
                component="artifact_packager",
                status="started",
                notes=f"version={self.tokenizer_version}",
            )

        for fname in _TOKENIZER_FILES:
            shutil.copy2(self.artifact_dir / fname, self.output_dir / fname)

        (self.output_dir / "VERSION").write_text(self.tokenizer_version, encoding="utf-8")
        self._write_readme()

        shutil.copy2(self.training_config_path, self.output_dir / "tokenizer_training_config.yaml")
        shutil.copy2(self.validation_report_path, self.output_dir / "tokenizer_validation_report.json")
        shutil.copy2(self.comparison_report_path, self.output_dir / "tokenizer_comparison_report.md")

        if run_logger is not None:
            run_logger.log_event(
                phase="packaging",
                component="artifact_packager",
                status="completed",
                notes=f"output_dir={self.output_dir}",
            )

        if file_registry is not None:
            for p in self.output_dir.iterdir():
                if p.is_file():
                    file_registry.register_file(path=p, role="output", stage="packaging")

        return self.output_dir

    def _write_readme(self) -> None:
        vocab_size = "unknown"
        metadata_path = self.output_dir / "tokenizer_metadata.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                vocab_size = str(meta.get("vocab_size", "unknown"))
            except (json.JSONDecodeError, KeyError):
                pass

        readme = (
            f"# Hindi SLM Tokenizer — {self.tokenizer_version}\n\n"
            f"**Algorithm:** Unigram (subword)\n"
            f"**Vocab size:** {vocab_size}\n"
            f"**Normalizer:** NFKC\n"
            f"**Pre-tokenizer:** Metaspace\n\n"
            f"## Loading\n\n"
            f"```python\n"
            f"from transformers import AutoTokenizer\n"
            f'tokenizer = AutoTokenizer.from_pretrained(".")\n'
            f"```\n"
        )
        (self.output_dir / "README.md").write_text(readme, encoding="utf-8")
