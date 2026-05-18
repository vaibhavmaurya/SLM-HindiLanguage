"""Trains one TokenizerTrainer per vocab-size variant and manages artifact directories."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from hindi_tokenizer.training.tokenizer_trainer import TokenizerTrainer

if TYPE_CHECKING:
    from hindi_tokenizer.observability.file_registry import FileRegistry
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger


class ExperimentRunner:
    def __init__(
        self,
        corpus_file: str | Path,
        output_base_dir: str | Path,
        vocab_sizes: list[int],
        corpus_version: str = "unknown",
    ) -> None:
        self.corpus_file = Path(corpus_file)
        self.output_base_dir = Path(output_base_dir)
        self.vocab_sizes = vocab_sizes
        self.corpus_version = corpus_version

    def run(
        self,
        force_retrain: bool = False,
        run_logger: TokenizerRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[Path]:
        artifact_dirs: list[Path] = []

        for vocab_size in self.vocab_sizes:
            variant_dir = self.output_base_dir / f"vocab_{vocab_size}"
            tokenizer_json = variant_dir / "tokenizer.json"

            if tokenizer_json.exists() and not force_retrain:
                artifact_dirs.append(variant_dir)
                continue

            if force_retrain and variant_dir.exists():
                shutil.rmtree(variant_dir)

            if run_logger is not None:
                run_logger.log_event(
                    phase="experiment_run",
                    component="experiment_runner",
                    status="started",
                    notes=f"vocab_size={vocab_size}",
                )

            TokenizerTrainer(vocab_size=vocab_size, corpus_version=self.corpus_version).train(
                corpus_file=self.corpus_file,
                output_dir=variant_dir,
                file_registry=file_registry,
            )

            if run_logger is not None:
                run_logger.log_event(
                    phase="experiment_run",
                    component="experiment_runner",
                    status="completed",
                    notes=f"vocab_size={vocab_size}",
                )

            artifact_dirs.append(variant_dir)

        return artifact_dirs
