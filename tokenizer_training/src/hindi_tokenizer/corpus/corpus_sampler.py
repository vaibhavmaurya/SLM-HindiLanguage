"""Samples, filters, and normalizes Hindi text from Parquet files into a plain-text corpus file."""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from hindi_tokenizer.corpus.parquet_reader import ParquetReader
from hindi_tokenizer.schema.records import SampleManifest

if TYPE_CHECKING:
    from hindi_tokenizer.observability.file_registry import FileRegistry
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    deva = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    non_space = sum(1 for c in text if not c.isspace())
    return deva / non_space if non_space > 0 else 0.0


def normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    collapsed = " ".join(normalized.split())
    return collapsed


def is_valid_hindi_text(
    text: str,
    min_char_count: int = 30,
    max_char_count: int = 5000,
    min_devanagari_ratio: float = 0.60,
) -> bool:
    if not text or not text.strip():
        return False
    t = text.strip()
    if len(t) < min_char_count or len(t) > max_char_count:
        return False
    return devanagari_ratio(t) >= min_devanagari_ratio


class CorpusSampler:
    def __init__(
        self,
        input_folder: str | Path,
        text_column: str = "final_text",
        file_pattern: str = "*.parquet",
        min_char_count: int = 30,
        max_char_count: int = 5000,
        min_devanagari_ratio: float = 0.60,
        random_seed: int = 42,
        corpus_version: str = "unknown",
    ) -> None:
        self.input_folder = Path(input_folder)
        self.text_column = text_column
        self.file_pattern = file_pattern
        self.min_char_count = min_char_count
        self.max_char_count = max_char_count
        self.min_devanagari_ratio = min_devanagari_ratio
        self.random_seed = random_seed
        self.corpus_version = corpus_version

    def sample(
        self,
        target_size_gb: float,
        output_file: str | Path,
        run_logger: TokenizerRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> SampleManifest:
        output_path = Path(output_file)
        target_bytes = int(target_size_gb * 1024**3)

        if run_logger is not None:
            run_logger.log_event(phase="corpus_sample", component="corpus_sampler", status="started")

        reader = ParquetReader(self.input_folder, self.text_column, self.file_pattern)
        raw_texts = reader.read_texts()

        valid: list[str] = []
        for t in raw_texts:
            normalized = normalize_unicode(t)
            if is_valid_hindi_text(normalized, self.min_char_count, self.max_char_count, self.min_devanagari_ratio):
                clean = normalized.replace("\n", " ").replace("\r", " ")
                valid.append(clean)

        rng = random.Random(self.random_seed)
        rng.shuffle(valid)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        written_records = 0
        written_bytes = 0

        with output_path.open("w", encoding="utf-8") as f:
            for i, text in enumerate(valid):
                line = text + "\n"
                f.write(line)
                written_bytes += len(line.encode("utf-8"))
                written_records += 1
                if progress_callback is not None:
                    progress_callback(i + 1)
                if written_bytes >= target_bytes:
                    break

        actual_size_gb = written_bytes / (1024**3)

        manifest = SampleManifest(
            version=self.corpus_version,
            source_folder=str(self.input_folder),
            text_column=self.text_column,
            target_size_gb=target_size_gb,
            actual_size_gb=actual_size_gb,
            written_records=written_records,
            random_seed=self.random_seed,
        )

        if run_logger is not None:
            run_logger.log_event(
                phase="corpus_sample",
                component="corpus_sampler",
                status="completed",
                records_out=written_records,
            )

        if file_registry is not None:
            file_registry.register_file(path=output_path, role="output", stage="corpus_sample")

        return manifest
