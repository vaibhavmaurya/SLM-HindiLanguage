"""Reads text records from a folder of Parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger


class ParquetReader:
    def __init__(
        self,
        folder: str | Path,
        text_column: str = "final_text",
        file_pattern: str = "*.parquet",
        run_logger: TokenizerRunLogger | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.text_column = text_column
        self.file_pattern = file_pattern
        self._run_logger = run_logger

    def read_texts(self, path: str | Path | None = None) -> list[str]:
        if path is not None:
            texts = self._read_file(Path(path))
        else:
            texts = self._read_all()

        if self._run_logger is not None:
            self._run_logger.log_event(
                phase="parquet_read",
                component="parquet_reader",
                status="completed",
                records_out=len(texts),
            )
        return texts

    def _read_all(self) -> list[str]:
        if not self.folder.exists():
            raise FileNotFoundError(f"Input folder not found: {self.folder}")
        files = sorted(self.folder.glob(self.file_pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{self.file_pattern}' in: {self.folder}")
        texts: list[str] = []
        for f in files:
            texts.extend(self._read_file(f))
        return texts

    def _read_file(self, path: Path) -> list[str]:
        df = pd.read_parquet(path)
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in {path.name}. Available: {list(df.columns)}")
        texts: list[str] = []
        for val in df[self.text_column]:
            if val is None or not isinstance(val, str):
                continue
            texts.append(val)
        return texts
