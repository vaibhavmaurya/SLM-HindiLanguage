"""Publishes the frozen tokenizer artifact to Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import HfApi

if TYPE_CHECKING:
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger


class TokenizerPublisher:
    def __init__(self, source_dir: str | Path, repo_id: str) -> None:
        self.source_dir = Path(source_dir)
        self.repo_id = repo_id

    def publish(
        self,
        *,
        dry_run: bool = False,
        run_logger: TokenizerRunLogger | None = None,
    ) -> None:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {self.source_dir}")

        if run_logger is not None:
            run_logger.log_event(
                phase="publishing",
                component="tokenizer_publisher",
                status="started",
                notes=f"repo_id={self.repo_id} dry_run={dry_run}",
            )

        if dry_run:
            if run_logger is not None:
                run_logger.log_event(
                    phase="publishing",
                    component="tokenizer_publisher",
                    status="completed",
                    notes="dry_run=True — skipped HF Hub upload",
                )
            return

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(self.source_dir), repo_id=self.repo_id, repo_type="model")

        if run_logger is not None:
            run_logger.log_event(
                phase="publishing",
                component="tokenizer_publisher",
                status="completed",
                notes=f"repo_id={self.repo_id}",
            )
