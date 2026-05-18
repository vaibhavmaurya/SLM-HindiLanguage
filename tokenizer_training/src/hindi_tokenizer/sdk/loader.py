"""Loads a Hindi SLM tokenizer from a local artifact directory."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer


def load_hindi_slm_tokenizer(artifact_dir: str | Path) -> AutoTokenizer:
    path = Path(artifact_dir)
    if not path.exists():
        raise OSError(f"Artifact directory not found: {path}")
    return AutoTokenizer.from_pretrained(str(path))
