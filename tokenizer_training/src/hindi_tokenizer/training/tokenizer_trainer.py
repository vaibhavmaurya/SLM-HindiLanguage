"""Trains a Unigram+NFKC+Metaspace tokenizer and writes an HF-compatible artifact directory."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from hindi_tokenizer.observability.file_registry import FileRegistry
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger

# IDs 0-7 are frozen; UnigramTrainer places special_tokens at the front of the vocab in order.
_SPECIAL_TOKENS = [
    "<pad>",        # 0
    "<unk>",        # 1
    "<s>",          # 2
    "</s>",         # 3
    "<|system|>",   # 4
    "<|user|>",     # 5
    "<|assistant|>",# 6
    "<|end|>",      # 7
]


class TokenizerTrainer:
    def __init__(
        self,
        vocab_size: int = 32000,
        corpus_version: str = "unknown",
    ) -> None:
        self.vocab_size = vocab_size
        self.corpus_version = corpus_version

    def train(
        self,
        corpus_file: str | Path,
        output_dir: str | Path,
        run_logger: TokenizerRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Path:
        corpus_path = Path(corpus_file)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if run_logger is not None:
            run_logger.log_event(
                phase="tokenizer_train",
                component="tokenizer_trainer",
                status="started",
                notes=f"vocab_size={self.vocab_size}",
            )

        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        tokenizer.decoder = decoders.Metaspace()

        trainer = trainers.UnigramTrainer(
            vocab_size=self.vocab_size,
            special_tokens=_SPECIAL_TOKENS,
            unk_token="<unk>",
        )
        tokenizer.train([str(corpus_path)], trainer=trainer)

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
        )
        hf_tokenizer.save_pretrained(str(out_dir))

        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
        }
        (out_dir / "special_tokens_map.json").write_text(
            json.dumps(special_tokens_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        metadata = {
            "algorithm": "unigram",
            "vocab_size": self.vocab_size,
            "normalizer": "NFKC",
            "pre_tokenizer": "Metaspace",
            "corpus_version": self.corpus_version,
            "special_tokens": _SPECIAL_TOKENS,
        }
        (out_dir / "tokenizer_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if run_logger is not None:
            run_logger.log_event(
                phase="tokenizer_train",
                component="tokenizer_trainer",
                status="completed",
                notes=f"vocab_size={self.vocab_size}",
            )

        if file_registry is not None:
            file_registry.register_file(
                path=out_dir / "tokenizer.json",
                role="output",
                stage="tokenizer_train",
            )

        return out_dir
