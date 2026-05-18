"""Loads a trained tokenizer artifact and computes quality metrics."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

from hindi_tokenizer.schema.records import ValidationReport, ValidationThresholdValues

if TYPE_CHECKING:
    from hindi_tokenizer.observability.file_registry import FileRegistry
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger

_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_DEVANAGARI_START = "ऀ"
_DEVANAGARI_END = "ॿ"


class TokenizerValidator:
    def __init__(
        self,
        artifact_dir: str | Path,
        validation_sentences: list[str],
        variant_name: str = "unknown",
        tokenizer_version: str = "unknown",
        thresholds: ValidationThresholdValues | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.validation_sentences = validation_sentences
        self.variant_name = variant_name
        self.tokenizer_version = tokenizer_version
        self.thresholds = thresholds or ValidationThresholdValues()

    def validate(
        self,
        report_path: str | Path,
        run_logger: TokenizerRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> ValidationReport:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        if run_logger is not None:
            run_logger.log_event(
                phase="tokenizer_validate",
                component="tokenizer_validator",
                status="started",
                notes=f"variant={self.variant_name}",
            )

        tokenizer = AutoTokenizer.from_pretrained(str(self.artifact_dir))

        all_token_ids: list[int] = []
        all_chars = 0
        all_words = 0
        roundtrip_successes = 0

        for sentence in self.validation_sentences:
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            all_token_ids.extend(ids)
            all_chars += len(sentence)
            all_words += len(sentence.split())
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if decoded.strip() == sentence.strip():
                roundtrip_successes += 1

        n_tokens = len(all_token_ids)
        n_sentences = len(self.validation_sentences)
        unk_id = tokenizer.unk_token_id

        unk_rate = sum(1 for t in all_token_ids if t == unk_id) / n_tokens if n_tokens else 0.0
        chars_per_token = all_chars / n_tokens if n_tokens else 0.0
        tokens_per_word = n_tokens / all_words if all_words else 0.0
        roundtrip_success_rate = roundtrip_successes / n_sentences if n_sentences else 0.0

        failed_tokens: list[str] = []
        for token in _SPECIAL_TOKENS:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                failed_tokens.append(token)

        devanagari_coverage = self._devanagari_coverage(tokenizer, unk_id)

        report = ValidationReport(
            variant_name=self.variant_name,
            vocab_size=len(tokenizer),
            tokenizer_version=self.tokenizer_version,
            unk_rate=unk_rate,
            chars_per_token=chars_per_token,
            tokens_per_word=tokens_per_word,
            roundtrip_success_rate=roundtrip_success_rate,
            devanagari_char_coverage=devanagari_coverage,
            special_token_split_failures=len(failed_tokens),
            thresholds=self.thresholds,
        )

        report_dict = report.model_dump()
        report_dict.update(
            {
                "special_token_failures": failed_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "unk_token_id": tokenizer.unk_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        )
        report_path.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        final_status = "completed" if report.passes_thresholds else "failed"
        if run_logger is not None:
            run_logger.log_event(
                phase="tokenizer_validate",
                component="tokenizer_validator",
                status=final_status,
                notes=f"vocab_size={report.vocab_size}",
            )

        if file_registry is not None:
            file_registry.register_file(path=report_path, role="report", stage="tokenizer_validate")

        return report

    def _devanagari_coverage(self, tokenizer, unk_id: int) -> float:
        devanagari_chars = {
            ch
            for sentence in self.validation_sentences
            for ch in sentence
            if _DEVANAGARI_START <= ch <= _DEVANAGARI_END
        }
        if not devanagari_chars:
            return 1.0
        covered = 0
        for ch in devanagari_chars:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if ids and ids[0] != unk_id:
                covered += 1
        return covered / len(devanagari_chars)
