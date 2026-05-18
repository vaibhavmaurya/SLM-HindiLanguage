"""Loads validation reports for all variants, ranks them, and writes a markdown comparison."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from hindi_tokenizer.schema.records import ComparisonResult, ValidationReport, ValidationThresholdValues

if TYPE_CHECKING:
    from hindi_tokenizer.observability.file_registry import FileRegistry
    from hindi_tokenizer.observability.run_logger import TokenizerRunLogger

_PREFERRED_VOCAB_SIZE = 32000


class TokenizerComparator:
    def __init__(
        self,
        report_paths: list[str | Path],
        preferred_vocab_size: int = _PREFERRED_VOCAB_SIZE,
    ) -> None:
        self.report_paths = [Path(p) for p in report_paths]
        self.preferred_vocab_size = preferred_vocab_size

    def compare(
        self,
        output_path: str | Path,
        run_logger: TokenizerRunLogger | None = None,
        file_registry: FileRegistry | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> ComparisonResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if run_logger is not None:
            run_logger.log_event(
                phase="comparison",
                component="tokenizer_comparator",
                status="started",
                notes=f"n_variants={len(self.report_paths)}",
            )

        reports = self._load_reports()
        passing = [r for r in reports if r.passes_thresholds]

        if not passing:
            raise ValueError(
                "All variants failed validation thresholds — cannot recommend a tokenizer. "
                "Review the validation reports and retrain."
            )

        recommended = self._select_recommended(passing)
        result = ComparisonResult(variants=reports, recommended_variant=recommended.variant_name)

        markdown = self._render_markdown(reports, recommended)
        output_path.write_text(markdown, encoding="utf-8")

        if run_logger is not None:
            run_logger.log_event(
                phase="comparison",
                component="tokenizer_comparator",
                status="completed",
                notes=f"recommended={recommended.variant_name}",
            )

        if file_registry is not None:
            file_registry.register_file(path=output_path, role="report", stage="comparison")

        return result

    def _load_reports(self) -> list[ValidationReport]:
        reports: list[ValidationReport] = []
        for path in self.report_paths:
            if not path.exists():
                raise FileNotFoundError(f"Validation report not found: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            # Extract only ValidationReport fields; ignore extra keys written by TokenizerValidator
            vr_fields = {
                "variant_name", "vocab_size", "tokenizer_version", "unk_rate",
                "chars_per_token", "tokens_per_word", "roundtrip_success_rate",
                "devanagari_char_coverage", "special_token_split_failures", "thresholds",
            }
            filtered = {k: v for k, v in data.items() if k in vr_fields}
            if "thresholds" not in filtered:
                filtered["thresholds"] = ValidationThresholdValues().model_dump()
            reports.append(ValidationReport.model_validate(filtered))
        return reports

    def _select_recommended(self, passing: list[ValidationReport]) -> ValidationReport:
        def sort_key(r: ValidationReport) -> tuple[float, int, int]:
            preferred = 0 if r.vocab_size == self.preferred_vocab_size else 1
            return (r.unk_rate, preferred, r.vocab_size)

        return min(passing, key=sort_key)

    def _render_markdown(self, reports: list[ValidationReport], recommended: ValidationReport) -> str:
        lines: list[str] = ["# Tokenizer Variant Comparison\n"]

        header = "| variant | vocab_size | unk_rate | chars_per_token | tokens_per_word | roundtrip_success_rate | passes_thresholds |"
        sep = "|---|---|---|---|---|---|---|"
        lines += [header, sep]

        for r in reports:
            lines.append(
                f"| {r.variant_name} | {r.vocab_size} | {r.unk_rate:.6f} | "
                f"{r.chars_per_token:.3f} | {r.tokens_per_word:.3f} | "
                f"{r.roundtrip_success_rate:.4f} | {r.passes_thresholds} |"
            )

        lines += [
            "",
            "## Recommended Variant",
            "",
            f"**{recommended.variant_name}** — lowest `unk_rate` among variants passing all thresholds.",
        ]
        return "\n".join(lines) + "\n"
