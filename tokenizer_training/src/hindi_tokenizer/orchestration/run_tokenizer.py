"""Typer CLI for the Hindi SLM tokenizer training pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env.local", override=False)

from hindi_tokenizer.config.settings import load_settings
from hindi_tokenizer.corpus.corpus_sampler import CorpusSampler
from hindi_tokenizer.observability.file_registry import FileRegistry
from hindi_tokenizer.observability.run_logger import TokenizerRunLogger
from hindi_tokenizer.packaging.artifact_packager import ArtifactPackager
from hindi_tokenizer.packaging.checksum_generator import ChecksumGenerator
from hindi_tokenizer.publishing.tokenizer_publisher import TokenizerPublisher
from hindi_tokenizer.training.experiment_runner import ExperimentRunner
from hindi_tokenizer.validation.tokenizer_comparator import TokenizerComparator
from hindi_tokenizer.validation.tokenizer_validator import TokenizerValidator

app = typer.Typer(add_completion=False)

_DEFAULT_CONFIG = Path("configs/tokenizer_training_config.yaml")
_VALID_STEPS = {"sample", "train", "validate", "compare", "package", "publish", "all"}


@app.command()
def main(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", help="Path to training config YAML"),
    step: str = typer.Option("all", "--step", help="Pipeline step: sample|train|validate|compare|package|publish|all"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config; print summary; no writes"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Use smoke-test sample profile (small, fast)"),
    force_retrain: bool = typer.Option(False, "--force-retrain", help="Delete existing artifacts and retrain from scratch"),
) -> None:
    settings = load_settings(config)

    if dry_run:
        typer.echo(f"Project: {settings.project.name}")
        typer.echo(f"Config: {config}")
        return

    if step not in _VALID_STEPS:
        typer.echo(f"Unknown step '{step}'. Choose from: {sorted(_VALID_STEPS)}", err=True)
        raise typer.Exit(code=1)

    run_id = str(uuid.uuid4())
    typer.echo(f"Run ID: {run_id}")

    data_root = Path(settings.project.data_root)
    _ensure_dirs(data_root)

    log_file = data_root / "reports" / "pipeline_run_log.csv"
    registry_file = data_root / "reports" / "data_file_registry.csv"
    run_logger = TokenizerRunLogger(run_id=run_id, log_file=log_file)
    file_registry = FileRegistry(run_id=run_id, registry_file=registry_file)

    vocab_sizes = settings.tokenizer.vocab_sizes
    artifact_base = Path(settings.artifacts.artifact_dir)
    sample_profile = settings.sampling.smoke_test if smoke_test else settings.sampling.experiment
    corpus_file = Path(sample_profile.output_file)

    if step in ("sample", "all"):
        sampler = CorpusSampler(
            input_folder=settings.input.parquet_train_folder,
            text_column=settings.input.text_column,
            file_pattern=settings.input.file_pattern,
            min_char_count=settings.text_filters.min_char_count,
            max_char_count=settings.text_filters.max_char_count,
            min_devanagari_ratio=settings.text_filters.min_devanagari_ratio,
            random_seed=settings.sampling.random_seed,
            corpus_version=settings.project.tokenizer_version,
        )
        sampler.sample(
            target_size_gb=sample_profile.target_size_gb,
            output_file=corpus_file,
            run_logger=run_logger,
            file_registry=file_registry,
        )

    if step in ("train", "all"):
        runner = ExperimentRunner(
            corpus_file=corpus_file,
            output_base_dir=artifact_base,
            vocab_sizes=vocab_sizes,
            corpus_version=settings.project.tokenizer_version,
        )
        runner.run(force_retrain=force_retrain, run_logger=run_logger, file_registry=file_registry)

    reports_dir = data_root / "reports"
    val_report_paths: list[Path] = []

    if step in ("validate", "all"):
        val_sentences_file = Path("tests/fixtures/validation_sentences.txt")
        if val_sentences_file.exists():
            val_sentences = [ln for ln in val_sentences_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            val_sentences = []

        for vocab_size in vocab_sizes:
            variant_dir = artifact_base / f"vocab_{vocab_size}"
            report_path = reports_dir / f"validation_vocab_{vocab_size}.json"
            TokenizerValidator(
                artifact_dir=variant_dir,
                validation_sentences=val_sentences,
                variant_name=f"hindi_unigram_{vocab_size // 1000}k_v001",
                tokenizer_version=settings.project.tokenizer_version,
            ).validate(report_path=report_path, run_logger=run_logger, file_registry=file_registry)
            val_report_paths.append(report_path)
    else:
        for vocab_size in vocab_sizes:
            report_path = reports_dir / f"validation_vocab_{vocab_size}.json"
            if report_path.exists():
                val_report_paths.append(report_path)

    comparison_path = reports_dir / "tokenizer_comparison_report.md"
    recommended_variant: str | None = None

    if step in ("compare", "all") and val_report_paths:
        result = TokenizerComparator(report_paths=val_report_paths).compare(
            output_path=comparison_path,
            run_logger=run_logger,
            file_registry=file_registry,
        )
        recommended_variant = result.recommended_variant

    if step in ("package", "all"):
        if recommended_variant is None:
            recommended_variant = f"hindi_unigram_{vocab_sizes[len(vocab_sizes) // 2] // 1000}k_v001"
        recommended_dir = _resolve_variant_dir(artifact_base, recommended_variant, vocab_sizes)
        val_vocab = _vocab_size_from_name(recommended_variant, default=vocab_sizes[len(vocab_sizes) // 2])
        ArtifactPackager(
            artifact_dir=recommended_dir,
            output_dir=Path(settings.artifacts.final_dir),
            validation_report_path=reports_dir / f"validation_vocab_{val_vocab}.json",
            comparison_report_path=comparison_path,
            training_config_path=config,
            tokenizer_version=settings.project.tokenizer_version,
        ).package(run_logger=run_logger, file_registry=file_registry)
        ChecksumGenerator().generate(Path(settings.artifacts.final_dir))

    if step in ("publish", "all"):
        TokenizerPublisher(
            source_dir=Path(settings.artifacts.final_dir),
            repo_id=settings.project.hf_repo_id or settings.project.tokenizer_version,
        ).publish(run_logger=run_logger)


def _ensure_dirs(data_root: Path) -> None:
    for sub in ("samples/smoke_test", "samples/experiment", "samples/final", "artifacts", "reports", "final"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)


def _resolve_variant_dir(base: Path, variant_name: str, vocab_sizes: list[int]) -> Path:
    vocab = _vocab_size_from_name(variant_name, default=None)
    if vocab is not None:
        candidate = base / f"vocab_{vocab}"
        if candidate.exists():
            return candidate
    for vs in vocab_sizes:
        candidate = base / f"vocab_{vs}"
        if candidate.exists():
            return candidate
    return base / f"vocab_{vocab_sizes[len(vocab_sizes) // 2]}"


def _vocab_size_from_name(name: str, *, default: int | None = None) -> int | None:
    try:
        for part in name.split("_"):
            if part.endswith("k") and part[:-1].isdigit():
                return int(part[:-1]) * 1000
    except (ValueError, AttributeError):
        pass
    return default


if __name__ == "__main__":
    app()
