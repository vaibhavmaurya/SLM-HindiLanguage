"""CLI entrypoint for the Hindi SLM data ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(name="slm-ingest", help="Hindi SLM data ingestion pipeline")


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", help="Path to ingestion_config.yaml")] = Path(
        "configs/ingestion_config.yaml"
    ),
    source: Annotated[
        str, typer.Option("--source", help="Which sources to ingest: sangraha, pdf, or all")
    ] = "all",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate config and paths without writing data")
    ] = False,
) -> None:
    from slm_hindi.config.settings import load_settings  # noqa: PLC0415
    from slm_hindi.ingestion.corpus_exporter import CorpusExporter  # noqa: PLC0415
    from slm_hindi.ingestion.corpus_splitter import CorpusSplitter  # noqa: PLC0415
    from slm_hindi.ingestion.deduplicator import Deduplicator  # noqa: PLC0415
    from slm_hindi.ingestion.manifest_generator import ManifestGenerator  # noqa: PLC0415
    from slm_hindi.ingestion.ollama_cleaner import OllamaCleaner  # noqa: PLC0415
    from slm_hindi.ingestion.cleaning_validator import CleaningValidator  # noqa: PLC0415
    from slm_hindi.ingestion.pdf_extractor import PdfExtractor  # noqa: PLC0415
    from slm_hindi.ingestion.pdf_registry import PdfRegistry  # noqa: PLC0415
    from slm_hindi.ingestion.quality_filter import QualityFilter  # noqa: PLC0415
    from slm_hindi.ingestion.sangraha_loader import SangrahaLoader  # noqa: PLC0415
    from slm_hindi.ingestion.text_normalizer import TextNormalizer  # noqa: PLC0415
    from slm_hindi.observability.file_registry import FileRegistry  # noqa: PLC0415
    from slm_hindi.observability.run_logger import IngestionRunLogger  # noqa: PLC0415

    settings = load_settings(config)
    logging.basicConfig(level=getattr(logging, settings.project.log_level, logging.INFO))
    log = logging.getLogger(__name__)

    run_id = str(uuid.uuid4())
    data_root = Path(settings.project.data_root)
    reports_dir = data_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_logger = IngestionRunLogger(run_id, reports_dir / "pipeline_run_log.csv")
    file_registry = FileRegistry(run_id, reports_dir / "data_file_registry.csv")

    log.info("Pipeline run_id=%s source=%s dry_run=%s", run_id, source, dry_run)

    if dry_run:
        typer.echo(f"Dry run complete. Config loaded from {config}. Run ID: {run_id}")
        run_logger.log_event(phase="dry_run", component="run_ingestion", status="completed", notes="dry_run=True")
        return

    all_records: list = []

    # --- Sangraha source ---
    if source in ("sangraha", "all") and settings.sources.sangraha.enabled:
        loader = SangrahaLoader(settings.sources.sangraha, run_id=run_id)
        records = loader.load(run_logger=run_logger)
        normalizer = TextNormalizer(settings.quality_filter)
        records = normalizer.normalize_records(records, run_logger=run_logger)
        all_records.extend(records)

    # --- PDF source ---
    if source in ("pdf", "all") and settings.sources.pdf.enabled:
        registry = PdfRegistry(settings.sources.pdf.input_dir, settings.sources.pdf.require_metadata_json)
        pdf_sources = registry.discover()

        extractor = PdfExtractor(settings.pdf_extraction, run_id=run_id)
        cleaner = OllamaCleaner(settings.model_cleaning, run_id=run_id)
        validator = CleaningValidator(settings.model_cleaning.validation)

        for pdf_source in pdf_sources:
            extracted = extractor.extract(pdf_source, run_logger=run_logger)
            cleaned = cleaner.clean(extracted, run_logger=run_logger)
            passed, quarantined = validator.validate_batch(cleaned, run_logger=run_logger)
            if quarantined:
                quarantine_path = data_root / "model_cleaned" / "pdf" / "rejected_model_outputs.parquet"
                validator.save_quarantine(quarantined, quarantine_path)
                file_registry.register_file(quarantine_path, role="intermediate", stage="cleaning_validate", file_format="parquet", compression="zstd")
            normalizer = TextNormalizer(settings.quality_filter)
            passed = normalizer.normalize_records(passed, run_logger=run_logger)
            all_records.extend(passed)

    # --- Shared stages ---
    quality_filter = QualityFilter(settings.quality_filter)
    all_records, rejected = quality_filter.filter(all_records, run_logger=run_logger)

    deduplicator = Deduplicator()
    all_records = deduplicator.deduplicate(all_records, run_logger=run_logger)

    splitter = CorpusSplitter(settings.export.splits)
    splits = splitter.split(all_records, run_logger=run_logger)

    exporter = CorpusExporter(settings.export, data_root=data_root)
    exporter.export(splits, run_logger=run_logger, file_registry=file_registry)

    manifest_gen = ManifestGenerator(settings.export, data_root=data_root)
    manifest_gen.generate(splits, run_logger=run_logger, file_registry=file_registry)

    typer.echo(f"Pipeline complete. Run ID: {run_id}")


if __name__ == "__main__":
    app()
