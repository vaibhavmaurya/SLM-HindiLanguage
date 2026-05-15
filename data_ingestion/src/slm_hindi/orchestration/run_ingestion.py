"""CLI entrypoint for the Hindi SLM data ingestion pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

import typer

from slm_hindi.ui.progress import console

app = typer.Typer(name="slm-ingest", help="Hindi SLM data ingestion pipeline", add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", help="Path to ingestion_config.yaml")] = Path(
        "configs/ingestion_config.yaml"
    ),
    source: Annotated[
        str, typer.Option("--source", help="Which sources to ingest: sangraha, pdf, wiki, or all")
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
    from slm_hindi.ingestion.wiki_crawler import WikiCrawler  # noqa: PLC0415
    from slm_hindi.observability.file_registry import FileRegistry  # noqa: PLC0415
    from slm_hindi.observability.run_logger import IngestionRunLogger  # noqa: PLC0415
    from slm_hindi.ui.progress import make_progress, setup_logging  # noqa: PLC0415

    settings = load_settings(config)
    setup_logging(settings.project.log_level)
    log = logging.getLogger(__name__)

    run_id = str(uuid.uuid4())
    data_root = Path(settings.project.data_root)

    for _subdir in [
        "raw/pdf",
        "raw/huggingface",
        "extracted/pdf",
        "model_cleaned/pdf",
        "normalized",
        "filtered",
        "deduplicated",
        "final/parquet",
        "final/training_jsonl",
        "final/training_text",
        "reports",
    ]:
        (data_root / _subdir).mkdir(parents=True, exist_ok=True)

    reports_dir = data_root / "reports"

    run_logger = IngestionRunLogger(run_id, reports_dir / "pipeline_run_log.csv")
    file_registry = FileRegistry(run_id, reports_dir / "data_file_registry.csv")

    console.rule("[bold cyan]Hindi SLM Ingestion Pipeline")
    console.print(f"  run_id : [bold]{run_id}[/bold]")
    console.print(f"  source : [bold]{source}[/bold]")
    console.print(f"  config : [bold]{config}[/bold]")
    console.print(f"  dry_run: [bold]{dry_run}[/bold]")
    console.rule()

    if dry_run:
        console.print("[green]Dry run complete.[/green] Config loaded successfully.")
        run_logger.log_event(phase="dry_run", component="run_ingestion", status="completed", notes="dry_run=True")
        return

    # Sangraha is pre-verified and pre-filtered by AI4Bharat — collected separately
    # and merged after quality filtering so its records are never rejected.
    sangraha_records: list = []
    # PDF and Wiki records need normalization + quality filtering.
    unfiltered_records: list = []

    # --- Sangraha source (load only — no normalize, no quality filter) ---
    if source in ("sangraha", "all") and settings.sources.sangraha.enabled:
        log.info("Loading Sangraha dataset…")
        with make_progress() as progress:
            task = progress.add_task("[cyan]Sangraha load", total=settings.sources.sangraha.max_records or 0)
            loader = SangrahaLoader(settings.sources.sangraha, run_id=run_id)
            records = loader.load(
                run_logger=run_logger,
                progress_callback=lambda n: progress.advance(task, n),
            )
        sangraha_records.extend(records)
        console.print(f"  Sangraha: [green]{len(records)} records loaded[/green] (skipping normalize + quality filter)")

    # --- PDF source (Ollama clean → normalize → quality filter) ---
    if source in ("pdf", "all") and settings.sources.pdf.enabled:
        registry = PdfRegistry(settings.sources.pdf.input_dir, settings.sources.pdf.require_metadata_json)
        pdf_sources = registry.discover()
        log.info("Found %d PDF source(s)", len(pdf_sources))

        extractor = PdfExtractor(settings.pdf_extraction, run_id=run_id)
        cleaner = OllamaCleaner(settings.model_cleaning, run_id=run_id)
        validator = CleaningValidator(settings.model_cleaning.validation)

        for pdf_source in pdf_sources:
            console.print(f"[bold]PDF:[/bold] {pdf_source.source_id}")
            extracted = extractor.extract(pdf_source, run_logger=run_logger)

            with make_progress() as progress:
                task = progress.add_task(f"[cyan]Clean PDF ({pdf_source.source_id})", total=len(extracted))
                cleaned = cleaner.clean(extracted, run_logger=run_logger)
                progress.advance(task, len(cleaned))

            passed, quarantined = validator.validate_batch(cleaned, run_logger=run_logger)
            if quarantined:
                quarantine_path = data_root / "model_cleaned" / "pdf" / "rejected_model_outputs.parquet"
                quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                validator.save_quarantine(quarantined, quarantine_path)
                file_registry.register_file(
                    quarantine_path, role="intermediate", stage="cleaning_validate",
                    file_format="parquet", compression="zstd",
                )

            with make_progress() as progress:
                task = progress.add_task(f"[cyan]Normalize PDF ({pdf_source.source_id})", total=len(passed))
                normalizer = TextNormalizer(settings.quality_filter)
                passed = normalizer.normalize_records(
                    passed,
                    run_logger=run_logger,
                    progress_callback=lambda n: progress.advance(task, n),
                )
            unfiltered_records.extend(passed)

    # --- Wiki source (normalize → quality filter) ---
    if source in ("wiki", "all") and settings.sources.wiki.enabled:
        seeds = settings.sources.wiki.seeds
        log.info("Crawling %d Wikipedia seed(s)…", len(seeds))
        crawler = WikiCrawler(settings.wiki_crawl, run_id=run_id)

        for seed in seeds:
            total = seed.max_pages
            with make_progress() as progress:
                task = progress.add_task(f"[cyan]Wiki crawl ({seed.name})", total=total)
                wiki_records = crawler.crawl_seed(
                    seed,
                    run_logger=run_logger,
                    file_registry=file_registry,
                    progress_callback=lambda n: progress.advance(task, n),
                )
            console.print(f"  [green]✓[/green] {seed.name}: {len(wiki_records)} records crawled")

            with make_progress() as progress:
                task = progress.add_task(f"[cyan]Normalize Wiki ({seed.name})", total=len(wiki_records))
                normalizer = TextNormalizer(settings.quality_filter)
                wiki_records = normalizer.normalize_records(
                    wiki_records,
                    run_logger=run_logger,
                    progress_callback=lambda n: progress.advance(task, n),
                )
            unfiltered_records.extend(wiki_records)

    if not sangraha_records and not unfiltered_records:
        console.print("[yellow]No records ingested — nothing to export.[/yellow]")
        return

    # --- Quality filter (PDF + Wiki only) ---
    filtered_records: list = []
    if unfiltered_records:
        with make_progress() as progress:
            task = progress.add_task("[cyan]Quality filter (PDF + Wiki)", total=len(unfiltered_records))
            quality_filter = QualityFilter(settings.quality_filter)
            filtered_records, rejected = quality_filter.filter(
                unfiltered_records,
                run_logger=run_logger,
                progress_callback=lambda n: progress.advance(task, n),
            )
        console.print(
            f"  Quality filter: [green]{len(filtered_records)} passed[/green], "
            f"[red]{len(rejected)} rejected[/red]  (Sangraha bypassed)"
        )

    # Dedup only PDF + Wiki (Sangraha is pre-deduplicated by AI4Bharat)
    if filtered_records:
        with make_progress() as progress:
            task = progress.add_task("[cyan]Deduplication (PDF + Wiki)", total=len(filtered_records))
            deduplicator = Deduplicator()
            filtered_records = deduplicator.deduplicate(
                filtered_records,
                run_logger=run_logger,
                progress_callback=lambda n: progress.advance(task, n),
            )
        console.print(f"  After dedup (PDF/Wiki): [green]{len(filtered_records)} records[/green]")

    all_records = sangraha_records + filtered_records
    console.print(
        f"  Total records: [bold]{len(all_records)}[/bold] "
        f"(Sangraha {len(sangraha_records)} + PDF/Wiki {len(filtered_records)})"
    )

    with make_progress() as progress:
        task = progress.add_task("[cyan]Corpus split", total=len(all_records))
        splitter = CorpusSplitter(settings.export.splits)
        splits = splitter.split(
            all_records,
            run_logger=run_logger,
            progress_callback=lambda n: progress.advance(task, n),
        )

    with make_progress() as progress:
        total_splits = sum(len(r) for r in splits.values())
        task = progress.add_task("[cyan]Export corpus", total=total_splits)
        exporter = CorpusExporter(settings.export, data_root=data_root)
        exporter.export(
            splits,
            run_logger=run_logger,
            file_registry=file_registry,
            progress_callback=lambda n: progress.advance(task, n),
        )

    manifest_gen = ManifestGenerator(settings.export, data_root=data_root)
    manifest_gen.generate(splits, run_logger=run_logger, file_registry=file_registry)

    console.rule("[bold green]Pipeline complete")
    console.print(f"  Run ID : [bold]{run_id}[/bold]")
    console.print(f"  Records: [bold]{len(all_records)}[/bold]")
    for split_name, recs in splits.items():
        console.print(f"    {split_name}: {len(recs)}")
    console.rule()


if __name__ == "__main__":
    app()
