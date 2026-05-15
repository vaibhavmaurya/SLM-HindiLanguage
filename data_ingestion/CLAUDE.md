# CLAUDE.md — Data Ingestion Workstream

## Project Overview

This workstream builds a clean, compressed, training-ready Hindi language corpus for pretraining a Small Language Model. The pipeline ingests three sources — AI4Bharat Sangraha (HuggingFace), user-provided PDFs, and Hindi Wikipedia — and produces sharded Parquet, JSONL, and TXT outputs ready for tokenizer training and SLM pretraining.

All computation runs locally. Ollama + Qwen3 handles model-assisted PDF text cleaning. No cloud services are required for MVP.

Design reference: https://github.com/vaibhavmaurya/SLM-HindiLanguage/wiki/Hindi-SLM-MVP-%E2%80%90-Data-Ingestion

The corpus output contract for downstream phases is documented in `../CORPUS_HANDOFF.md` at the project root.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| PDF extraction | PyMuPDF (`fitz`) primary, `pdfplumber` fallback |
| Wiki crawling | MediaWiki Action API (plain text via `explaintext=1`) |
| Data manipulation | `pandas`, `pyarrow` |
| Schema validation | `pydantic` v2 |
| Config loading | `pydantic-settings` + `PyYAML` |
| Near-deduplication | `datasketch` (MinHash LSH) |
| Model-assisted cleaning | Ollama REST API (`requests`) + Qwen3 |
| CLI | `typer` |
| Rich terminal UI | `rich` (Console, Progress, RichHandler) |
| Testing | `pytest`, `pytest-cov`, `pytest-mock` |
| Logging | Python stdlib `logging` + Rich handler |
| Observability | Custom CSV loggers (`IngestionRunLogger`, `FileRegistry`) |

---

## Directory Conventions

All paths below are relative to `data_ingestion/`.

```
configs/              YAML configuration files (one per pipeline concern)
  ingestion_config.yaml       Master config — sources, runtime settings
  pdf_extraction_config.yaml  PyMuPDF / pdfplumber tuning
  model_cleaning_config.yaml  Ollama model, chunking, retry settings
  quality_filter_config.yaml  Hindi ratio thresholds, length bounds
  export_config.yaml          Shard sizes, split ratios, output formats
  wiki_crawl_config.yaml      BFS depth, rate limits, text length bounds

src/slm_hindi/        All production Python code
  config/             Config loading (pydantic-settings)
  schema/             Pydantic data models (CorpusRecord)
  ingestion/          One module per pipeline stage
    sangraha_loader.py
    pdf_registry.py
    pdf_extractor.py
    ollama_cleaner.py
    cleaning_validator.py
    text_normalizer.py
    quality_filter.py
    deduplicator.py
    corpus_splitter.py
    corpus_exporter.py
    manifest_generator.py
    wiki_crawler.py
  observability/      CSV run logger and file registry
    run_logger.py
    file_registry.py
  orchestration/      CLI entrypoint that wires all stages
    run_ingestion.py
  ui/                 Rich terminal utilities (shared console singleton)
    progress.py

tests/
  fixtures/           Small sample files committed to git
    sample_pdfs/
    sample_sangraha/
    sample_metadata/
    sample_configs/
    sample_wiki/
  unit/               One test file per source module
  integration/        Multi-stage tests (Ollama mocked unless marked)

data/                 All pipeline data — gitignored except sample fixtures
  raw/                Input data (Sangraha download cache, user PDFs)
  extracted/          PDF extraction outputs
  model_cleaned/      Ollama cleaning outputs + rejected quarantine
  normalized/         After Unicode + whitespace normalization
  filtered/           After Hindi quality filter
  deduplicated/       After exact + near-dedup
  final/              Sharded train/val/test corpus
    parquet/train/ validation/ test/
    training_jsonl/train/ validation/ test/
    training_text/train/ validation/ test/
  reports/            CSV run logs, file registry, quality reports
```

---

## How to Run Tests

From the `data_ingestion/` directory:

```bash
# Convenience scripts (activate venv automatically)
./run_tests.sh                         # Linux/macOS
run_tests.bat                          # Windows

# Or directly with pytest
pytest tests/unit/ -v --cov=src/slm_hindi --cov-report=term-missing

# All tests excluding live Ollama
pytest tests/ -v -m "not requires_ollama" --cov=src/slm_hindi

# Single module
pytest tests/unit/test_wiki_crawler.py -v

# Integration tests (requires Ollama running)
pytest tests/integration/ -v -m requires_ollama
```

Coverage target: **≥ 80%** across all source modules.

---

## How to Run the Pipeline

From the `data_ingestion/` directory:

```bash
# Convenience scripts (pass any flags through)
./run_pipeline.sh --source all         # Linux/macOS
./run_pipeline.sh --source sangraha
./run_pipeline.sh --source pdf
./run_pipeline.sh --source wiki
./run_pipeline.sh --dry-run
run_pipeline.bat --source all          # Windows

# Or directly with python
python -m slm_hindi.orchestration.run_ingestion \
    --config configs/ingestion_config.yaml --source all

# Or via Makefile
make run            # all sources
make run-sangraha
make run-pdf
make run-wiki
make run-dry
make test
make lint
make lint-fix
```

---

## Code Style Rules

- **Type hints required** on all function signatures.
- **Pydantic v2** for all data schemas — never bare dicts for structured data crossing module boundaries.
- **No inline comments** unless the WHY is non-obvious (hidden constraint, bug workaround, subtle invariant).
- **No docstrings** beyond a single short line for public functions.
- All configuration values from YAML files — **no hardcoded paths, thresholds, or model names** in source code.
- All data written to `data/` — never to `src/` or `tests/`.
- Progress reporting via `progress_callback: Callable[[int], None] | None = None` on all pipeline methods — keeps `rich` out of component code.
- Import `console` from `slm_hindi.ui.progress` — never create a second `Console()` instance.

---

## TDD Mandate

1. Write a failing test using sample fixture data from `tests/fixtures/`.
2. Implement the minimum code to make it pass.
3. Refactor keeping tests green.

Tests never download full Sangraha data or call live Ollama. Mock external calls with `pytest-mock`. Integration tests requiring Ollama are marked `@pytest.mark.requires_ollama` and skipped by default.

---

## Observability

Every pipeline run appends to two CSV files in `data/reports/`:

- **`pipeline_run_log.csv`** — one row per activity event (phase, component, status, record counts, duration, errors).
- **`data_file_registry.csv`** — one row per file produced/consumed (path, size, SHA-256, row count, compression).

Each run is identified by a UUID `run_id`. CSVs accumulate across runs; never truncated.

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

```bash
OLLAMA_ENDPOINT=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen3
DATA_ROOT=data
CORPUS_VERSION=hindi_corpus_v001
```

---

## Adding a New PDF Source

1. Create `data/raw/pdf/<source_id>/`.
2. Place `original.pdf` and `metadata.json` (see `configs/ingestion_config.yaml` for schema).
3. Run: `./run_pipeline.sh --source pdf`

## Adding a New Wiki Seed

Append an entry to the `sources.wiki.seeds` list in `configs/ingestion_config.yaml`:

```yaml
- url: "https://hi.wikipedia.org/wiki/रामायण"
  name: "ramayana"
  category: "epic_literature"
  follow_links: true
  max_depth: 1
  max_pages: 80
```
