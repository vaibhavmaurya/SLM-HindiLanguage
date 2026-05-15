# CLAUDE.md — Hindi SLM Data Ingestion

## Project Overview

This project builds a clean, compressed, training-ready Hindi language corpus for pretraining a Small Language Model (SLM). The data ingestion pipeline ingests two sources — the AI4Bharat Sangraha HuggingFace dataset and user-provided PDFs — and produces sharded Parquet, JSONL, and TXT outputs suitable for tokenizer training and SLM pretraining.

All computation runs locally (Windows/Linux/macOS laptop or VM). The Ollama + Qwen3 model handles model-assisted PDF text cleaning. No cloud services are required for MVP.

Design reference: https://github.com/vaibhavmaurya/SLM-HindiLanguage/wiki/Hindi-SLM-MVP-%E2%80%90-Data-Ingestion

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| PDF extraction | PyMuPDF (`fitz`) primary, `pdfplumber` fallback |
| Data manipulation | `pandas`, `pyarrow` |
| Schema validation | `pydantic` v2 |
| Config loading | `pydantic-settings` + `PyYAML` |
| Near-deduplication | `datasketch` (MinHash LSH) |
| Model-assisted cleaning | Ollama REST API (`requests`) + Qwen3 |
| CLI | `typer` |
| Testing | `pytest`, `pytest-cov`, `pytest-mock` |
| Logging | Python stdlib `logging` |
| Observability | Custom CSV loggers (`IngestionRunLogger`, `FileRegistry`) |

---

## Directory Conventions

```
configs/          YAML configuration files (one per pipeline concern)
src/slm_hindi/    All production Python code
  config/         Config loading (pydantic-settings)
  schema/         Pydantic data models (CorpusRecord)
  ingestion/      One module per pipeline stage
  observability/  CSV run logger and file registry
  orchestration/  CLI entrypoint that wires all stages
tests/
  fixtures/       Small sample files used by all tests (committed to git)
  unit/           One test file per source module
  integration/    Multi-stage pipeline tests (Ollama mocked unless marked)
data/             All pipeline data — gitignored except sample fixtures
  raw/            Input data (Sangraha download, user PDFs)
  extracted/      PDF extraction outputs
  model_cleaned/  Ollama cleaning outputs + rejected quarantine
  normalized/     After Unicode + whitespace normalization
  filtered/       After Hindi quality filtering
  deduplicated/   After exact + near-dedup
  final/          Sharded train/val/test corpus (Parquet, JSONL, TXT)
  reports/        CSV run logs, file registry, quality reports
```

---

## How to Run Tests

```bash
# All unit tests with coverage
pytest tests/unit/ -v --cov=src/slm_hindi --cov-report=term-missing

# All tests excluding those requiring a live Ollama instance
pytest tests/ -v -m "not requires_ollama" --cov=src/slm_hindi

# Single module test
pytest tests/unit/test_text_normalizer.py -v

# Integration tests (requires Ollama running locally)
pytest tests/integration/ -v -m requires_ollama
```

Coverage target: **≥ 80%** across all source modules.

---

## How to Run the Pipeline

```bash
# Full pipeline (Sangraha + PDFs)
python -m slm_hindi.orchestration.run_ingestion --config configs/ingestion_config.yaml --source all

# Sangraha only
python -m slm_hindi.orchestration.run_ingestion --config configs/ingestion_config.yaml --source sangraha

# PDF only
python -m slm_hindi.orchestration.run_ingestion --config configs/ingestion_config.yaml --source pdf

# Dry run (validates config and file paths, no data written)
python -m slm_hindi.orchestration.run_ingestion --config configs/ingestion_config.yaml --dry-run
```

Or via Makefile:
```bash
make test        # run all tests
make lint        # run ruff linter
make run         # run full pipeline
make run-dry     # dry run
```

---

## Code Style Rules

- **Type hints required** on all function signatures.
- **Pydantic v2** for all data schemas — never bare dicts for structured data crossing module boundaries.
- **No inline comments** unless the WHY is non-obvious (hidden constraint, bug workaround, subtle invariant). Do not describe WHAT the code does.
- **No docstrings** beyond a single short line for public functions — the function name and type hints should be self-documenting.
- All configuration values come from YAML files — **no hardcoded paths, thresholds, or model names** in source code.
- All data written to `data/` — never to `src/` or `tests/` (except fixture generation scripts).

---

## TDD Mandate

Every feature must have tests written **before or alongside** implementation:

1. Write a failing test using sample fixture data from `tests/fixtures/`.
2. Implement the minimum code to make it pass.
3. Refactor if needed, keeping tests green.

Tests never download full Sangraha data or call live Ollama. Mock external calls with `pytest-mock` (`mocker.patch`). Integration tests that require a live Ollama instance are marked `@pytest.mark.requires_ollama` and skipped by default.

---

## Observability

Every pipeline run produces two append-only CSV files in `data/reports/`:

- **`pipeline_run_log.csv`** — one row per activity event (phase start/complete/fail, record counts, duration).
- **`data_file_registry.csv`** — one row per file touched (input, intermediate, output) with path, size, SHA-256, and row count.

These CSVs accumulate across runs. Each run is identified by a UUID `run_id`.

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

1. Create `data/raw/pdf/<source_id>/` directory.
2. Place `original.pdf` and `metadata.json` in it (see `configs/ingestion_config.yaml` for schema).
3. Run: `python -m slm_hindi.orchestration.run_ingestion --source pdf`
