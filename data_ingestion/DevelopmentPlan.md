# Development Plan — Hindi SLM Data Ingestion

**Project:** Hindi Small Language Model — Data Ingestion Workstream  
**Version:** 2.1  
**Date:** 2026-05-15  
**Status:** Implementation Complete — Pipeline Running

---

## 1. Executive Summary

This workstream builds a reproducible, test-driven data ingestion pipeline that produces a clean, compressed, training-ready Hindi corpus. The pipeline ingests three sources — the AI4Bharat Sangraha HuggingFace dataset, user-provided PDF documents, and Hindi Wikipedia — and outputs sharded Parquet, JSONL, and plain-text files ready to load directly into tokenizer training and SLM pretraining code.

The entire system runs locally on a laptop or VM with no cloud dependencies. Ollama + Qwen3 handles model-assisted PDF cleaning. AWS migration paths are noted but out of scope for MVP.

The corpus output contract for downstream phases is documented in `../CORPUS_HANDOFF.md` at the monorepo root.

---

## 2. Architecture Overview

### 2.1 High-Level Pipeline

Sangraha is pre-verified and pre-deduplicated by AI4Bharat, so it bypasses normalization, quality filtering, and deduplication entirely. PDF and Wiki records go through all cleaning stages before being merged with Sangraha for the final split and export.

```
┌──────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────┐
│  AI4Bharat Sangraha  │   │   User-Provided PDFs      │   │  Hindi Wikipedia        │
│  (HuggingFace)       │   │   data/raw/pdf/<id>/      │   │  (MediaWiki Action API) │
└──────────┬───────────┘   └──────────┬───────────────┘   └───────────┬────────────┘
           │                          │                                 │
           ▼                          ▼                                 ▼
  ┌─────────────────┐      ┌──────────────────────┐       ┌────────────────────────┐
  │ SangrahaLoader  │      │ PDF Registry         │       │ WikiCrawler             │
  │ (HF datasets)   │      │ (validate + register)│       │ BFS, depth+page limits  │
  │ dedup_hash set  │      └──────────┬───────────┘       └───────────┬────────────┘
  │ at load time    │                 │                                 │
  └────────┬────────┘                 ▼                                 │
           │               ┌──────────────────────┐                    │
           │  (bypass)     │ PDF Text Extractor   │                    │
           │               │ PyMuPDF / pdfplumber │                    │
           │               └──────────┬───────────┘                    │
           │                          │                                 │
           │                          ▼                                 │
           │               ┌──────────────────────┐                    │
           │               │ Ollama Cleaner       │                    │
           │               │ Qwen3 (local)        │                    │
           │               └──────────┬───────────┘                    │
           │                          │                                 │
           │                          ▼                                 │
           │               ┌──────────────────────┐                    │
           │               │ Cleaning Validator   │                    │
           │               │ (7 checks)           │                    │
           │               └──────────┬───────────┘                    │
           │                          │                                 │
           │                          ▼                                 ▼
           │               ┌─────────────────────────────────────────────┐
           │               │ Text Normalizer  (PDF + Wiki only)           │
           │               │ Unicode NFC + whitespace + danda             │
           │               └──────────────────────┬──────────────────────┘
           │                                       │
           │                                       ▼
           │               ┌─────────────────────────────────────────────┐
           │               │ Quality Filter  (PDF + Wiki only)            │
           │               │ Hindi ratio + length thresholds              │
           │               └──────────────────────┬──────────────────────┘
           │                                       │
           │                                       ▼
           │               ┌─────────────────────────────────────────────┐
           │               │ Deduplicator  (PDF + Wiki only)              │
           │               │ SHA-256 exact + MinHash LSH near-dedup       │
           │               └──────────────────────┬──────────────────────┘
           │                                       │
           └───────────────────────┬───────────────┘
                                   │  merge
                                   ▼
                          ┌────────────────────┐
                          │ Corpus Splitter    │
                          │ 98/1/1 doc-level   │
                          └────────┬───────────┘
                                   │
                                   ▼
                          ┌────────────────────────────────────┐
                          │ Corpus Exporter                    │
                          │ Parquet (zstd) + JSONL.gz + TXT.gz │
                          └────────┬───────────────────────────┘
                                   │
                                   ▼
                          ┌────────────────────┐
                          │ Manifest Generator │
                          │ SHA-256 + profile  │
                          └────────────────────┘
```

### 2.2 Observability Cross-Cut

```
Every stage ──► IngestionRunLogger ──► data/reports/pipeline_run_log.csv
Every file  ──► FileRegistry       ──► data/reports/data_file_registry.csv
```

### 2.3 Wiki Crawler — BFS Design

```
seed URL
    │
    ▼
fetch_extract(title)   ← MediaWiki API: prop=extracts&explaintext=1
    │
    ├── clean_extract()  (strip section headers, collapse blank lines)
    │
    ▼
fetch_links(title)     ← MediaWiki API: prop=links&plnamespace=0
    │
    ├── filter_links()   (namespace prefix check, include/exclude patterns)
    │
    ▼
BFS queue  (visited set prevents cycles; depth + page count limits enforced)
    │
    ▼
_process_page()  ──► list[CorpusRecord]  (one record per paragraph ≥ min_chars)
```

Rate limiting: configurable `delay_between_requests` (default 1.0 s) between API calls.

### 2.4 Rich CLI Architecture

```
run_ingestion.py
    │
    ├── imports console from slm_hindi.ui.progress   (single Console instance)
    │
    ├── setup_logging()  ──► RichHandler → console   (structured log output)
    │
    └── each pipeline stage:
            make_progress() as progress
                task = progress.add_task(...)
                component.method(
                    ...,
                    progress_callback=lambda n: progress.advance(task, n)
                )

All pipeline components:
    method(... progress_callback: Callable[[int], None] | None = None)
        for record in records:
            ...process...
            if progress_callback: progress_callback(1)
```

`rich` is entirely absent from component code — the orchestrator owns all terminal output.

---

## 3. Technology Stack

| Concern | Choice | Rationale |
|---|---|---|
| **PDF extraction** | PyMuPDF (`fitz`) primary, `pdfplumber` fallback | PyMuPDF is 5–10× faster; pdfplumber handles complex tables |
| **Wiki crawling** | MediaWiki Action API (`requests`) | `explaintext=1` gives clean plain text without HTML parsing; no BeautifulSoup needed |
| **Data processing** | `pandas` + `pyarrow` | Sufficient for local MVP; pyarrow gives HuggingFace-compatible zstd Parquet |
| **Schema validation** | `pydantic` v2 | Type-safe record models, JSON serialization, field-level validation |
| **Config** | `pydantic-settings` + `PyYAML` | YAML files for human readability + env var overrides |
| **Near-deduplication** | `datasketch` MinHash LSH | Industry-standard, memory-efficient; no pairwise comparison |
| **Model cleaning** | Ollama REST API + Qwen3 | Fully local, no API key, `temperature=0` for determinism |
| **HTTP client** | `requests` | Simple; all endpoints are local or rate-limited |
| **CLI** | `typer` | Auto-generates `--help`, clean flag definition |
| **Rich terminal UI** | `rich` | Progress bars, structured logging, Unicode-safe on Windows |
| **Testing** | `pytest` + `pytest-cov` + `pytest-mock` | Mock external I/O with `mocker.patch` |
| **Logging** | Python stdlib `logging` + `RichHandler` | Rich-formatted logs share the same console as progress bars |
| **Observability** | Custom CSV appenders | Zero-dependency audit trail that survives process restarts |

### 3.1 Windows UTF-8 Compatibility

Rich box-drawing characters (`─`, `│`) are not in the Windows CP1252 codepage. Fix applied at import time in `ui/progress.py`:

```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
console = Console(legacy_windows=False)
```

A single `Console` instance is shared across the whole application — multiple instances interleave output and break progress bars.

### 3.2 Performance, Availability, Scalability, Cost

| Dimension | Decision |
|---|---|
| **Performance** | PyMuPDF fast PDF parsing; chunked Ollama calls (≤6 000 chars) prevent OOM; zstd Parquet for fast I/O |
| **Availability** | Sequential local pipeline — no distributed failure modes for MVP |
| **Scalability** | Batch size configurable; Sangraha streaming mode available; shard size configurable. Polars + Ray/Dask for scale |
| **Cost** | 100% local — no cloud compute or storage cost. Ollama + Qwen3 runs on 8 GB VRAM |

---

## 4. Project Directory Structure

All paths below are relative to `data_ingestion/` (within the `SLM_HINDI/` monorepo).

```
data_ingestion/
├── CLAUDE.md                          Claude Code initialization (this workstream)
├── DevelopmentPlan.md                 This document
├── action-items.md                    Phase-by-phase task list with completion status
├── Makefile                           Shortcuts: test, lint, run-*, clean
├── pyproject.toml                     Build system (hatchling), pytest config, ruff config
├── requirements.txt                   Production dependencies
├── requirements-dev.txt               Dev/test-only dependencies
├── .env.example                       Environment variable template
├── run_tests.sh / run_tests.bat       Activate venv + run pytest (cross-platform)
├── run_pipeline.sh / run_pipeline.bat Activate venv + run pipeline (cross-platform)
│
├── configs/
│   ├── ingestion_config.yaml          Master pipeline config (sources, runtime)
│   ├── pdf_extraction_config.yaml     PyMuPDF / pdfplumber tuning
│   ├── model_cleaning_config.yaml     Ollama model, chunking, retry settings
│   ├── quality_filter_config.yaml     Hindi ratio thresholds, length bounds
│   ├── export_config.yaml             Shard sizes, split ratios, output formats
│   └── wiki_crawl_config.yaml         BFS depth, rate limits, text length bounds
│
├── src/
│   └── slm_hindi/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py            IngestionSettings (pydantic-settings)
│       ├── schema/
│       │   ├── __init__.py
│       │   └── corpus_record.py       CorpusRecord — unified 30-field schema
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── sangraha_loader.py     Load + map AI4Bharat Sangraha
│       │   ├── pdf_registry.py        Validate + register user PDFs
│       │   ├── pdf_extractor.py       PyMuPDF / pdfplumber extraction
│       │   ├── ollama_cleaner.py      Chunk + clean via Ollama REST
│       │   ├── cleaning_validator.py  7-check validation of cleaned output
│       │   ├── text_normalizer.py     Unicode NFC + whitespace + danda
│       │   ├── quality_filter.py      Hindi ratio + length filter
│       │   ├── deduplicator.py        SHA-256 exact + MinHash near-dedup
│       │   ├── corpus_splitter.py     Document-level 98/1/1 split
│       │   ├── corpus_exporter.py     Sharded Parquet + JSONL.gz + TXT.gz
│       │   ├── manifest_generator.py  SHA-256 manifest + corpus profile
│       │   └── wiki_crawler.py        BFS crawler over Hindi Wikipedia
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── run_logger.py          IngestionRunLogger (CSV append)
│       │   └── file_registry.py       FileRegistry (CSV append + SHA-256)
│       ├── orchestration/
│       │   ├── __init__.py
│       │   └── run_ingestion.py       CLI entrypoint (typer) with rich output
│       └── ui/
│           ├── __init__.py
│           └── progress.py            Shared console singleton, make_progress(), setup_logging()
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    Shared fixtures, pytest marks
│   ├── fixtures/
│   │   ├── sample_pdfs/
│   │   │   └── sample_hindi_2page.pdf
│   │   ├── sample_sangraha/
│   │   │   └── sample_rows.jsonl      5 rows of Sangraha-format data
│   │   ├── sample_metadata/
│   │   │   └── metadata.json
│   │   ├── sample_configs/
│   │   │   └── test_ingestion_config.yaml
│   │   └── sample_wiki/
│   │       ├── sample_extract_response.json   Mocked MediaWiki extracts response
│   │       └── sample_links_response.json     Mocked MediaWiki links response
│   ├── unit/
│   │   ├── test_settings.py
│   │   ├── test_corpus_record.py
│   │   ├── test_run_logger.py
│   │   ├── test_file_registry.py
│   │   ├── test_sangraha_loader.py
│   │   ├── test_pdf_registry.py
│   │   ├── test_pdf_extractor.py
│   │   ├── test_ollama_cleaner.py
│   │   ├── test_cleaning_validator.py
│   │   ├── test_text_normalizer.py
│   │   ├── test_quality_filter.py
│   │   ├── test_deduplicator.py
│   │   ├── test_corpus_splitter.py
│   │   ├── test_corpus_exporter.py
│   │   ├── test_manifest_generator.py
│   │   └── test_wiki_crawler.py       23 tests covering BFS, retry, progress_callback
│   └── integration/
│       ├── test_pdf_pipeline.py
│       ├── test_sangraha_pipeline.py
│       └── test_full_pipeline.py
│
└── data/                              Gitignored (except fixtures)
    ├── raw/
    │   ├── huggingface/sangraha/verified_hin/
    │   └── pdf/
    ├── extracted/pdf/
    ├── model_cleaned/pdf/
    ├── normalized/
    ├── filtered/
    ├── deduplicated/
    ├── final/
    │   ├── parquet/{train,validation,test}/
    │   ├── training_text/{train,validation,test}/
    │   └── training_jsonl/{train,validation,test}/
    └── reports/
```

Monorepo siblings (outside this workstream):

```
SLM_HINDI/
├── CORPUS_HANDOFF.md     Cross-phase reference: data locations, schema, load examples
├── tokenizer_training/   Future workstream (placeholder)
└── slm_training/         Future workstream (placeholder)
```

---

## 5. Component Catalogue

| Module | Class / Function | Responsibility |
|---|---|---|
| `config/settings.py` | `IngestionSettings` | Load and validate all YAML configs |
| `schema/corpus_record.py` | `CorpusRecord` | Unified 30-field pydantic model for all sources |
| `observability/run_logger.py` | `IngestionRunLogger` | Append activity events to `pipeline_run_log.csv` |
| `observability/file_registry.py` | `FileRegistry` | Append file metadata + SHA-256 to `data_file_registry.csv` |
| `ingestion/sangraha_loader.py` | `SangrahaLoader` | Load HuggingFace dataset, map to `CorpusRecord`; sets `dedup_hash`, `cleaning_status="clean"`, `cleaning_method="none"` at load time — no further cleaning applied |
| `ingestion/pdf_registry.py` | `PdfRegistry` | Discover and validate PDF folders + `metadata.json` |
| `ingestion/pdf_extractor.py` | `PdfExtractor` | Extract text per-page via PyMuPDF / pdfplumber |
| `ingestion/ollama_cleaner.py` | `OllamaCleaner` | Chunk text, call Ollama REST, reassemble output |
| `ingestion/cleaning_validator.py` | `CleaningValidator` | 7-check validation; quarantine failures |
| `ingestion/text_normalizer.py` | `TextNormalizer` | NFC unicode, whitespace, danda, quote, URL normalization |
| `ingestion/quality_filter.py` | `QualityFilter` | Hindi character ratio + length thresholds |
| `ingestion/deduplicator.py` | `Deduplicator` | SHA-256 exact dedup + MinHash LSH near-dedup |
| `ingestion/corpus_splitter.py` | `CorpusSplitter` | Document-level stratified 98/1/1 train/val/test split |
| `ingestion/corpus_exporter.py` | `CorpusExporter` | Write sharded Parquet (zstd), JSONL.gz, TXT.gz |
| `ingestion/manifest_generator.py` | `ManifestGenerator` | Generate SHA-256 manifest + corpus profile JSON |
| `ingestion/wiki_crawler.py` | `WikiCrawler` | BFS crawler over Hindi Wikipedia via MediaWiki API |
| `orchestration/run_ingestion.py` | `app` (typer) | CLI wiring: instantiate components, run in order, rich output |
| `ui/progress.py` | `console`, `make_progress()`, `setup_logging()` | Shared Rich console; progress bar factory; logging setup |

### progress_callback Protocol

Every component method that iterates over records accepts:

```python
progress_callback: Callable[[int], None] | None = None
```

The orchestrator passes `lambda n: progress.advance(task, n)`. Components call `progress_callback(n)` where `n` is the number of records just processed. This keeps `rich` entirely out of component code.

---

## 6. Data Schema — Unified Corpus Record

All records from all sources conform to this schema (`CorpusRecord` pydantic model in `schema/corpus_record.py`):

| Field | Type | Description |
|---|---|---|
| `record_id` | `str` | Unique record identifier (UUID) |
| `document_id` | `str` | Parent document identifier |
| `paragraph_id` | `str` | Paragraph/block index within document |
| `source_type` | `"huggingface_dataset" \| "pdf" \| "wiki"` | Origin of the record |
| `source_name` | `str` | e.g. `ai4bharat/sangraha`, `user_provided_pdfs`, `hi.wikipedia.org` |
| `source_dataset` | `str \| None` | HuggingFace dataset path (Sangraha only) |
| `source_file_name` | `str \| None` | Original PDF filename (PDF source only) |
| `source_url_or_path` | `str \| None` | Wikipedia article URL or PDF file path |
| `page_number` | `int \| None` | PDF page number (PDF source only) |
| `raw_text` | `str` | Original unmodified text |
| `cleaned_text` | `str \| None` | After model-assisted cleaning (PDF only) |
| `final_text` | `str` | **The training text** — normalized, validated |
| `language` | `str` | ISO 639-1 code, always `hi` |
| `script` | `str` | Always `Devanagari` |
| `char_count` | `int` | Character count of `final_text` |
| `word_count` | `int` | Whitespace-split word count |
| `estimated_token_count` | `int` | ~4 chars/token estimate |
| `devanagari_ratio` | `float` | Fraction of Devanagari characters in `final_text` |
| `latin_ratio` | `float` | Fraction of Latin characters |
| `digit_ratio` | `float` | Fraction of digit characters |
| `symbol_ratio` | `float` | Fraction of symbol characters |
| `quality_score` | `float` | Composite quality score (0.0–1.0) |
| `cleaning_method` | `"deterministic_normalization" \| "ollama_model_assisted" \| "none"` | How text was cleaned |
| `cleaning_model` | `str \| None` | Ollama model name (PDF only) |
| `cleaning_model_version` | `str \| None` | Model version tag |
| `cleaning_status` | `"pending" \| "clean" \| "quarantined" \| "skipped"` | Outcome of cleaning validation |
| `dedup_hash` | `str` | SHA-256 of `final_text` (exact dedup key) |
| `near_dedup_cluster_id` | `str \| None` | MinHash LSH cluster ID |
| `split_name` | `"train" \| "validation" \| "test" \| None` | Corpus split assignment |
| `created_at` | `str` | ISO-8601 UTC timestamp |
| `ingestion_run_id` | `str` | UUID of the pipeline run that produced this record |

---

## 7. Configuration Strategy

Six YAML files, each scoped to a single pipeline concern:

| File | Purpose |
|---|---|
| `configs/ingestion_config.yaml` | Master switches: source enables, seeds, batch size, random seed, corpus version |
| `configs/pdf_extraction_config.yaml` | Primary/fallback engine, OCR flag, min page chars |
| `configs/model_cleaning_config.yaml` | Ollama endpoint, model, chunking params, validation thresholds |
| `configs/quality_filter_config.yaml` | Min/max char count, min Devanagari ratio, URL handling |
| `configs/export_config.yaml` | Output formats (Parquet/JSONL/TXT), compression, shard size, split ratios |
| `configs/wiki_crawl_config.yaml` | BFS depth/page limits, rate delay, min paragraph chars, retry settings |

All configs are loaded once at startup by `IngestionSettings` and injected into components. Components never read environment variables or files directly.

---

## 8. Storage Design

### 8.1 System-of-Record (Parquet)

```
data/final/parquet/
  train/       hindi_corpus_v001_train_00000.parquet  (≤512 MB/shard, zstd)
  validation/  hindi_corpus_v001_validation_00000.parquet
  test/        hindi_corpus_v001_test_00000.parquet
```

Directly loadable via `datasets.load_dataset("parquet", data_files=...)`.

### 8.2 Tokenizer Training Text

```
data/final/training_text/
  train/    hindi_corpus_v001_train_00000.txt.gz  (one paragraph per line)
```

### 8.3 Training JSONL

```
data/final/training_jsonl/
  train/    hindi_corpus_v001_train_00000.jsonl.gz
```

Each JSONL line: `{"text": "...", "source_type": "...", "document_id": "...", "record_id": "..."}`.

### 8.4 Manifest and Profile

```
data/final/
  hindi_corpus_v001_manifest.json   SHA-256 per output file + metadata
  hindi_corpus_v001_profile.json    Total chars, words, records per split/source
```

---

## 9. Observability Design

### 9.1 `data/reports/pipeline_run_log.csv`

Append-only CSV capturing every pipeline event. Written by `IngestionRunLogger`.

Columns: `run_id`, `timestamp`, `phase`, `component`, `source_id`, `record_id`, `status`, `records_in`, `records_out`, `records_rejected`, `duration_seconds`, `error_message`, `notes`

**Status values:** `started` | `completed` | `skipped` | `failed` | `quarantined`

### 9.2 `data/reports/data_file_registry.csv`

Append-only CSV capturing every file touched. Written by `FileRegistry`.

Columns: `run_id`, `timestamp`, `role`, `stage`, `source_id`, `file_path`, `file_name`, `format`, `size_bytes`, `row_count`, `sha256`, `compression`, `notes`

**Role values:** `input` | `intermediate` | `output` | `report`

Both files accumulate across all pipeline runs — never truncated.

---

## 10. Testing Strategy

### 10.1 Principles

- **TDD first:** Tests written before or alongside implementation.
- **Sample data only:** Tests never download Sangraha or call live Ollama.
- **One test file per source module:** `tests/unit/test_<module>.py`.
- **Mock external I/O:** `datasets.load_dataset`, `requests.post`, and `requests.Session.get` always mocked in unit tests.
- **Integration tests:** Full sub-pipeline on sample fixtures with Ollama mocked (or live with `@pytest.mark.requires_ollama`).

### 10.2 Test Fixtures

| Fixture | Purpose |
|---|---|
| `sample_hindi_2page.pdf` | Two-page PDF with Hindi text for extraction tests |
| `sample_rows.jsonl` | 5 rows of Sangraha-format data for loader tests |
| `metadata.json` | Valid PDF source metadata for registry tests |
| `test_ingestion_config.yaml` | Minimal config pointing to fixture paths |
| `sample_extract_response.json` | Mocked MediaWiki API response for page text |
| `sample_links_response.json` | Mocked MediaWiki API response for page links |

### 10.3 Coverage

**139 tests, ≥ 83% coverage** as of implementation completion. Run with:

```bash
./run_tests.sh                  # Linux/macOS
run_tests.bat                   # Windows
pytest tests/ -v -m "not requires_ollama" --cov=src/slm_hindi
```

---

## 11. Phase Overview

| Phase | Deliverable | Status |
|---|---|---|
| **1** | Project scaffold + tooling | ✅ Complete |
| **2** | Config + schema layer + observability | ✅ Complete |
| **3** | Sangraha loader | ✅ Complete |
| **4** | PDF registry + extractor | ✅ Complete |
| **5** | Ollama cleaner + cleaning validator | ✅ Complete |
| **6** | Text normalizer + quality filter | ✅ Complete |
| **7** | Deduplicator + corpus splitter | ✅ Complete |
| **8** | Corpus exporter + manifest + orchestration CLI | ✅ Complete |
| **9** | Wikipedia crawler (BFS, MediaWiki API) | ✅ Complete |
| **10** | Rich terminal UI + progress bars + monorepo reorganization | ✅ Complete |
| **11** | Pipeline source-track refinements + setup_and_run.bat | ✅ Complete |

---

## 12. Future AWS Migration Notes

| Concern | Local MVP | AWS Path |
|---|---|---|
| Storage | Local filesystem `data/` | S3 — replace path strings with `s3://` URIs via `s3fs` |
| Data processing | `pandas` | Polars or Spark on EMR/Glue |
| Orchestration | `run_ingestion.py` CLI | AWS Step Functions or Airflow on MWAA |
| Model cleaning | Ollama local | SageMaker Endpoint (Qwen3 on g5.xlarge) |
| Wiki crawling | Local BFS | Distributed crawl with Scrapy on ECS |
| Observability CSVs | Local files | CloudWatch + S3 Parquet audit logs |
| Near-dedup | `datasketch` local | Spark LSH at scale |

The config-driven design (no hardcoded paths) and modular `progress_callback` pattern make this migration incremental — swap storage backends and parallelism without changing business logic.
