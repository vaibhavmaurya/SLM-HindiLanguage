# Development Plan — Hindi SLM Data Ingestion

**Project:** Hindi Small Language Model — Data Ingestion Workstream  
**Version:** 1.0  
**Date:** 2026-05-15  
**Status:** Active Development  

---

## 1. Executive Summary

This workstream builds a reproducible, test-driven data ingestion pipeline that produces a clean, compressed, training-ready Hindi corpus. The pipeline ingests two sources — the AI4Bharat Sangraha HuggingFace dataset and user-provided PDF documents — and outputs sharded Parquet, JSONL, and plain-text files ready to load directly into tokenizer training and SLM pretraining code.

The entire system runs locally on a laptop or VM with no cloud dependencies. Ollama + Qwen3 handles model-assisted PDF cleaning. AWS migration paths are noted but out of scope for MVP.

---

## 2. Architecture Overview

### 2.1 High-Level Pipeline

```
┌──────────────────────┐          ┌──────────────────────────┐
│  AI4Bharat Sangraha  │          │   User-Provided PDFs      │
│  (HuggingFace)       │          │   data/raw/pdf/<id>/      │
└──────────┬───────────┘          └──────────┬───────────────┘
           │                                  │
           ▼                                  ▼
  ┌─────────────────┐              ┌──────────────────────┐
  │ SangrahaLoader  │              │ PDF Registry         │
  │ (HF datasets)   │              │ (validate + register)│
  └────────┬────────┘              └──────────┬───────────┘
           │                                  │
           │                                  ▼
           │                       ┌──────────────────────┐
           │                       │ PDF Text Extractor   │
           │                       │ PyMuPDF / pdfplumber │
           │                       └──────────┬───────────┘
           │                                  │
           │                                  ▼
           │                       ┌──────────────────────┐
           │                       │ Ollama Cleaner       │
           │                       │ Qwen3 (local)        │
           │                       └──────────┬───────────┘
           │                                  │
           │                                  ▼
           │                       ┌──────────────────────┐
           │                       │ Cleaning Validator   │
           │                       │ (7 checks)           │
           │                       └──────────┬───────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │ Text Normalizer    │
               │ Unicode + WS + ।  │
               └────────┬───────────┘
                        │
                        ▼
               ┌────────────────────┐
               │ Quality Filter     │
               │ Hindi ratio, len   │
               └────────┬───────────┘
                        │
                        ▼
               ┌────────────────────┐
               │ Deduplicator       │
               │ SHA-256 + MinHash  │
               └────────┬───────────┘
                        │
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

---

## 3. Technology Stack

| Concern | Choice | Rationale |
|---|---|---|
| **PDF extraction** | PyMuPDF (`fitz`) primary, `pdfplumber` fallback | PyMuPDF is 5–10× faster and handles most PDFs; pdfplumber handles complex table layouts better |
| **Data processing** | `pandas` + `pyarrow` | Pandas is sufficient for local MVP; pyarrow gives HuggingFace-compatible Parquet with zstd compression. Polars can replace pandas for AWS scale. |
| **Schema validation** | `pydantic` v2 | Type-safe record models, JSON serialization, field-level validation |
| **Config** | `pydantic-settings` + `PyYAML` | YAML files for human readability + environment variable overrides for CI/CD |
| **Near-deduplication** | `datasketch` MinHash LSH | Industry-standard, memory-efficient for local; does not require full pairwise comparison |
| **Model cleaning** | Ollama REST API + Qwen3 | Fully local, no API key, conservative cleaning with `temperature=0` for determinism |
| **HTTP client** | `requests` | Simple; Ollama is local-only so no async needed |
| **CLI** | `typer` | Auto-generates `--help`, clean flag definition, minimal boilerplate |
| **Testing** | `pytest` + `pytest-cov` + `pytest-mock` | Standard; mock Ollama via `mocker.patch("requests.post")` |
| **Logging** | Python stdlib `logging` | No extra dependency; structured logging (structlog) can be added for AWS |
| **Observability** | Custom CSV appenders | Zero-dependency, human-readable audit trail that survives process restarts |

### 3.1 Performance, Availability, Scalability, Cost Trade-offs

| Dimension | Decision |
|---|---|
| **Performance** | PyMuPDF for fast PDF parsing; chunked Ollama calls (≤6 000 chars) prevent OOM; zstd Parquet for fast I/O |
| **Availability** | Sequential local pipeline — no distributed failure modes for MVP |
| **Scalability** | Batch size configurable; streaming mode available for Sangraha; shard size configurable for large corpora. Polars + Ray/Dask are natural next steps for scale |
| **Cost** | 100% local — no cloud compute or storage cost during development. Ollama + Qwen3 runs on consumer hardware (8 GB VRAM minimum for 8B model) |

---

## 4. Project Directory Structure

```
SLM_HINDI/
├── CLAUDE.md                          Claude Code initialization
├── DevelopmentPlan.md                 This document
├── action-items.md                    Phase-by-phase task list
├── Makefile                           Shortcuts: test, lint, run
├── pyproject.toml                     Build system, pytest config, ruff config
├── requirements.txt                   Production dependencies
├── requirements-dev.txt               Dev/test-only dependencies
├── .env.example                       Environment variable template
├── .gitignore
│
├── configs/
│   ├── ingestion_config.yaml          Master pipeline config
│   ├── pdf_extraction_config.yaml     PDF extraction settings
│   ├── model_cleaning_config.yaml     Ollama + Qwen3 settings
│   ├── quality_filter_config.yaml     Hindi quality thresholds
│   └── export_config.yaml            Parquet/JSONL/TXT export settings
│
├── src/
│   └── slm_hindi/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py            Pydantic-settings config loader
│       ├── schema/
│       │   ├── __init__.py
│       │   └── corpus_record.py       CorpusRecord pydantic model (unified schema)
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
│       │   └── manifest_generator.py  SHA-256 manifest + corpus profile
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── run_logger.py          IngestionRunLogger (CSV append)
│       │   └── file_registry.py       FileRegistry (CSV append + SHA-256)
│       └── orchestration/
│           ├── __init__.py
│           └── run_ingestion.py       CLI entrypoint (typer)
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
│   │   └── sample_configs/
│   │       └── test_ingestion_config.yaml
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
│   │   └── test_manifest_generator.py
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

---

## 5. Component Catalogue

| Module | Class / Function | Responsibility |
|---|---|---|
| `config/settings.py` | `IngestionSettings` | Load and validate all YAML configs via pydantic-settings |
| `schema/corpus_record.py` | `CorpusRecord` | Unified 30-field pydantic model for all records regardless of source |
| `observability/run_logger.py` | `IngestionRunLogger` | Append activity events to `pipeline_run_log.csv` |
| `observability/file_registry.py` | `FileRegistry` | Append file metadata to `data_file_registry.csv`, compute SHA-256 |
| `ingestion/sangraha_loader.py` | `SangrahaLoader` | Stream/load HuggingFace dataset, map to `CorpusRecord` |
| `ingestion/pdf_registry.py` | `PdfRegistry` | Discover and validate PDF folders + `metadata.json` |
| `ingestion/pdf_extractor.py` | `PdfExtractor` | Extract text per-page via PyMuPDF with pdfplumber fallback |
| `ingestion/ollama_cleaner.py` | `OllamaCleaner` | Chunk text, call Ollama REST, reassemble cleaned output |
| `ingestion/cleaning_validator.py` | `CleaningValidator` | Run 7 validation checks; quarantine failures |
| `ingestion/text_normalizer.py` | `TextNormalizer` | NFC unicode, whitespace, danda, quote, URL normalization |
| `ingestion/quality_filter.py` | `QualityFilter` | Apply Hindi character ratio + length thresholds |
| `ingestion/deduplicator.py` | `Deduplicator` | SHA-256 exact dedup + MinHash LSH near-dedup |
| `ingestion/corpus_splitter.py` | `CorpusSplitter` | Document-level stratified 98/1/1 train/val/test split |
| `ingestion/corpus_exporter.py` | `CorpusExporter` | Write sharded Parquet (zstd), JSONL.gz, TXT.gz |
| `ingestion/manifest_generator.py` | `ManifestGenerator` | Generate SHA-256 manifest + corpus profile JSON |
| `orchestration/run_ingestion.py` | `app` (typer) | CLI wiring: instantiate components, run in order |

---

## 6. Data Schema — Unified Corpus Record

All records from all sources conform to this schema (defined as a pydantic `CorpusRecord`):

| Field | Type | Description |
|---|---|---|
| `record_id` | `str` | Unique record identifier |
| `document_id` | `str` | Parent document identifier |
| `paragraph_id` | `str` | Paragraph/block index within document |
| `source_type` | `str` | `huggingface_dataset` or `pdf` |
| `source_name` | `str` | e.g. `ai4bharat/sangraha` or `user_provided_pdfs` |
| `source_dataset` | `str \| None` | HuggingFace dataset path if applicable |
| `source_file_name` | `str \| None` | Original PDF filename |
| `source_url_or_path` | `str \| None` | URL or file path |
| `page_number` | `int \| None` | PDF page number |
| `raw_text` | `str` | Original unmodified text |
| `cleaned_text` | `str \| None` | After model-assisted cleaning |
| `final_text` | `str` | Final text used in corpus |
| `language` | `str` | ISO 639-1 code, default `hi` |
| `script` | `str` | e.g. `Devanagari` |
| `char_count` | `int` | Character count of `final_text` |
| `word_count` | `int` | Word count of `final_text` |
| `estimated_token_count` | `int` | Estimated tokens (char_count / 4.5) |
| `devanagari_ratio` | `float` | Fraction of Devanagari chars in `final_text` |
| `latin_ratio` | `float` | Fraction of Latin chars |
| `digit_ratio` | `float` | Fraction of digit chars |
| `symbol_ratio` | `float` | Fraction of symbol chars |
| `quality_score` | `float` | Composite quality score (0–1) |
| `cleaning_method` | `str` | `deterministic_normalization` or `ollama_model_assisted` |
| `cleaning_model` | `str \| None` | e.g. `qwen3` or `null` for Sangraha |
| `cleaning_model_version` | `str \| None` | Model version tag |
| `cleaning_status` | `str` | `clean`, `quarantined`, `skipped` |
| `dedup_hash` | `str` | SHA-256 of `final_text` |
| `near_dedup_cluster_id` | `str \| None` | MinHash LSH cluster ID |
| `split_name` | `str \| None` | `train`, `validation`, `test` |
| `created_at` | `str` | ISO-8601 datetime |
| `ingestion_run_id` | `str` | UUID of the pipeline run |

---

## 7. Configuration Strategy

Five YAML files, each scoped to a single pipeline concern:

| File | Purpose |
|---|---|
| `configs/ingestion_config.yaml` | Master switches: source enables, batch size, random seed, corpus version |
| `configs/pdf_extraction_config.yaml` | Primary/fallback engine, OCR flag, min page chars |
| `configs/model_cleaning_config.yaml` | Ollama endpoint, model, chunking params, validation thresholds |
| `configs/quality_filter_config.yaml` | Min/max char count, min Devanagari ratio, URL handling |
| `configs/export_config.yaml` | Output formats (Parquet/JSONL/TXT), compression, shard size, split ratios |

All configs are loaded once at startup by `IngestionSettings` (pydantic-settings) and injected into components. Components never read environment variables or files directly.

---

## 8. Storage Design

### 8.1 System-of-Record (Parquet)

```
data/final/parquet/
  train/       hindi_corpus_v001_train_00000.parquet  (≤512 MB/shard, zstd)
  validation/  hindi_corpus_v001_validation_00000.parquet
  test/         hindi_corpus_v001_test_00000.parquet
```

Directly loadable via HuggingFace `datasets.load_dataset("parquet", data_files=...)`.

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

### 8.4 Manifest

```
data/reports/
  hindi_corpus_v001_manifest.json   SHA-256 per output file + source summary
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

Both files accumulate across all pipeline runs, enabling historical trend analysis and audit.

---

## 10. Testing Strategy

### 10.1 Principles

- **TDD first:** Tests are written before or alongside implementation.
- **Sample data only:** Tests never download Sangraha or call live Ollama.
- **One test file per source module:** `tests/unit/test_<module>.py`.
- **Mock external I/O:** HuggingFace `load_dataset` and `requests.post` are always mocked in unit tests.
- **Integration tests:** Run the full sub-pipeline on sample fixtures with Ollama mocked (or live with `@pytest.mark.requires_ollama`).

### 10.2 Test Fixtures

| Fixture | Purpose |
|---|---|
| `sample_hindi_2page.pdf` | Two-page PDF with Hindi text for extraction tests |
| `sample_rows.jsonl` | 5 rows of Sangraha-format data for loader tests |
| `metadata.json` | Valid PDF source metadata for registry tests |
| `test_ingestion_config.yaml` | Minimal config pointing to fixture paths |

### 10.3 Coverage Target

**≥ 80%** per module. Run with `pytest --cov=src/slm_hindi --cov-report=term-missing`.

---

## 11. Phase Overview

| Phase | Deliverable | Acceptance Criteria |
|---|---|---|
| **1** | Project scaffold + tooling | `pytest --collect-only` succeeds; `make test` runs |
| **2** | Config + schema layer | Config loads from YAML; CorpusRecord validates sample dict |
| **3** | Sangraha loader | Loads 5 mocked rows; maps to CorpusRecord correctly |
| **4** | PDF registry + extractor | Registers sample PDF; extracts 2 pages with correct char count |
| **5** | Ollama cleaner + validator | Mocked clean passes; all 7 reject cases tested |
| **6** | Normalizer + quality filter | Known noisy string normalizes correctly; boundary cases pass/reject |
| **7** | Deduplicator + splitter | Known duplicates removed; split ratios ±0.5% of 98/1/1 |
| **8** | Exporter + manifest + CLI | Parquet roundtrip valid; manifest checksums correct; CLI --help works |

---

## 12. Future AWS Migration Notes

These are out of scope for MVP but inform design decisions made now:

| Concern | Local MVP | AWS Path |
|---|---|---|
| Storage | Local filesystem `data/` | S3 — replace path strings with `s3://` URIs via `s3fs` |
| Data processing | `pandas` | Polars or Spark on EMR/Glue |
| Orchestration | `run_ingestion.py` CLI | AWS Step Functions or Airflow on MWAA |
| Model cleaning | Ollama local | SageMaker Endpoint (Qwen3 on g5.xlarge) |
| Observability CSVs | Local files | CloudWatch + S3 parquet audit logs |
| Near-dedup | `datasketch` local | Deduplicate at scale with Spark LSH |

The config-driven design (no hardcoded paths) and modular component boundaries make this migration incremental — swap storage backends and parallelism without changing business logic.
