# Action Items — Hindi SLM Data Ingestion

**Development methodology:** Test-Driven Development (TDD). For every component:  
1. Write failing tests using sample fixtures.  
2. Implement minimum code to pass.  
3. Refactor keeping tests green.

Each phase ends with `pytest tests/unit/test_<module>.py -v --cov=src/slm_hindi` — all tests must pass before proceeding.

---

## Phase 1 — Project Scaffold & Tooling

**Goal:** Runnable project skeleton with dependency management, linting, and an empty-but-collectible test suite.

**Acceptance Criteria:**
- `pytest --collect-only` succeeds (no import errors)
- `make test` runs and reports 0 failures
- `make lint` runs ruff with no errors
- Sample fixture files exist and are non-empty

### Tasks

- [x] 1.1 Create full directory tree (see `DevelopmentPlan.md §4`)
- [x] 1.2 Create `.gitignore`
- [x] 1.3 Create `CLAUDE.md`
- [ ] 1.4 Create `pyproject.toml` — build system (`hatchling`), pytest settings, ruff config
- [ ] 1.5 Create `requirements.txt` — production dependencies with pinned minor versions
- [ ] 1.6 Create `requirements-dev.txt` — dev/test dependencies
- [ ] 1.7 Create `.env.example` — all required env vars with defaults
- [ ] 1.8 Create `Makefile` — targets: `test`, `lint`, `run`, `run-dry`, `clean`
- [ ] 1.9 Create `tests/fixtures/sample_sangraha/sample_rows.jsonl` — 5 sample Sangraha rows
- [ ] 1.10 Create `tests/fixtures/sample_metadata/metadata.json` — valid PDF metadata
- [ ] 1.11 Create `tests/fixtures/sample_configs/test_ingestion_config.yaml`
- [ ] 1.12 Generate `tests/fixtures/sample_pdfs/sample_hindi_2page.pdf` programmatically
- [ ] 1.13 Create `tests/conftest.py` — shared fixtures, custom pytest marks
- [ ] 1.14 Create all `__init__.py` stubs under `src/slm_hindi/`
- [ ] 1.15 Verify: `pip install -e ".[dev]"` and `pytest --collect-only`

**Verify:**
```bash
pip install -e ".[dev]"
pytest --collect-only
make lint
```

---

## Phase 2 — Config & Schema Layer

**Goal:** Type-safe configuration loading from YAML files and a validated pydantic data model for all corpus records.

**Acceptance Criteria:**
- `IngestionSettings` loads from `test_ingestion_config.yaml` without error
- `CorpusRecord` validates a sample dict and rejects invalid inputs
- `IngestionRunLogger` appends correct CSV rows
- `FileRegistry` appends correct CSV rows with SHA-256

### Tasks

- [ ] 2.1 Write tests: `tests/unit/test_settings.py`
  - Load config from sample YAML
  - Assert nested field values (e.g. `settings.sources.sangraha.enabled == True`)
  - Assert validation error on missing required field
- [ ] 2.2 Write tests: `tests/unit/test_corpus_record.py`
  - Validate a fully-populated `CorpusRecord` dict
  - Assert defaults populate correctly
  - Assert `ValidationError` on bad type (e.g. `char_count="abc"`)
- [ ] 2.3 Write tests: `tests/unit/test_run_logger.py`
  - `log_event()` appends one CSV row
  - Row contains all expected columns
  - Multiple calls append multiple rows (not overwrite)
- [ ] 2.4 Write tests: `tests/unit/test_file_registry.py`
  - `register_file()` appends one row with correct `file_name` and `sha256`
  - Non-existent file raises `FileNotFoundError`
- [ ] 2.5 Implement `configs/ingestion_config.yaml`
- [ ] 2.6 Implement `configs/pdf_extraction_config.yaml`
- [ ] 2.7 Implement `configs/model_cleaning_config.yaml`
- [ ] 2.8 Implement `configs/quality_filter_config.yaml`
- [ ] 2.9 Implement `configs/export_config.yaml`
- [ ] 2.10 Implement `src/slm_hindi/config/settings.py` (`IngestionSettings`)
- [ ] 2.11 Implement `src/slm_hindi/schema/corpus_record.py` (`CorpusRecord`)
- [ ] 2.12 Implement `src/slm_hindi/observability/run_logger.py` (`IngestionRunLogger`)
- [ ] 2.13 Implement `src/slm_hindi/observability/file_registry.py` (`FileRegistry`)

**Verify:**
```bash
pytest tests/unit/test_settings.py tests/unit/test_corpus_record.py \
       tests/unit/test_run_logger.py tests/unit/test_file_registry.py -v \
       --cov=src/slm_hindi/config --cov=src/slm_hindi/schema --cov=src/slm_hindi/observability
```

---

## Phase 3 — Sangraha Loader

**Goal:** Load AI4Bharat Sangraha records from HuggingFace and map them to the unified `CorpusRecord` schema.

**Acceptance Criteria:**
- 5 mocked Sangraha rows load and map to `CorpusRecord` objects without error
- `source_type`, `cleaning_method`, `language` fields populated correctly
- `IngestionRunLogger` receives `started` and `completed` events

### Tasks

- [ ] 3.1 Write tests: `tests/unit/test_sangraha_loader.py`
  - Mock `datasets.load_dataset` to return 5 rows from `sample_rows.jsonl`
  - Assert all 5 rows produce valid `CorpusRecord` objects
  - Assert `source_type == "huggingface_dataset"`
  - Assert `cleaning_method == "deterministic_normalization"`
  - Assert `cleaning_model is None`
  - Assert logger receives `started` then `completed` events
- [ ] 3.2 Implement `src/slm_hindi/ingestion/sangraha_loader.py` (`SangrahaLoader`)
  - `load(run_logger=None) -> list[CorpusRecord]`
  - Supports streaming mode (configurable)
  - Maps HuggingFace row fields to `CorpusRecord`
  - Logs `sangraha_load` phase events

**Verify:**
```bash
pytest tests/unit/test_sangraha_loader.py -v --cov=src/slm_hindi/ingestion/sangraha_loader
```

---

## Phase 4 — PDF Registry & Extractor

**Goal:** Discover and validate user-provided PDFs; extract text per-page using PyMuPDF with pdfplumber fallback.

**Acceptance Criteria:**
- `PdfRegistry.discover()` finds the sample PDF folder and validates `metadata.json`
- `PdfExtractor.extract()` extracts 2 pages from the sample PDF
- Each page record has correct `page_number`, `char_count`, `extraction_method`
- Missing `metadata.json` raises a `ValueError`

### Tasks

- [ ] 4.1 Write tests: `tests/unit/test_pdf_registry.py`
  - Valid folder with `original.pdf` + `metadata.json` → returns 1 `PdfSource`
  - Folder missing `metadata.json` → raises `ValueError`
  - Folder missing `original.pdf` → raises `FileNotFoundError`
  - `metadata.json` with wrong schema → raises `ValidationError`
- [ ] 4.2 Write tests: `tests/unit/test_pdf_extractor.py`
  - Extract `sample_hindi_2page.pdf` → 2 page records
  - Each record has `page_number`, `raw_text`, `char_count ≥ 10`
  - `extraction_method == "pymupdf"`
  - Force fallback: mock fitz to raise exception → `extraction_method == "pdfplumber"`
- [ ] 4.3 Implement `src/slm_hindi/ingestion/pdf_registry.py` (`PdfRegistry`, `PdfSource`, `PdfMetadata`)
  - `discover(input_dir) -> list[PdfSource]`
  - Validates folder structure and `metadata.json` via pydantic
- [ ] 4.4 Implement `src/slm_hindi/ingestion/pdf_extractor.py` (`PdfExtractor`)
  - `extract(pdf_source) -> list[CorpusRecord]` stubs (raw_text populated)
  - Primary: `fitz.open()` per page
  - Fallback: `pdfplumber.open()` if fitz raises
  - Sets `ocr_used=False` (OCR out of scope for MVP)

**Verify:**
```bash
pytest tests/unit/test_pdf_registry.py tests/unit/test_pdf_extractor.py -v \
       --cov=src/slm_hindi/ingestion/pdf_registry \
       --cov=src/slm_hindi/ingestion/pdf_extractor
```

---

## Phase 5 — Ollama Cleaner & Cleaning Validator

**Goal:** Chunk extracted PDF text, clean via Ollama REST API, and validate each cleaned output against 7 quality checks.

**Acceptance Criteria:**
- `OllamaCleaner.clean()` calls `requests.post` once per chunk
- Chunking splits text at paragraph boundaries (≤6 000 chars, 200-char overlap)
- All 7 validator checks independently reject the right inputs
- Failed records are quarantined (not discarded) with `cleaning_status="quarantined"`

### Tasks

- [ ] 5.1 Write tests: `tests/unit/test_ollama_cleaner.py`
  - Mock `requests.post` to return `{"response": "साफ हिंदी पाठ।"}`
  - Assert one API call per chunk for 10 000-char input (≥2 chunks expected)
  - Assert overlap: last 200 chars of chunk N appear in chunk N+1
  - Simulate timeout → assert retry (max 3 attempts with backoff)
  - Simulate 500 response → raises `OllamaError`
- [ ] 5.2 Write tests: `tests/unit/test_cleaning_validator.py`
  - Empty output → reject
  - Output length < 50% of input → reject
  - Output length > 120% of input → reject
  - Devanagari ratio < 0.60 → reject
  - Prompt echo in output → reject
  - Repeated lines (≥3 duplicates) → reject
  - Output is entirely English → reject
  - Valid Hindi output → accept
- [ ] 5.3 Implement `src/slm_hindi/ingestion/ollama_cleaner.py` (`OllamaCleaner`, `OllamaError`)
  - `clean(records: list[CorpusRecord]) -> list[CorpusRecord]`
  - Chunk each record's `raw_text` with overlap
  - POST to Ollama with `temperature=0.0`
  - Exponential backoff on `requests.Timeout` (3 retries, base 2s)
  - Sets `cleaned_text` on each record
- [ ] 5.4 Implement `src/slm_hindi/ingestion/cleaning_validator.py` (`CleaningValidator`)
  - `validate(record: CorpusRecord) -> tuple[CorpusRecord, bool]`
  - On pass: set `cleaning_status="clean"`; on fail: set `cleaning_status="quarantined"`, keep `cleaned_text`
  - `save_quarantine(records, path)` → writes `rejected_model_outputs.parquet`

**Verify:**
```bash
pytest tests/unit/test_ollama_cleaner.py tests/unit/test_cleaning_validator.py -v \
       --cov=src/slm_hindi/ingestion/ollama_cleaner \
       --cov=src/slm_hindi/ingestion/cleaning_validator
```

---

## Phase 6 — Text Normalizer & Quality Filter

**Goal:** Apply deterministic text normalization (Unicode, whitespace, punctuation) and filter records by Hindi language quality thresholds.

**Acceptance Criteria:**
- Unicode NFC applied; Devanagari danda `।` preserved
- Extra whitespace and repeated newlines collapsed
- Records below Devanagari ratio threshold are rejected
- Records below/above char count bounds are rejected
- All thresholds configurable from YAML

### Tasks

- [ ] 6.1 Write tests: `tests/unit/test_text_normalizer.py`
  - Input with NFD chars → output is NFC
  - Input with multiple spaces → single space
  - Input with repeated newlines → max 2 newlines preserved
  - Danda `।` preserved and not replaced
  - Input with repeated header line (`line\nline\nline`) → deduplicated
  - URL in text → removed (or kept) per config
  - Decorative symbols removed
- [ ] 6.2 Write tests: `tests/unit/test_quality_filter.py`
  - Record with 80% Devanagari → passes (default threshold 0.60)
  - Record with 30% Devanagari → rejected
  - Record with `char_count < 30` → rejected
  - Record with `char_count > 1_000_000` → rejected (upper bound)
  - Record flagged as table fragment → rejected
- [ ] 6.3 Implement `src/slm_hindi/ingestion/text_normalizer.py` (`TextNormalizer`)
  - `normalize(text: str) -> str`
  - Unicode `unicodedata.normalize("NFC", text)`
  - Whitespace: collapse spaces/tabs, limit consecutive newlines to 2
  - Danda: never strip `।`
  - Repeated line dedup: remove any line appearing ≥3 times consecutively
  - URL removal if `quality_filter_config.remove_urls == true`
- [ ] 6.4 Implement `src/slm_hindi/ingestion/quality_filter.py` (`QualityFilter`)
  - `filter(records: list[CorpusRecord]) -> tuple[list[CorpusRecord], list[CorpusRecord]]`
  - Returns `(passed, rejected)` tuples
  - Computes `devanagari_ratio`, `char_count`, `word_count` on `final_text`
  - Sets `quality_score` as `devanagari_ratio * 0.7 + length_score * 0.3`

**Verify:**
```bash
pytest tests/unit/test_text_normalizer.py tests/unit/test_quality_filter.py -v \
       --cov=src/slm_hindi/ingestion/text_normalizer \
       --cov=src/slm_hindi/ingestion/quality_filter
```

---

## Phase 7 — Deduplicator & Corpus Splitter

**Goal:** Remove exact and near-duplicate records; split the corpus into train/validation/test at document level.

**Acceptance Criteria:**
- Exact duplicates (identical SHA-256) removed; one copy kept
- Near-duplicates (MinHash Jaccard ≥ 0.85) assigned same `near_dedup_cluster_id`; one copy per cluster kept
- Split ratios within ±0.5% of 98/1/1 on a 1 000-record test set
- Split is document-level (all paragraphs of one document go to the same split)

### Tasks

- [ ] 7.1 Write tests: `tests/unit/test_deduplicator.py`
  - 3 identical records → 1 kept, 2 removed
  - 2 near-identical records (99% overlap) → 1 kept
  - 5 distinct records → all 5 kept
  - `dedup_hash` field populated on all records
  - `near_dedup_cluster_id` populated on near-dups
- [ ] 7.2 Write tests: `tests/unit/test_corpus_splitter.py`
  - 100 records from 10 documents → split ratios ~98/1/1
  - All paragraphs of same `document_id` end up in same split
  - `split_name` field set on each record
  - `random_seed=42` produces deterministic result
- [ ] 7.3 Implement `src/slm_hindi/ingestion/deduplicator.py` (`Deduplicator`)
  - `deduplicate(records) -> list[CorpusRecord]`
  - Pass 1: group by `dedup_hash` (SHA-256 of `final_text`), keep first
  - Pass 2: MinHash LSH with `datasketch` (num_perm=128, threshold=0.85, 3-gram shingles)
  - Assign `near_dedup_cluster_id` (UUID) to each cluster
- [ ] 7.4 Implement `src/slm_hindi/ingestion/corpus_splitter.py` (`CorpusSplitter`)
  - `split(records) -> dict[str, list[CorpusRecord]]`
  - Group by `document_id`; shuffle documents with seed
  - Assign 98% of documents to train, 1% to validation, 1% to test
  - Set `split_name` on each record

**Verify:**
```bash
pytest tests/unit/test_deduplicator.py tests/unit/test_corpus_splitter.py -v \
       --cov=src/slm_hindi/ingestion/deduplicator \
       --cov=src/slm_hindi/ingestion/corpus_splitter
```

---

## Phase 8 — Exporter, Manifest & Orchestration

**Goal:** Write the final corpus in all output formats, generate SHA-256 manifest and profile, and wire everything into a runnable CLI.

**Acceptance Criteria:**
- Parquet files round-trip: write then `pd.read_parquet()` returns same data
- JSONL.gz files contain valid JSON on every line with `text` field
- TXT.gz files are non-empty
- `manifest.json` SHA-256 values match actual files
- `corpus_profile.json` reports correct record/word counts per split
- `python -m slm_hindi.orchestration.run_ingestion --help` works
- `--dry-run` exits 0 without writing data files
- `pipeline_run_log.csv` and `data_file_registry.csv` populated after a run

### Tasks

- [ ] 8.1 Write tests: `tests/unit/test_corpus_exporter.py`
  - Write 10 records to temp dir → Parquet roundtrip matches input
  - Write 10 records → JSONL.gz has 10 lines, each parseable JSON with `text` key
  - Write 10 records → TXT.gz is non-empty
  - Shard size respected: 5 MB limit with 100-record dataset → multiple shards
  - `FileRegistry.register_file()` called for each output file
- [ ] 8.2 Write tests: `tests/unit/test_manifest_generator.py`
  - SHA-256 in manifest matches `hashlib.sha256(file.read_bytes())`
  - `corpus_version` matches config
  - `profile.json` totals match sum of per-split counts
- [ ] 8.3 Write tests: `tests/integration/test_pdf_pipeline.py`
  - Registry → Extract → Clean (mocked) → Validate → Normalize → Filter
  - Input: `sample_hindi_2page.pdf`
  - Assert at least 1 record passes all stages
- [ ] 8.4 Write tests: `tests/integration/test_sangraha_pipeline.py`
  - Load (mocked) → Normalize → Filter → Dedup
  - Input: `sample_rows.jsonl`
  - Assert all 5 rows process without error
- [ ] 8.5 Write tests: `tests/integration/test_full_pipeline.py`
  - End-to-end with both sources mocked
  - Assert final Parquet exists and is loadable
  - Assert `pipeline_run_log.csv` has `completed` rows for each phase
  - Assert `data_file_registry.csv` has entries for all output files
- [ ] 8.6 Implement `src/slm_hindi/ingestion/corpus_exporter.py` (`CorpusExporter`)
  - `export(split_records: dict[str, list[CorpusRecord]], output_dir: Path)`
  - Write sharded Parquet (`pyarrow`, zstd, 512 MB target)
  - Write sharded JSONL.gz
  - Write sharded TXT.gz
  - Register each output file with `FileRegistry`
- [ ] 8.7 Implement `src/slm_hindi/ingestion/manifest_generator.py` (`ManifestGenerator`)
  - `generate(output_dir, corpus_version) -> dict`
  - Walk output files, compute SHA-256, write `manifest.json`
  - Compute word/char/record counts per split, write `profile.json`
- [ ] 8.8 Implement `src/slm_hindi/orchestration/run_ingestion.py`
  - `typer` app with commands: main pipeline
  - Flags: `--config PATH`, `--source [sangraha|pdf|all]`, `--dry-run`
  - Instantiates `IngestionRunLogger` and `FileRegistry` with UUID `run_id`
  - Passes logger/registry into every component
  - Catches exceptions, logs to `pipeline_run_log.csv` with `status=failed`

**Verify:**
```bash
pytest tests/ -v -m "not requires_ollama" --cov=src/slm_hindi --cov-report=term-missing
python -m slm_hindi.orchestration.run_ingestion --help
python -m slm_hindi.orchestration.run_ingestion --config configs/ingestion_config.yaml --dry-run
```

---

## Completion Checklist

- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] All integration tests pass (Ollama mocked): `pytest tests/integration/ -v -m "not requires_ollama"`
- [ ] Coverage ≥ 80%: `pytest --cov=src/slm_hindi --cov-fail-under=80`
- [ ] Linting clean: `make lint`
- [ ] `DevelopmentPlan.md` up to date with final design
- [ ] `data/reports/pipeline_run_log.csv` and `data_file_registry.csv` generated after a dry run
- [ ] `data/final/` output loadable via `datasets.load_dataset("parquet", data_files=...)`
