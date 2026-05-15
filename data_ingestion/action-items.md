# Action Items — Hindi SLM Data Ingestion

**Development methodology:** Test-Driven Development (TDD). For every component:  
1. Write failing tests using sample fixtures.  
2. Implement minimum code to pass.  
3. Refactor keeping tests green.

Each phase ends with `pytest tests/unit/test_<module>.py -v --cov=src/slm_hindi` — all tests must pass before proceeding.

**Status as of 2026-05-15:** All 10 phases complete. 131 tests passing. 83% coverage. Pipeline executed successfully against live data.

---

## Phase 1 — Project Scaffold & Tooling ✅

**Goal:** Runnable project skeleton with dependency management, linting, and an empty-but-collectible test suite.

- [x] 1.1 Create full directory tree
- [x] 1.2 Create `.gitignore` (updated to `data_ingestion/data/` paths after monorepo reorganization)
- [x] 1.3 Create `CLAUDE.md` (updated to v2 reflecting current state)
- [x] 1.4 Create `pyproject.toml` — hatchling build, pytest settings, ruff config
- [x] 1.5 Create `requirements.txt` — production dependencies
- [x] 1.6 Create `requirements-dev.txt` — dev/test dependencies
- [x] 1.7 Create `.env.example`
- [x] 1.8 Create `Makefile` — test, lint, run-*, clean
- [x] 1.9 Create `tests/fixtures/sample_sangraha/sample_rows.jsonl`
- [x] 1.10 Create `tests/fixtures/sample_metadata/metadata.json`
- [x] 1.11 Create `tests/fixtures/sample_configs/test_ingestion_config.yaml`
- [x] 1.12 Generate `tests/fixtures/sample_pdfs/sample_hindi_2page.pdf`
- [x] 1.13 Create `tests/conftest.py`
- [x] 1.14 Create all `__init__.py` stubs under `src/slm_hindi/`
- [x] 1.15 Verify: `pip install -e ".[dev]"` and `pytest --collect-only`

---

## Phase 2 — Config & Schema Layer ✅

**Goal:** Type-safe configuration loading from YAML files and a validated pydantic data model for all corpus records.

- [x] 2.1 Tests: `test_settings.py` — load from sample YAML, field assertions, validation error
- [x] 2.2 Tests: `test_corpus_record.py` — valid dict, defaults, ValidationError on bad type
- [x] 2.3 Tests: `test_run_logger.py` — CSV append, column presence, multi-row
- [x] 2.4 Tests: `test_file_registry.py` — register_file, sha256, FileNotFoundError
- [x] 2.5 Implement `configs/ingestion_config.yaml`
- [x] 2.6 Implement `configs/pdf_extraction_config.yaml`
- [x] 2.7 Implement `configs/model_cleaning_config.yaml`
- [x] 2.8 Implement `configs/quality_filter_config.yaml`
- [x] 2.9 Implement `configs/export_config.yaml`
- [x] 2.10 Implement `src/slm_hindi/config/settings.py` (`IngestionSettings`)
- [x] 2.11 Implement `src/slm_hindi/schema/corpus_record.py` (`CorpusRecord`)
- [x] 2.12 Implement `src/slm_hindi/observability/run_logger.py` (`IngestionRunLogger`)
- [x] 2.13 Implement `src/slm_hindi/observability/file_registry.py` (`FileRegistry`)

---

## Phase 3 — Sangraha Loader ✅

**Goal:** Load AI4Bharat Sangraha records and map to `CorpusRecord`.

- [x] 3.1 Tests: `test_sangraha_loader.py` — mock load_dataset, schema mapping, logger events
- [x] 3.2 Implement `sangraha_loader.py` — streaming optional, maps HF rows to CorpusRecord

---

## Phase 4 — PDF Registry & Extractor ✅

**Goal:** Discover and validate PDFs; extract text via PyMuPDF with pdfplumber fallback.

- [x] 4.1 Tests: `test_pdf_registry.py` — valid folder, missing metadata, missing PDF
- [x] 4.2 Tests: `test_pdf_extractor.py` — 2 pages extracted, PyMuPDF fallback to pdfplumber
- [x] 4.3 Implement `pdf_registry.py` (`PdfRegistry`, `PdfSource`, `PdfMetadata`)
- [x] 4.4 Implement `pdf_extractor.py` (`PdfExtractor`) — fitz primary, pdfplumber fallback

---

## Phase 5 — Ollama Cleaner & Cleaning Validator ✅

**Goal:** Chunk PDF text, clean via Ollama REST, validate output against 7 quality checks.

- [x] 5.1 Tests: `test_ollama_cleaner.py` — mock requests.post, chunking, retry on timeout
- [x] 5.2 Tests: `test_cleaning_validator.py` — all 7 reject cases + pass case
- [x] 5.3 Implement `ollama_cleaner.py` — chunking with overlap, exponential backoff
- [x] 5.4 Implement `cleaning_validator.py` — 7 checks, quarantine to Parquet

---

## Phase 6 — Text Normalizer & Quality Filter ✅

**Goal:** Deterministic text normalization and Hindi quality filtering.

- [x] 6.1 Tests: `test_text_normalizer.py` — NFC, whitespace, danda, repeated lines, URL
- [x] 6.2 Tests: `test_quality_filter.py` — Devanagari ratio, char bounds, table fragment
- [x] 6.3 Implement `text_normalizer.py` — NFC, whitespace, danda, URL handling
- [x] 6.4 Implement `quality_filter.py` — ratio + length filter, quality_score computation

---

## Phase 7 — Deduplicator & Corpus Splitter ✅

**Goal:** Remove exact and near-duplicate records; split at document level.

- [x] 7.1 Tests: `test_deduplicator.py` — exact dups removed, near-dups clustered, distinct kept
- [x] 7.2 Tests: `test_corpus_splitter.py` — 98/1/1 ratios, document-level grouping, seed
- [x] 7.3 Implement `deduplicator.py` — SHA-256 pass 1, MinHash LSH pass 2
- [x] 7.4 Implement `corpus_splitter.py` — document shuffle by seed, 98/1/1 assignment

---

## Phase 8 — Exporter, Manifest & Orchestration ✅

**Goal:** Write corpus in all output formats, generate manifest, wire CLI.

- [x] 8.1 Tests: `test_corpus_exporter.py` — Parquet roundtrip, JSONL.gz, TXT.gz, shards
- [x] 8.2 Tests: `test_manifest_generator.py` — SHA-256 match, version, profile totals
- [x] 8.3 Tests: `test_pdf_pipeline.py` — registry → extract → clean (mocked) → normalize → filter
- [x] 8.4 Tests: `test_sangraha_pipeline.py` — load (mocked) → normalize → filter → dedup
- [x] 8.5 Tests: `test_full_pipeline.py` — end-to-end mocked, Parquet exists, CSV populated
- [x] 8.6 Implement `corpus_exporter.py` — sharded Parquet (zstd), JSONL.gz, TXT.gz
- [x] 8.7 Implement `manifest_generator.py` — manifest.json + profile.json
- [x] 8.8 Implement `run_ingestion.py` — typer CLI, --config/--source/--dry-run flags

---

## Phase 9 — Wikipedia Crawler ✅

**Goal:** BFS crawler over Hindi Wikipedia producing `CorpusRecord` objects with `source_type="wiki"`.

**Acceptance Criteria:**
- BFS respects `max_depth` and `max_pages` per seed
- `link_include_pattern` / `link_exclude_pattern` regexes filter outbound links
- Namespace-prefixed titles (e.g. `Wikipedia:`, `Help:`) excluded automatically
- Paragraphs shorter than `min_paragraph_chars` skipped
- Rate limiting enforced between API calls
- Timeout retries with exponential backoff; raises `WikiCrawlError` after max retries
- `progress_callback` fires per page crawled
- `source_type == "wiki"` on all produced records

### Tasks

- [x] 9.1 Add `WikiSourceConfig`, `WikiSeedConfig`, `WikiCrawlConfig` to `settings.py`
- [x] 9.2 Create `configs/wiki_crawl_config.yaml`
- [x] 9.3 Create `tests/fixtures/sample_wiki/sample_extract_response.json`
- [x] 9.4 Create `tests/fixtures/sample_wiki/sample_links_response.json`
- [x] 9.5 Write `tests/unit/test_wiki_crawler.py` (23 tests):
  - `TestTitleFromUrl` — encoded URL decoding, underscore-to-space
  - `TestCleanExtract` — section removal, heading markers, blank line collapse, empty input
  - `TestWikiCrawlerFetchExtract` — returns text+URL, raises on missing page
  - `TestWikiCrawlerFetchLinks` — namespace-0 titles returned, pagination followed
  - `TestWikiCrawlerFilterLinks` — namespace prefixes, global/seed exclude patterns
  - `TestWikiCrawlerProcessPage` — produces CorpusRecords, skips short paragraphs, skips missing page
  - `TestWikiCrawlerRetry` — retries on Timeout, raises WikiCrawlError after max retries
  - `TestWikiCrawlerCrawlSeed` — BFS respects max_pages, progress_callback fires
- [x] 9.6 Implement `ingestion/wiki_crawler.py` (`WikiCrawler`, `WikiCrawlError`)
  - `crawl_seed(seed, run_logger, file_registry, progress_callback) -> list[CorpusRecord]`
  - `_fetch_extract(title) -> tuple[str, str]` — MediaWiki `prop=extracts&explaintext=1`
  - `_fetch_links(title) -> list[str]` — MediaWiki `prop=links&plnamespace=0` with pagination
  - `_clean_extract(text) -> str` — strip `== Section ==` headers, collapse blank lines
  - `_filter_links(titles, seed) -> list[str]` — namespace + regex filtering
  - `_process_page(title, text, url) -> list[CorpusRecord]` — split on `\n\n`, min_chars
  - `_title_from_url(url) -> str` — URL decode + underscore-to-space
- [x] 9.7 Add `source_type="wiki"` Literal to `CorpusRecord`
- [x] 9.8 Add wiki stage to `run_ingestion.py`
- [x] 9.9 Remove unused `BeautifulSoup` import (MediaWiki `explaintext=1` returns plain text)
- [x] 9.10 Verify: all 23 wiki tests pass

---

## Phase 10 — Rich Terminal UI, Progress Bars & Monorepo Reorganization ✅

**Goal:** Rich-formatted CLI output with per-stage progress bars; reorganize project into `data_ingestion/` subfolder to support future `tokenizer_training/` and `slm_training/` phases.

**Acceptance Criteria:**
- All pipeline stages show spinner + progress bar in terminal
- Logging output uses Rich formatting (coloured, timestamped)
- Windows CP1252 encoding issue resolved (box-drawing chars render correctly)
- Single `Console` instance shared across all modules
- All pipeline component methods accept `progress_callback: Callable[[int], None] | None = None`
- Project lives in `data_ingestion/` subfolder
- `CORPUS_HANDOFF.md` at monorepo root documents corpus contract for downstream phases
- `run_tests.sh/.bat` and `run_pipeline.sh/.bat` convenience scripts exist

### Tasks

- [x] 10.1 Create `src/slm_hindi/ui/__init__.py`
- [x] 10.2 Create `src/slm_hindi/ui/progress.py`:
  - `sys.stdout.reconfigure(encoding="utf-8")` at import (Windows fix)
  - `console = Console(legacy_windows=False)` — single shared instance
  - `setup_logging(log_level)` — configures `RichHandler` pointing at `console`
  - `make_progress() -> Progress` — spinner + bar + MofN + elapsed
  - `pipeline_progress(description, total)` — context manager variant
- [x] 10.3 Add `progress_callback` to `sangraha_loader.py` (`load()`)
- [x] 10.4 Add `progress_callback` to `text_normalizer.py` (`normalize_records()`)
- [x] 10.5 Add `progress_callback` to `quality_filter.py` (`filter()`)
- [x] 10.6 Add `progress_callback` to `deduplicator.py` (`deduplicate()`)
- [x] 10.7 Add `progress_callback` to `corpus_splitter.py` (`split()`)
- [x] 10.8 Add `progress_callback` to `corpus_exporter.py` (`export()`)
- [x] 10.9 Rewrite `run_ingestion.py`:
  - Import `console` from `slm_hindi.ui.progress` (no own Console)
  - Call `setup_logging()` at startup
  - Wrap each stage in `make_progress()` context manager
  - Pass `lambda n: progress.advance(task, n)` as `progress_callback`
  - Add `--source wiki` support
  - Rich `console.rule()` / `console.print()` for stage headers and summaries
- [x] 10.10 Create `run_tests.sh` + `run_tests.bat` (venv activation + pytest)
- [x] 10.11 Create `run_pipeline.sh` + `run_pipeline.bat` (venv activation + python -m)
- [x] 10.12 Update `Makefile` — add `run-wiki` target, `test-int` alias
- [x] 10.13 Reorganize into monorepo:
  - Move all files into `data_ingestion/` subfolder
  - Create `tokenizer_training/.gitkeep` and `slm_training/.gitkeep`
  - Update `.gitignore` path prefixes from `data/` to `data_ingestion/data/`
  - Move `CLAUDE.md`, `DevelopmentPlan.md`, `action-items.md` into `data_ingestion/`
  - Create `CORPUS_HANDOFF.md` at monorepo root
- [x] 10.14 Verify: 131 tests pass, ≥83% coverage, rich CLI renders correctly on Windows

---

## Completion Checklist ✅

- [x] All unit tests pass: `pytest tests/unit/ -v` (131 tests)
- [x] All integration tests pass (Ollama mocked): `pytest tests/integration/ -v -m "not requires_ollama"`
- [x] Coverage ≥ 80%: 83% achieved
- [x] Linting clean: `make lint`
- [x] `DevelopmentPlan.md` up to date (v2.0)
- [x] `CORPUS_HANDOFF.md` created at monorepo root
- [x] `pipeline_run_log.csv` and `data_file_registry.csv` generated during live pipeline run
- [x] Pipeline successfully ingesting live Sangraha data (17.4M records downloaded)
- [ ] `data/final/` output loadable via `datasets.load_dataset("parquet", data_files=...)` — pending pipeline completion
