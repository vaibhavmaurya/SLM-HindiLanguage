# Development Best Practices — Hindi SLM Project

Extracted from the data ingestion workstream. Apply these practices to every subsequent workstream: tokenizer training, SLM pretraining, evaluation.

---

## 1. Test-Driven Development (TDD)

This project uses strict TDD. No production code is written without a failing test first.

**The cycle — enforce it, never skip it:**
1. Write a failing test against sample fixture data
2. Implement the minimum code to make it pass
3. Refactor while keeping tests green
4. Run the full suite before declaring a phase complete

**Why it matters here:** We learned that skipping TDD on even one component (e.g., wiki normalization) caused a silent gap — wiki records were crawled but never normalized before quality filtering. The bug was only caught when restructuring the pipeline. A test would have caught it immediately.

**Rules:**
- Every module gets its own `tests/unit/test_<module>.py`
- Tests use only fixture data from `tests/fixtures/` — never download real data
- External calls (HuggingFace, Ollama, HTTP APIs) are always mocked with `unittest.mock.patch`
- Integration tests that require live services are marked `@pytest.mark.requires_ollama` (or equivalent) and skipped by default
- Coverage target: **≥ 80%**. Measure after every phase: `pytest --cov=src/<package> --cov-report=term-missing`
- Each phase ends with `pytest tests/ -v -m "not requires_<service>"` — all tests must pass before proceeding

---

## 2. Project & Directory Structure

**Monorepo layout — one subfolder per workstream:**
```
SLM_HINDI/
├── development_best_practices.md   ← this file
├── CORPUS_HANDOFF.md               ← contract between data ingestion and downstream
├── data_ingestion/
├── tokenizer_training/
└── slm_training/
```

Each workstream subfolder is self-contained:
```
<workstream>/
├── CLAUDE.md           ← Claude Code init file for this workstream
├── configs/            ← one YAML file per concern
├── src/<package>/      ← all production code, installed as editable package
├── tests/
│   ├── conftest.py
│   ├── fixtures/       ← small committed sample files only
│   ├── unit/
│   └── integration/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── run_tests.sh / run_tests.bat
└── run_pipeline.sh / run_pipeline.bat
```

**Key rules:**
- Source code lives under `src/<package>/` and is installed with `pip install -e ".[dev]"` — never add `src/` to `sys.path` manually
- All pipeline data goes to `data/` inside the workstream folder — gitignored except test fixtures
- `CLAUDE.md` at the workstream root documents the tech stack, directory layout, run commands, and code style for that workstream

---

## 3. Configuration Design

**One YAML file per concern, not one giant config:**
- `ingestion_config.yaml` — sources, runtime settings, data root
- `pdf_extraction_config.yaml` — PyMuPDF/pdfplumber tuning
- `model_cleaning_config.yaml` — Ollama model, chunking, retry
- `quality_filter_config.yaml` — ratio thresholds, length bounds
- `export_config.yaml` — shard sizes, split ratios, output formats

**Rules:**
- Load all config with `pydantic-settings` + `PyYAML` → type-safe, validated at startup
- **Zero hardcoded values** in source code — paths, thresholds, model names, URLs all come from config
- Environment variables override YAML values (pydantic-settings handles this automatically)
- Provide `.env.example` — never commit `.env`
- Pass the config object into components as a typed pydantic model, not as raw dicts

**Pattern:**
```python
class TextNormalizerConfig(BaseModel):
    remove_urls: bool = True
    min_char_count: int = 50

class TextNormalizer:
    def __init__(self, config: TextNormalizerConfig) -> None:
        self._cfg = config
```

---

## 4. Schema Design

**Use Pydantic v2 for all structured data crossing module boundaries.** Never pass bare `dict` between pipeline stages.

- One unified schema per workstream output (e.g., `CorpusRecord` for data ingestion, `TokenizerVocabEntry` for tokenizer)
- Use `Literal` types for categorical fields: `Literal["train", "validation", "test"]`
- Use `| None` for optional fields, never `Optional` (Python 3.10+ union syntax)
- Define sensible defaults so records can be constructed incrementally
- Version output filenames using a `corpus_version` string from config (e.g., `hindi_corpus_v001_train_00000.parquet`) — bumping the version string forces all downstream caches to refresh

**Pre-compute fields that bypassed stages would normally set.** We learned this when Sangraha bypassed the `Deduplicator` — `dedup_hash` was never populated. Fix: compute `dedup_hash` in `SangrahaLoader._map_row()` using `hashlib.sha256` so the field is always valid regardless of pipeline track.

---

## 5. Pipeline Architecture

### 5.1 One Module Per Stage

Each pipeline stage is a separate Python class in its own module. Stages are independently testable and independently runnable.

```
ingestion/
  sangraha_loader.py   → SangrahaLoader
  pdf_extractor.py     → PdfExtractor
  text_normalizer.py   → TextNormalizer
  quality_filter.py    → QualityFilter
  deduplicator.py      → Deduplicator
  corpus_splitter.py   → CorpusSplitter
  corpus_exporter.py   → CorpusExporter
```

### 5.2 Two-Track Design for Pre-Verified Data

**Not all data needs all cleaning stages.** If a source is pre-verified (e.g., AI4Bharat Sangraha), bypass the cleaning stages entirely rather than running them and passing everything through.

```
Sangraha  → load only → merge ─────────────────────────────────┐
PDF       → extract → clean → validate → normalize → filter ──► merge → split → export
Wiki      → crawl → normalize → filter ─────────────────────────┘
```

**Why:** Running a quality filter on pre-verified data wastes time and risks incorrectly rejecting legitimate records (e.g., Sangraha records where `devanagari_ratio` defaults to `0.0` because the metric was never computed).

### 5.3 Standard Component Signature

Every pipeline method that processes records accepts these optional kwargs:

```python
def process(
    self,
    records: list[CorpusRecord],
    run_logger: IngestionRunLogger | None = None,
    file_registry: FileRegistry | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> list[CorpusRecord]:
```

- Pass `None` in unit tests — no observability deps required
- Pass real instances in integration tests and production runs
- The orchestrator wires them together; components never import the orchestrator

### 5.4 Directory Creation at Startup

**Create all expected output directories at pipeline startup, unconditionally.** Never assume a directory exists before a stage tries to write to it.

```python
for _subdir in ["raw/pdf", "extracted/pdf", "normalized", "filtered", "final/parquet", "reports", ...]:
    (data_root / _subdir).mkdir(parents=True, exist_ok=True)
```

Also call `path.parent.mkdir(parents=True, exist_ok=True)` immediately before writing any individual file (quarantine files, manifest, etc.) — a stage might write to a path not in the standard list.

---

## 6. Resumability and Checkpointing

**Expensive operations must be checkpointed.** A pipeline that cannot resume from a failure wastes hours of compute.

**Pattern used for Sangraha (17.4M records, hours to download):**
1. After the first successful load, write sharded zstd Parquet to `data/raw/huggingface/sangraha_checkpoint_{version}_{idx:05d}.parquet`
2. On startup, check for checkpoint shards via glob — if found, load from disk and skip the download
3. Key checkpoint filenames by `corpus_version` — bumping the version automatically invalidates the cache
4. Provide a `--force-reload` flag to explicitly delete and re-download

**Generalise this to any stage that is slow or calls an external service:**
- Tokenizer training output: checkpoint vocab after each epoch
- Any HuggingFace download: checkpoint locally as Parquet/Arrow
- Any LLM API call batch: write results incrementally, skip already-processed IDs on retry

**Shard size:** Use the same `shard_size_mb` as the final export config (default 512MB). Consistent shard sizing makes the checkpoint files directly usable as intermediate inputs.

---

## 7. Observability

**Two persistent audit files, append-only across all runs:**

| File | Purpose |
|---|---|
| `data/reports/pipeline_run_log.csv` | One row per pipeline event — phase, component, status, record counts, duration, errors |
| `data/reports/data_file_registry.csv` | One row per file produced/consumed — path, size, SHA-256, row count, compression |

**Every run gets a UUID `run_id`** generated at startup. All log rows and file registry entries carry it. This lets you trace exactly what a given run produced.

**Implementation pattern:**
```python
run_id = str(uuid.uuid4())
run_logger = IngestionRunLogger(run_id, reports_dir / "pipeline_run_log.csv")
file_registry = FileRegistry(run_id, reports_dir / "data_file_registry.csv")
```

**At the start and end of every stage:**
```python
run_logger.log_event(phase="normalize", component="text_normalizer", status="started", records_in=len(records))
# ... do work ...
run_logger.log_event(phase="normalize", component="text_normalizer", status="completed", records_out=len(result))
```

**After every output file is written:**
```python
file_registry.register_file(path, role="output", stage="export", file_format="parquet",
                             row_count=len(shard), compression="zstd", notes="split=train,shard_index=0")
```

---

## 8. Progress Reporting

**Keep Rich (or any UI library) out of component code.** Use a callback pattern instead:

```python
# Component accepts a callback
def normalize_records(
    self,
    records: list[CorpusRecord],
    progress_callback: Callable[[int], None] | None = None,
) -> list[CorpusRecord]:
    result = []
    for record in records:
        result.append(self._normalize(record))
        if progress_callback:
            progress_callback(1)
    return result

# Orchestrator wires the callback to the progress bar
with make_progress() as progress:
    task = progress.add_task("[cyan]Normalize", total=len(records))
    result = normalizer.normalize_records(
        records,
        progress_callback=lambda n: progress.advance(task, n),
    )
```

**Single `Console` instance** — create it once in `ui/progress.py` and import it everywhere. Never instantiate a second `Console()`.

```python
# ui/progress.py
console = Console(legacy_windows=False)

# All other modules
from slm_hindi.ui.progress import console
```

---

## 9. External API Integration

**Rate limiting and retries are mandatory for any external HTTP call.**

- Implement exponential backoff with jitter on timeouts and 5xx errors
- Respect `Retry-After` headers on 429 responses
- Add configurable `rate_limit_delay_seconds` between calls
- Raise a typed domain exception (e.g., `WikiCrawlError`) after max retries, not a raw `requests` exception

**When an external API is unreliable, disable it via config — not by commenting out code:**
```yaml
sources:
  wiki:
    enabled: false  # disabled — hitting 429 Too Many Requests
```

The pipeline reads `if settings.sources.wiki.enabled:` and skips the stage cleanly. Re-enable by flipping the flag.

**Mock at the HTTP layer in tests:**
```python
with patch("requests.post") as mock_post:
    mock_post.return_value.json.return_value = {"response": "cleaned text"}
    result = cleaner.clean(records)
```

---

## 10. Windows-Specific Gotchas

**Batch file encoding:** cmd.exe on Windows reads `.bat` files as ANSI (Windows-1252), not UTF-8. Writing a `.bat` with the standard `Write` tool produces a UTF-8 file that cmd.exe misinterprets — every line is garbled.

Fix: write `.bat` files using PowerShell `Set-Content -Encoding Default`:
```powershell
$content | Set-Content -Path "setup_and_run.bat" -Encoding Default
```

**Rich terminal rendering:** Rich uses box-drawing Unicode characters that break on Windows CP1252 consoles.

Fix at import time in your UI module:
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
console = Console(legacy_windows=False)
```

**Venv location:** Always create the venv as `.venv/` inside the workstream folder. Batch scripts that use a relative path like `python -m venv .venv` can accidentally create the venv in the wrong directory if the working directory is not set correctly before the command runs. Always `cd` to the workstream folder explicitly at the top of setup scripts.

**Avoid nested `for /f` loops with `setlocal enabledelayedexpansion`** in batch files — they interact in subtle ways and produce `'variable' is not recognized` errors. Use `findstr` pattern matching for simple string checks instead.

---

## 11. Git and Repository Hygiene

**`.gitignore` must cover all pipeline data paths relative to the monorepo root:**
```
data_ingestion/data/raw/huggingface/
data_ingestion/data/extracted/
data_ingestion/data/final/
# etc.
```

If the project moves into a monorepo subfolder (as happened in Phase 10), update all `.gitignore` path prefixes — they are not relative to the module root, they are relative to the `.gitignore` file location.

**Also gitignore bare venv artefacts** that can appear if a setup script runs in the wrong directory:
```
pyvenv.cfg
Lib/
Scripts/
Include/
```

**Commit discipline:**
- One commit per logical change (phase, feature, fix)
- Commit messages: `feat:`, `fix:`, `refactor:`, `docs:` prefix
- Always run the full test suite before committing
- Every phase ending is a commit + push

**Phase handoff documents:** When a workstream completes, write a `*_HANDOFF.md` at the monorepo root describing the output contract: file locations, schema, loading code, quality thresholds. The next workstream reads this instead of reading source code.

---

## 12. Code Style Rules

These apply across all workstreams in this project:

| Rule | Detail |
|---|---|
| Type hints | Required on every function signature — no bare `def f(x)` |
| Pydantic v2 | For all structured data crossing module boundaries |
| No comments | Only add a comment when the WHY is non-obvious (hidden constraint, bug workaround) |
| No docstrings | A single short line is the maximum; never multi-paragraph |
| No hardcoded values | Every path, threshold, model name comes from config |
| No premature abstraction | Three similar lines is better than a wrong abstraction |
| No unused error handling | Don't catch exceptions that can't happen; trust internal code |
| `from __future__ import annotations` | Add to every module for forward reference support |
| Ruff for linting | `ruff check` and `ruff format` — configured in `pyproject.toml` |

---

## 13. Phase-Based Development Template

Use this structure for every new workstream:

```
Phase 1 — Scaffold & Tooling
  - Directory tree, pyproject.toml, requirements, Makefile, .gitignore, CLAUDE.md
  - Empty test suite that collects: pytest --collect-only passes with 0 errors

Phase 2 — Config & Schema Layer
  - YAML config files + pydantic-settings loader
  - Pydantic schema for the workstream's primary data record
  - Tests: load config from sample YAML, validate schema on sample dict

Phase N — Each Pipeline Stage
  - Write failing tests first (fixtures only, external calls mocked)
  - Implement minimum code to pass
  - End with: pytest tests/unit/test_<module>.py -v --cov=src/<package>

Final Phase — Orchestration & End-to-End
  - CLI entrypoint wiring all stages
  - Integration test (all external calls mocked, sample fixtures only)
  - Verify: full test suite passes, coverage ≥ 80%, linting clean
  - Write HANDOFF.md for the next workstream
```

---

## 14. Dependency Management

```
requirements.txt      — production runtime deps (pinned major versions)
requirements-dev.txt  — testing + linting deps (pytest, ruff, pytest-cov, pytest-mock)
pyproject.toml        — build system (hatchling), pytest settings, ruff config, extras
```

Install for development:
```bash
pip install -e ".[dev]"
```

This installs the package in editable mode so `import <package>` always reflects the current source without reinstalling.

---

## 15. Summary Checklist Before Moving to a New Phase

- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] All integration tests pass (live services mocked): `pytest tests/integration/ -v -m "not requires_<service>"`
- [ ] Coverage ≥ 80%: `pytest --cov=src/<package> --cov-report=term-missing`
- [ ] Linting clean: `ruff check src/ tests/`
- [ ] All output directories created at pipeline startup
- [ ] Resumability: expensive operations have checkpoints or idempotent re-runs
- [ ] Observability: run logger and file registry wired into all stages
- [ ] HANDOFF document written or updated
- [ ] Code committed and pushed
