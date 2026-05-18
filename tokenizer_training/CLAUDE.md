# CLAUDE.md — Tokenizer Training Workstream

This file initializes Claude Code for the `tokenizer_training/` workstream.
Read this file at the start of every session — do not assume context from other workstreams.

---

## 1. What This Workstream Does

Trains a Hindi BPE/Unigram subword tokenizer using the HuggingFace `tokenizers` library.
Input: the clean Hindi corpus produced by the `data_ingestion/` pipeline.
Output: a frozen `tokenizer.json` artifact, HF-compatible, ready for SLM pretraining.

The tokenizer bridges raw Hindi text and model tensor indices. Once SLM pretraining begins,
the tokenizer is **frozen** — no changes to vocab, IDs, normalization, or special tokens.

---

## 2. Input Corpus (from Data Ingestion)

Read `../CORPUS_HANDOFF.md` for the authoritative reference. Key facts:

- **Primary location:** `../data_ingestion/data/final/parquet/train/`
- **File naming:** `hindi_corpus_v001_train_{shard:05d}.parquet`
- **Text field:** `final_text` (str) — this is the field to use for training
- **Format:** Parquet, zstd-compressed, ~512 MB per shard
- **Corpus version:** `hindi_corpus_v001` (set in `../data_ingestion/configs/ingestion_config.yaml`)
- **Splits:** train (98%), validation (1%), test (1%)
- **Sangraha note:** `devanagari_ratio` and `quality_score` default to `0.0` for Sangraha records
  (they bypassed quality filter). Filter on `word_count` / `char_count` for Sangraha.

Loading pattern:
```python
import pandas as pd
from pathlib import Path

parquet_dir = Path("../data_ingestion/data/final/parquet/train")
train_df = pd.concat(
    [pd.read_parquet(f) for f in sorted(parquet_dir.glob("*.parquet"))],
    ignore_index=True,
)
texts = train_df["final_text"].tolist()
```

---

## 3. Tech Stack

| Concern | Choice |
|---|---|
| Tokenizer training | HuggingFace `tokenizers` library (Rust-backed) |
| Algorithm | Unigram (subword, handles Hindi morphology better than BPE) |
| Normalization | NFKC — no lowercasing (Devanagari has no case; preserve Latin acronyms) |
| Pre-tokenizer | Metaspace (SentencePiece-compatible `▁` boundary marker) |
| Vocab sizes | 24k, 32k (default), 48k — compare before committing |
| Config | pydantic-settings + PyYAML |
| Schema | pydantic v2 (`SampleManifest`, `ValidationReport`, `ComparisonResult`) |
| Data I/O | pyarrow + pandas |
| CLI | typer + rich |
| Testing | pytest + pytest-cov + pytest-mock |
| Linting | ruff (line-length 120, py311) |
| HF publishing | huggingface_hub |

---

## 4. Package Layout

```
tokenizer_training/
├── CLAUDE.md                   ← this file
├── DevelopmentPlan.md
├── action_items.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── configs/
│   ├── tokenizer_training_config.yaml
│   ├── tokenizer_validation_config.yaml
│   └── publish_config.yaml
├── src/
│   └── hindi_tokenizer/
│       ├── config/settings.py          → TokenizerSettings (pydantic-settings)
│       ├── schema/records.py           → SampleManifest, ValidationReport, ComparisonResult
│       ├── corpus/
│       │   ├── parquet_reader.py       → ParquetReader
│       │   └── corpus_sampler.py       → CorpusSampler
│       ├── training/
│       │   ├── tokenizer_trainer.py    → TokenizerTrainer
│       │   └── experiment_runner.py    → ExperimentRunner
│       ├── validation/
│       │   ├── tokenizer_validator.py  → TokenizerValidator
│       │   └── tokenizer_comparator.py → TokenizerComparator
│       ├── packaging/
│       │   ├── artifact_packager.py    → ArtifactPackager
│       │   └── checksum_generator.py  → ChecksumGenerator
│       ├── publishing/
│       │   └── tokenizer_publisher.py → TokenizerPublisher
│       ├── sdk/
│       │   ├── loader.py              → load_tokenizer()
│       │   ├── encode.py              → encode()
│       │   └── decode.py              → decode()
│       ├── observability/
│       │   ├── run_logger.py          → TokenizerRunLogger (append-only CSV)
│       │   └── file_registry.py       → FileRegistry (append-only CSV)
│       ├── orchestration/
│       │   └── run_tokenizer.py       → Typer CLI entrypoint
│       └── ui/
│           └── progress.py            → singleton Rich Console + make_progress()
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── sample_parquet/            ← small Parquet files (50 rows each), NOT real corpus
│   │   ├── sample_text/               ← small_corpus.txt, validation_sentences.txt
│   │   └── sample_configs/            ← test YAML configs
│   ├── unit/                          ← one test file per module
│   └── integration/
└── data/                              ← gitignored; tokenizer artifacts live here
    ├── samples/                       ← sampled training text files
    ├── experiments/                   ← per-variant tokenizer training outputs
    ├── artifacts/                     ← packaged, checksum-verified artifacts
    └── reports/                       ← CSV observability logs, validation reports
```

---

## 5. How to Install

```bash
cd tokenizer_training/
pip install -e ".[dev]"
```

Requires Python 3.11+. All imports use the `hindi_tokenizer` package name.

---

## 6. How to Run Tests

```bash
# All tests (from tokenizer_training/)
pytest tests/ -v --cov=hindi_tokenizer --cov-report=term-missing

# Single module
pytest tests/unit/test_tokenizer_trainer.py -v

# Skip HF Hub tests (default for local dev)
pytest tests/ -v -m "not requires_hf_hub"

# Windows batch script
run_tests.bat
```

Target: ≥80% coverage before each phase is marked complete.

---

## 7. How to Run the Pipeline

```bash
cd tokenizer_training/

# Full pipeline (sample → train → validate → package → publish)
python -m hindi_tokenizer.orchestration.run_tokenizer --config configs/tokenizer_training_config.yaml

# Individual steps
python -m hindi_tokenizer.orchestration.run_tokenizer --step sample
python -m hindi_tokenizer.orchestration.run_tokenizer --step train
python -m hindi_tokenizer.orchestration.run_tokenizer --step validate
python -m hindi_tokenizer.orchestration.run_tokenizer --step compare
python -m hindi_tokenizer.orchestration.run_tokenizer --step package
python -m hindi_tokenizer.orchestration.run_tokenizer --step publish

# Dry run (config validation only, no writes)
python -m hindi_tokenizer.orchestration.run_tokenizer --dry-run
```

---

## 8. Special Tokens (fixed — do not change)

| Token | ID | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown token |
| `<s>` | 2 | BOS / sequence start |
| `</s>` | 3 | EOS / sequence end |
| `<\|system\|>` | 4 | Chat system turn |
| `<\|user\|>` | 5 | Chat user turn |
| `<\|assistant\|>` | 6 | Chat assistant turn |
| `<\|end\|>` | 7 | Chat turn end |

These IDs are **frozen**. The SLM embedding matrix is sized to `len(tokenizer)`.

---

## 9. Validation Thresholds (from DevelopmentPlan.md)

| Metric | Threshold |
|---|---|
| `unk_rate` | < 0.001 |
| `chars_per_token` | > 3.0 |
| `tokens_per_word` | < 2.5 |
| `roundtrip_success_rate` | > 0.99 |
| `special_token_split_failures` | == 0 |
| `devanagari_char_coverage` | > 0.995 |

---

## 10. Code Style Rules

- No comments unless the WHY is non-obvious (hidden constraint, workaround, subtle invariant).
- Type hints required on all function signatures.
- pydantic v2 for all data schemas — no raw dicts crossing module boundaries.
- One class per file. File name matches class name in snake_case.
- Progress callbacks (`progress_callback: Callable[[int], None] | None = None`) keep Rich out of component code.
- Standard component signature: `def run(self, ..., run_logger=None, file_registry=None, progress_callback=None)`.
- Config drives everything — no hardcoded paths, thresholds, or vocab sizes.

---

## 11. TDD Mandate

- Write test cases (described in `action_items.md`) **before** writing production code.
- All tests use fixture data — never download real corpus data during tests.
- HuggingFace Hub calls are always mocked (`pytest-mock`); mark with `@pytest.mark.requires_hf_hub`.
- Each phase completes when: all phase tests pass + ruff is clean + coverage target met.

---

## 12. Observability

Two append-only CSVs written throughout the pipeline:
- `data/reports/pipeline_run_log.csv` — every pipeline event (phase, component, status, records_in/out, duration)
- `data/reports/data_file_registry.csv` — every file produced/consumed (path, sha256, size, row_count)

Both share the same `run_id` (UUID per CLI invocation). Components accept `run_logger` and `file_registry`
as optional kwargs so unit tests can pass `None`.

---

## 13. Tokenizer Freezing Rules

Once SLM pretraining begins, these are **prohibited**:
- Changing `tokenizer.json`, vocab, or merge rules
- Adding, removing, or renaming special tokens
- Changing token IDs
- Changing normalization (NFKC) or pre-tokenizer (Metaspace)
- Re-training or fine-tuning the tokenizer

The artifact in `data/artifacts/` with its SHA-256 checksum is the source of truth.

---

## 14. Key Reference Files

| File | Purpose |
|---|---|
| `../CORPUS_HANDOFF.md` | Corpus schema, file locations, loading examples |
| `DevelopmentPlan.md` | Architecture, design decisions, all module specs |
| `action_items.md` | Phase-by-phase tasks with all TDD test case descriptions |
| `../development_best_practices.md` | General Python ML workstream conventions |
