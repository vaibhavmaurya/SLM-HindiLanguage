# Development Plan: Hindi SLM Tokenizer Training Workstream

Design reference: https://github.com/vaibhavmaurya/SLM-HindiLanguage/wiki/SLM%E2%80%90HindiLanguage-Tokenizer-Training-Design

---

## 1. Executive Summary

This workstream trains, validates, packages, and publishes a Hindi subword tokenizer for the Hindi SLM project. The tokenizer is trained once, validated rigorously, frozen, and never modified after SLM pretraining begins. It is the bridge between raw Hindi text and model training tensors — `vocab_size`, token IDs, and normalization rules become permanent constraints on the model architecture.

The tokenizer uses the Unigram algorithm with NFKC normalization and Metaspace pre-tokenization, trained on a representative sample from the Sangraha Hindi Parquet corpus produced by the data ingestion workstream.

**Input:** `data_ingestion/data/final/parquet/train/*.parquet`
**Output:** `tokenizer_training/data/final/hindi_slm_tokenizer_v001/` — a frozen, Hugging Face-compatible tokenizer artifact

---

## 2. Position in the SLM Pipeline

```
Data Ingestion (Sangraha Hindi Parquet corpus)
        ↓
Tokenizer Training Workstream  ← this document
        ↓
Frozen Tokenizer Artifact (tokenizer.json + HF config)
        ↓
SLM Pretraining (vocab_size locked to len(tokenizer))
        ↓
Model Validation & Publishing
```

---

## 3. Scope

**In scope:**
- Corpus sampling from Sangraha Parquet train files
- Tokenizer training: 24k, 32k, 48k vocabulary size variants
- Tokenizer validation: metrics, roundtrip, special token integrity
- Tokenizer comparison and final selection
- Artifact packaging: tokenizer.json, HF configs, README, checksums
- Publishing to Hugging Face Hub (private repository)
- Local Python SDK for loading, encoding, and decoding

**Out of scope:**
- PDF or JSONL corpus sources
- Synthetic or instruction-tuning data
- Chat template finalization
- SLM pretraining
- Inference API

---

## 4. Architecture Overview

```
Sangraha Parquet Train Folder
        ↓
ParquetReader        — streams text column from .parquet files
        ↓
CorpusSampler        — filters (Devanagari ratio, char length), shuffles,
                       writes plain-text samples to disk
        ↓
TokenizerTrainer     — trains Unigram+NFKC+Metaspace via HF tokenizers
        ↓
ExperimentRunner     — repeats training for 24k, 32k, 48k vocab variants
        ↓
TokenizerValidator   — loads each artifact, computes validation metrics
        ↓
TokenizerComparator  — merges reports, produces comparison markdown
        ↓
ArtifactPackager     — assembles final artifact directory
ChecksumGenerator    — SHA-256 per file, writes checksums.json
        ↓
TokenizerPublisher   — pushes artifact to Hugging Face Hub
```

All stages are individually testable. The orchestrator (`run_tokenizer.py`) wires them together.

---

## 5. Design Decisions

### 5.1 Algorithm: Unigram

Unigram subword tokenization is chosen over word-level (too many rare tokens, poor generalization on inflected forms) and character-level (very long sequences, slow training). Byte-level BPE is deferred — Unigram with Metaspace is more interpretable for Hindi-first development.

Hindi's rich morphology (multiple suffixes, compound forms) benefits from subword units:
```
प्रशिक्षण → ▁प्रशिक्ष + ण       (Unigram can share ▁प्रशिक्ष across inflections)
प्रशिक्षित → ▁प्रशिक्षित
प्रशिक्षकों → ▁प्रशिक्षकों
```

### 5.2 Vocabulary Size: 32,000 (default)

Three variants are trained and compared:

| Variant | Vocab Size | Embedding Params (hidden=768) | Expected Behavior |
|---|---:|---:|---|
| `hindi_unigram_24k_v001` | 24,000 | 18,432,000 | Smaller, more splitting |
| `hindi_unigram_32k_v001` | 32,000 | 24,576,000 | Balanced default |
| `hindi_unigram_48k_v001` | 48,000 | 36,864,000 | Fewer splits, larger embedding |

32k is the recommended default unless 24k or 48k clearly outperforms it on validation metrics.

### 5.3 Normalization: Unicode NFKC, No Lowercasing

NFKC normalizes Unicode representations without altering semantic content. Lowercasing is disabled: Devanagari has no case distinction, and English acronyms embedded in Hindi text (UPI, GST, ISRO, NASA) must be preserved intact.

### 5.4 Pre-tokenizer and Decoder: Metaspace

Metaspace replaces spaces with `▁`, giving SentencePiece-like behaviour. Word boundaries are explicit and inspectable. The decoder converts `▁` back to spaces, making roundtrip lossless for normal Hindi text.

### 5.5 Special Tokens

```python
SPECIAL_TOKENS = [
    "<pad>",        # ID 0 — padding
    "<unk>",        # ID 1 — unknown token fallback
    "<s>",          # ID 2 — beginning of sequence
    "</s>",         # ID 3 — end of sequence
    "<|system|>",   # reserved for future chat
    "<|user|>",     # reserved for future chat
    "<|assistant|>",# reserved for future chat
    "<|end|>",      # reserved for future chat
]
```

IDs 0–3 are fixed. Extra special tokens consume vocabulary slots; overloading the base tokenizer with many unused tokens degrades those representations during pretraining.

### 5.6 Training Data Strategy

Tokenizer training requires a representative sample, not the full corpus. The objective is vocabulary discovery, not language learning.

| Stage | Sample Size | Purpose |
|---|---:|---|
| Smoke test | 500 MB | Validate pipeline end-to-end |
| Experiment | 5 GB | Train and compare 24k / 32k / 48k |
| Final (optional) | 10 GB | Retrain selected variant for production |

Sampling rules:
- Text column: `text` (or `final_text` depending on ingestion schema — see `CORPUS_HANDOFF.md`)
- Minimum characters: 30
- Maximum characters: 5,000 per record
- Minimum Devanagari ratio: 0.60
- Shuffle with `random_seed = 42` for reproducibility
- Output: one record per line, plain UTF-8 text

### 5.7 Validation Thresholds

| Metric | Target |
|---|---|
| Unknown token rate | < 0.1% (0.001) |
| Characters per token | > 3.0 |
| Tokens per word | < 2.5 |
| Roundtrip success rate | > 99% (0.99) |
| Special token split failures | 0 |
| Devanagari character coverage | > 99.5% (0.995) |

---

## 6. Directory Structure

```
tokenizer_training/
├── CLAUDE.md
├── DevelopmentPlan.md          ← this file
├── action_items.md
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── run_tests.sh
├── run_tests.bat
├── run_pipeline.sh
├── run_pipeline.bat
│
├── configs/
│   ├── tokenizer_training_config.yaml   ← corpus sampling + trainer settings
│   ├── tokenizer_validation_config.yaml ← validation thresholds
│   └── publish_config.yaml              ← HF repo, privacy, target paths
│
├── src/hindi_tokenizer/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          ← pydantic-settings loader
│   ├── schema/
│   │   ├── __init__.py
│   │   └── records.py           ← SampleManifest, ValidationReport, ComparisonResult
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── parquet_reader.py    ← ParquetReader: stream text from .parquet files
│   │   └── corpus_sampler.py    ← CorpusSampler: filter, shuffle, write .txt
│   ├── training/
│   │   ├── __init__.py
│   │   ├── tokenizer_trainer.py ← TokenizerTrainer: HF tokenizers Unigram training
│   │   └── experiment_runner.py ← ExperimentRunner: runs 24k/32k/48k variants
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── tokenizer_validator.py  ← TokenizerValidator: metrics, roundtrip, special tokens
│   │   └── tokenizer_comparator.py ← TokenizerComparator: merge reports, produce markdown
│   ├── packaging/
│   │   ├── __init__.py
│   │   ├── artifact_packager.py    ← ArtifactPackager: assemble final artifact dir
│   │   └── checksum_generator.py  ← ChecksumGenerator: SHA-256 per file
│   ├── publishing/
│   │   ├── __init__.py
│   │   └── tokenizer_publisher.py ← TokenizerPublisher: push to HF Hub
│   ├── sdk/
│   │   ├── __init__.py
│   │   ├── loader.py    ← load_hindi_slm_tokenizer(path_or_repo_id)
│   │   ├── encode.py    ← encode_text(tokenizer, text, add_special_tokens)
│   │   └── decode.py    ← decode_ids(tokenizer, input_ids)
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── run_logger.py    ← TokenizerRunLogger (CSV, append-only)
│   │   └── file_registry.py ← FileRegistry (CSV, append-only, SHA-256)
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── run_tokenizer.py ← CLI entrypoint (typer)
│   └── ui/
│       ├── __init__.py
│       └── progress.py      ← singleton Console, make_progress(), setup_logging()
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── sample_parquet/          ← 2–3 small .parquet files with Hindi text
│   │   ├── sample_text/             ← small .txt file for trainer unit tests
│   │   ├── sample_configs/          ← test YAML configs
│   │   └── validation_sentences.txt ← ~20 Hindi validation sentences
│   ├── unit/
│   │   ├── test_settings.py
│   │   ├── test_records.py
│   │   ├── test_parquet_reader.py
│   │   ├── test_corpus_sampler.py
│   │   ├── test_tokenizer_trainer.py
│   │   ├── test_experiment_runner.py
│   │   ├── test_tokenizer_validator.py
│   │   ├── test_tokenizer_comparator.py
│   │   ├── test_artifact_packager.py
│   │   ├── test_checksum_generator.py
│   │   ├── test_tokenizer_publisher.py
│   │   ├── test_sdk_loader.py
│   │   ├── test_sdk_encode.py
│   │   ├── test_sdk_decode.py
│   │   ├── test_run_logger.py
│   │   └── test_file_registry.py
│   └── integration/
│       ├── test_corpus_pipeline.py   ← ParquetReader → CorpusSampler end-to-end
│       ├── test_training_pipeline.py ← CorpusSampler → TokenizerTrainer end-to-end
│       └── test_full_pipeline.py     ← full pipeline on fixtures (HF Hub mocked)
│
└── data/                             ← gitignored
    ├── samples/
    │   ├── smoke_test/
    │   ├── experiment/
    │   └── final/
    ├── artifacts/
    │   ├── hindi_unigram_24k_v001/
    │   ├── hindi_unigram_32k_v001/
    │   └── hindi_unigram_48k_v001/
    ├── final/
    │   └── hindi_slm_tokenizer_v001/
    │       ├── tokenizer.json
    │       ├── tokenizer_config.json
    │       ├── special_tokens_map.json
    │       ├── added_tokens.json
    │       ├── tokenizer_metadata.json
    │       ├── tokenizer_training_config.yaml
    │       ├── tokenizer_validation_report.json
    │       ├── tokenizer_comparison_report.md
    │       ├── README.md
    │       ├── VERSION
    │       └── checksums.json
    ├── validation/
    │   └── hindi_validation_sentences.txt
    └── reports/
        ├── pipeline_run_log.csv
        ├── data_file_registry.csv
        ├── hindi_unigram_24k_v001_report.json
        ├── hindi_unigram_32k_v001_report.json
        ├── hindi_unigram_48k_v001_report.json
        └── tokenizer_comparison_report.md
```

---

## 7. Module Catalogue

| Module | Class | Responsibility |
|---|---|---|
| `corpus/parquet_reader.py` | `ParquetReader` | Stream text column from Parquet files; filter nulls; return string iterator |
| `corpus/corpus_sampler.py` | `CorpusSampler` | NFKC normalize, filter by char count and Devanagari ratio, shuffle, write plain-text sample |
| `training/tokenizer_trainer.py` | `TokenizerTrainer` | Train Unigram+NFKC+Metaspace tokenizer via HF `tokenizers`; save HF-compatible artifact |
| `training/experiment_runner.py` | `ExperimentRunner` | Iterate over vocabulary size variants and call `TokenizerTrainer` for each |
| `validation/tokenizer_validator.py` | `TokenizerValidator` | Load artifact via `AutoTokenizer`, compute all validation metrics, write JSON report |
| `validation/tokenizer_comparator.py` | `TokenizerComparator` | Load multiple reports, rank by metrics, write comparison markdown |
| `packaging/artifact_packager.py` | `ArtifactPackager` | Copy winning artifact to `final/`, add VERSION, README, training config |
| `packaging/checksum_generator.py` | `ChecksumGenerator` | SHA-256 every file in artifact dir, write `checksums.json` |
| `publishing/tokenizer_publisher.py` | `TokenizerPublisher` | Create HF repo (private), upload artifact folder via `huggingface_hub` |
| `sdk/loader.py` | — | `load_hindi_slm_tokenizer(path_or_repo_id)` — thin `AutoTokenizer` wrapper |
| `sdk/encode.py` | — | `encode_text(tokenizer, text, add_special_tokens)` |
| `sdk/decode.py` | — | `decode_ids(tokenizer, input_ids)` |
| `observability/run_logger.py` | `TokenizerRunLogger` | Append-only CSV of pipeline events with UUID run_id |
| `observability/file_registry.py` | `FileRegistry` | Append-only CSV of files produced/consumed with SHA-256 |
| `orchestration/run_tokenizer.py` | — | Typer CLI; wires all stages; accepts `--step`, `--config`, `--dry-run` flags |
| `config/settings.py` | `TokenizerSettings` | pydantic-settings: load YAML + env var overrides |
| `schema/records.py` | `SampleManifest`, `ValidationReport`, `ComparisonResult` | Pydantic v2 structured output schemas |
| `ui/progress.py` | — | Singleton `Console`, `make_progress()`, `setup_logging()` |

---

## 8. Configuration Schema

### `tokenizer_training_config.yaml`

```yaml
project:
  name: hindi-slm-tokenizer
  tokenizer_version: hindi_slm_tokenizer_v001
  data_root: data
  log_level: INFO

input:
  parquet_train_folder: ../data_ingestion/data/final/parquet/train
  text_column: final_text
  file_pattern: "*.parquet"

sampling:
  random_seed: 42
  smoke_test:
    target_size_gb: 0.5
    output_file: data/samples/smoke_test/sangraha_tokenizer_sample_500mb.txt
  experiment:
    target_size_gb: 5.0
    output_file: data/samples/experiment/sangraha_tokenizer_sample_5gb.txt
  final:
    target_size_gb: 10.0
    output_file: data/samples/final/sangraha_tokenizer_sample_10gb.txt

text_filters:
  min_char_count: 30
  max_char_count: 5000
  min_devanagari_ratio: 0.60
  remove_empty: true
  normalize_unicode: true

tokenizer:
  algorithm: unigram
  vocab_sizes: [24000, 32000, 48000]
  default_vocab_size: 32000
  normalizer: nfkc
  pre_tokenizer: metaspace
  decoder: metaspace
  max_piece_length: 24
  n_sub_iterations: 2
  model_max_length: 2048

special_tokens:
  pad_token: "<pad>"
  unk_token: "<unk>"
  bos_token: "<s>"
  eos_token: "</s>"
  additional_special_tokens:
    - "<|system|>"
    - "<|user|>"
    - "<|assistant|>"
    - "<|end|>"

artifacts:
  artifact_dir: data/artifacts
  final_dir: data/final/hindi_slm_tokenizer_v001
```

### `tokenizer_validation_config.yaml`

```yaml
validation:
  validation_file: data/validation/hindi_validation_sentences.txt
  reports_dir: data/reports
  thresholds:
    max_unk_rate: 0.001
    min_chars_per_token: 3.0
    max_tokens_per_word: 2.5
    min_roundtrip_success_rate: 0.99
    min_devanagari_coverage: 0.995
    max_special_token_split_failures: 0
```

### `publish_config.yaml`

```yaml
publish:
  hf_repo_id: vaibhavmaurya/hindi-slm-tokenizer-v001
  private: true
  create_repo_if_missing: true
  source_dir: data/final/hindi_slm_tokenizer_v001
```

---

## 9. Pydantic Schema Design

### `records.py`

```python
class SampleManifest(BaseModel):
    output_file: str
    target_size_gb: float
    actual_size_gb: float
    written_records: int
    random_seed: int
    created_at: str = Field(default_factory=_utcnow)

class ValidationReport(BaseModel):
    tokenizer_dir: str
    tokenizer_version: str
    vocab_size: int
    total_examples: int
    total_tokens: int
    total_characters: int
    total_words: int
    unk_rate: float
    chars_per_token: float
    tokens_per_word: float
    roundtrip_success_rate: float
    special_token_failures: list[dict]
    pad_token_id: int
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    passes_thresholds: bool
    created_at: str = Field(default_factory=_utcnow)

class ComparisonResult(BaseModel):
    variants: list[str]
    reports: list[ValidationReport]
    recommended_variant: str
    recommendation_reason: str
    created_at: str = Field(default_factory=_utcnow)
```

---

## 10. Observability

Same pattern as data ingestion:

| File | Purpose |
|---|---|
| `data/reports/pipeline_run_log.csv` | One row per pipeline event |
| `data/reports/data_file_registry.csv` | One row per file produced/consumed |

Every run has a UUID `run_id`. All stages accept optional `run_logger` and `file_registry` kwargs — pass `None` in unit tests, real instances in production.

---

## 11. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.11+ | match-statement, tomllib |
| Tokenizer training | HF `tokenizers` library | Industry standard; Unigram trainer built-in |
| HF compatibility | `transformers` `PreTrainedTokenizerFast` | Load via `AutoTokenizer.from_pretrained` |
| Publishing | `huggingface_hub` | API for create_repo + upload_folder |
| Parquet reading | `pyarrow` | Fast, zero-copy |
| Config | `pydantic-settings` + `PyYAML` | Type-safe, env-var overrides |
| Schema validation | `pydantic` v2 | All structured data crossing module boundaries |
| CLI | `typer` | Clean help text; `--step`, `--config`, `--dry-run` flags |
| Rich UI | `rich` | Progress bars, console; singleton pattern |
| Testing | `pytest`, `pytest-cov`, `pytest-mock` | Mock HF Hub calls; no real network in unit tests |
| Linting | `ruff` | Consistent with data ingestion workstream |

---

## 12. CLI Design

```bash
# Full pipeline on experiment sample
python -m hindi_tokenizer.orchestration.run_tokenizer --step all

# Individual steps
python -m hindi_tokenizer.orchestration.run_tokenizer --step sample
python -m hindi_tokenizer.orchestration.run_tokenizer --step train
python -m hindi_tokenizer.orchestration.run_tokenizer --step validate
python -m hindi_tokenizer.orchestration.run_tokenizer --step compare
python -m hindi_tokenizer.orchestration.run_tokenizer --step package
python -m hindi_tokenizer.orchestration.run_tokenizer --step publish

# Smoke test (500 MB sample, 32k only)
python -m hindi_tokenizer.orchestration.run_tokenizer --step all --smoke-test

# Dry run (config validation only)
python -m hindi_tokenizer.orchestration.run_tokenizer --dry-run

# Custom config
python -m hindi_tokenizer.orchestration.run_tokenizer --config configs/tokenizer_training_config.yaml
```

---

## 13. Tokenizer Freezing Rule

Once SLM pretraining starts, the following are permanently prohibited:

- Modifying `tokenizer.json`
- Adding new tokens
- Changing special token IDs
- Retraining the vocabulary
- Reordering vocabulary
- Changing normalization, pre-tokenizer, or decoder settings

Allowed after freezing: updating README, adding chat template metadata (without changing token IDs), creating a new version (`v002`) only alongside a new model.

---

## 14. Integration with Downstream (SLM Pretraining)

The model configuration must match the frozen tokenizer exactly:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("data/final/hindi_slm_tokenizer_v001")

# vocab_size MUST equal len(tokenizer) — mismatches cause training failures
config = GPT2Config(
    vocab_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
```

The TOKENIZER_HANDOFF.md (written at the end of this workstream) documents all token IDs, vocab size, artifact path, and HF repo ID for the SLM pretraining workstream.

---

## 15. Phase Overview

| Phase | Goal | Key Deliverables |
|---|---|---|
| 1 | Scaffold & Tooling | Directory tree, pyproject.toml, Makefile, CLAUDE.md, empty test suite |
| 2 | Config & Schema | YAML configs, pydantic-settings loader, `SampleManifest`, `ValidationReport`, `ComparisonResult` schemas |
| 3 | Corpus Reader & Sampler | `ParquetReader`, `CorpusSampler`, smoke-test sample written |
| 4 | Tokenizer Trainer | `TokenizerTrainer`, `ExperimentRunner`, 3 artifacts trained |
| 5 | Tokenizer Validator | `TokenizerValidator`, validation reports for all 3 variants |
| 6 | Tokenizer Comparator | `TokenizerComparator`, comparison markdown, final selection |
| 7 | Artifact Packager | `ArtifactPackager`, `ChecksumGenerator`, final frozen artifact |
| 8 | Publisher & SDK | `TokenizerPublisher`, SDK modules, `AutoTokenizer` smoke test |
| 9 | Orchestration & End-to-End | CLI, integration tests, TOKENIZER_HANDOFF.md |

---

## 16. Acceptance Criteria

- [ ] Tokenizer trained from Sangraha Hindi Parquet train folder
- [ ] Unigram algorithm with NFKC normalization and Metaspace pre-tokenizer
- [ ] Three vocabulary size variants trained (24k, 32k, 48k)
- [ ] Validation metrics computed for all variants
- [ ] Comparison report generated and best variant selected
- [ ] Final artifact loads cleanly via `AutoTokenizer.from_pretrained`
- [ ] `len(tokenizer)` matches declared `vocab_size`
- [ ] Special tokens `<pad>`, `<unk>`, `<s>`, `</s>` at IDs 0–3
- [ ] No special token splits
- [ ] Unknown token rate < 0.1% on validation sentences
- [ ] Roundtrip success rate > 99%
- [ ] VERSION, README, checksums.json present in final artifact
- [ ] Unit test coverage ≥ 80%
- [ ] All tests pass: `pytest tests/ -v -m "not requires_hf_hub"`
- [ ] Linting clean: `ruff check src/ tests/`
- [ ] TOKENIZER_HANDOFF.md written at monorepo root
