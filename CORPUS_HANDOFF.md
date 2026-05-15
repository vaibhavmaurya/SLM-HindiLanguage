# CORPUS_HANDOFF.md — Hindi Corpus Reference for Downstream Phases

This document is the single source of truth about the Hindi corpus produced by the data ingestion pipeline. Tokenizer training and SLM pretraining should read this file to understand what data is available, where it lives, and how to load it.

---

## 1. What Was Produced

The data ingestion pipeline produces a clean, deduplicated, split Hindi corpus from three sources:

| Source | Description | Cleaning Method |
|---|---|---|
| AI4Bharat Sangraha | HuggingFace dataset `ai4bharat/sangraha`, subset `verified/hin` | Deterministic Unicode normalization |
| User-provided PDFs | Scanned/digital Hindi books and documents | Ollama + Qwen3 model-assisted cleaning |
| Hindi Wikipedia | BFS crawl of Hindi Wikipedia articles | Deterministic normalization |

Output splits: **train** (98%) · **validation** (1%) · **test** (1%) — document-level stratified, reproducible via `random_seed: 42`.

---

## 2. Output File Locations

All paths are relative to `data_ingestion/data/` (or absolute from wherever `DATA_ROOT` was set during ingestion).

### 2.1 Final Corpus — Primary Formats

```
data_ingestion/data/final/
├── parquet/
│   ├── train/
│   │   └── hindi_corpus_v001_train_00000.parquet   (+ more shards)
│   ├── validation/
│   │   └── hindi_corpus_v001_validation_00000.parquet
│   └── test/
│       └── hindi_corpus_v001_test_00000.parquet
├── training_jsonl/
│   ├── train/
│   │   └── hindi_corpus_v001_train_00000.jsonl.gz
│   ├── validation/
│   └── test/
└── training_text/
    ├── train/
    │   └── hindi_corpus_v001_train_00000.txt.gz
    ├── validation/
    └── test/
```

**Naming convention:** `hindi_corpus_v001_{split}_{shard_index:05d}.{ext}`

- Parquet: zstd-compressed, ~512 MB per shard (configurable)
- JSONL: one JSON object per line, gzip-compressed
- TXT: one document per line, gzip-compressed

### 2.2 Manifest and Profile

```
data_ingestion/data/final/
├── hindi_corpus_v001_manifest.json     # SHA-256 per file + metadata
└── hindi_corpus_v001_profile.json      # Corpus statistics
```

**Manifest fields:** `corpus_version`, `created_at`, `run_id`, `splits` (with file list, sha256, size_bytes, row_count per file).

**Profile fields:** total records, total characters, total words, records per source, records per split, average document length, character distribution.

### 2.3 Observability Reports

```
data_ingestion/data/reports/
├── pipeline_run_log.csv        # Every pipeline event across all runs (append-only)
├── data_file_registry.csv      # Every file produced/consumed (append-only)
├── model_cleaning_report.csv   # Per-record cleaning outcome (PDF source only)
├── duplicate_report.csv        # Dedup cluster summary
├── split_report.json           # Final split sizes and ratios
├── quality_report.md           # Human-readable quality summary
└── corpus_profile.json         # Per-run corpus statistics
```

---

## 3. CorpusRecord Schema

Every row in Parquet (and object in JSONL) conforms to the following schema. Defined in `data_ingestion/src/slm_hindi/schema/corpus_record.py`.

| Field | Type | Description |
|---|---|---|
| `record_id` | str | Unique ID for this paragraph/record |
| `document_id` | str | Groups records from the same source document |
| `paragraph_id` | str | Paragraph index within the document |
| `source_type` | `"huggingface_dataset"` \| `"pdf"` \| `"wiki"` | Origin of the record |
| `source_name` | str | Human-readable source label |
| `source_dataset` | str \| null | HuggingFace dataset name (Sangraha only) |
| `source_file_name` | str \| null | PDF filename (PDF source only) |
| `source_url_or_path` | str \| null | Wikipedia URL or PDF path |
| `page_number` | int \| null | PDF page number (PDF source only) |
| `raw_text` | str | Text before any cleaning |
| `cleaned_text` | str \| null | Text after model-assisted cleaning (PDF only) |
| `final_text` | str | **The training text** — normalized, validated |
| `language` | str | Always `"hi"` |
| `script` | str | Always `"Devanagari"` |
| `char_count` | int | Character count of `final_text` |
| `word_count` | int | Whitespace-split word count |
| `estimated_token_count` | int | ~4 chars/token estimate |
| `devanagari_ratio` | float | Fraction of Devanagari characters in `final_text` |
| `latin_ratio` | float | Fraction of Latin characters |
| `digit_ratio` | float | Fraction of digit characters |
| `symbol_ratio` | float | Fraction of symbol characters |
| `quality_score` | float | Composite quality score (0.0–1.0) |
| `cleaning_method` | `"deterministic_normalization"` \| `"ollama_model_assisted"` \| `"none"` | How text was cleaned |
| `cleaning_model` | str \| null | Ollama model name used (PDF only) |
| `cleaning_model_version` | str \| null | Model version |
| `cleaning_status` | `"pending"` \| `"clean"` \| `"quarantined"` \| `"skipped"` | Outcome of cleaning validation |
| `dedup_hash` | str | SHA-256 of `final_text` (exact dedup key) |
| `near_dedup_cluster_id` | str \| null | MinHash LSH cluster ID |
| `split_name` | `"train"` \| `"validation"` \| `"test"` \| null | Corpus split assignment |
| `created_at` | str | ISO-8601 UTC timestamp of record creation |
| `ingestion_run_id` | str | UUID of the pipeline run that produced this record |

**The field to use for training is `final_text`.** All other fields are metadata.

---

## 4. Loading the Corpus

### 4.1 Load Parquet with pandas

```python
import pandas as pd
from pathlib import Path

data_root = Path("data_ingestion/data/final/parquet")

# Load all train shards
train_df = pd.concat(
    [pd.read_parquet(f) for f in sorted((data_root / "train").glob("*.parquet"))],
    ignore_index=True,
)

# Extract training text
texts = train_df["final_text"].tolist()
```

### 4.2 Load Parquet with PyArrow (memory-efficient)

```python
import pyarrow.parquet as pq
from pathlib import Path

shard_path = "data_ingestion/data/final/parquet/train/hindi_corpus_v001_train_00000.parquet"
table = pq.read_table(shard_path, columns=["final_text", "source_type", "word_count"])
df = table.to_pandas()
```

### 4.3 Load JSONL for tokenizer training

```python
import gzip, json
from pathlib import Path

def iter_jsonl_texts(split: str):
    jsonl_dir = Path(f"data_ingestion/data/final/training_jsonl/{split}")
    for shard in sorted(jsonl_dir.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["final_text"]
```

### 4.4 Load TXT shards (one doc per line)

```python
import gzip
from pathlib import Path

def iter_txt_lines(split: str):
    txt_dir = Path(f"data_ingestion/data/final/training_text/{split}")
    for shard in sorted(txt_dir.glob("*.txt.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line
```

### 4.5 Load via HuggingFace datasets (from local Parquet)

```python
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "data_ingestion/data/final/parquet/train/*.parquet",
        "validation": "data_ingestion/data/final/parquet/validation/*.parquet",
        "test": "data_ingestion/data/final/parquet/test/*.parquet",
    },
)
# Access: dataset["train"]["final_text"]
```

---

## 5. Filtering and Subsetting

All quality metadata is available for filtering at load time:

```python
# Only high-quality Sangraha records with >200 words
filtered = train_df[
    (train_df["source_type"] == "huggingface_dataset") &
    (train_df["word_count"] >= 200) &
    (train_df["devanagari_ratio"] >= 0.7)
]

# Exclude PDF source if Ollama cleaning was not run
no_pdf = train_df[train_df["source_type"] != "pdf"]
```

---

## 6. Data Quality Thresholds Applied

The following filters were applied during ingestion (values from `configs/quality_filter_config.yaml`):

| Filter | Threshold |
|---|---|
| Minimum character count | Configurable (default ~50) |
| Maximum character count | Configurable (default ~50,000) |
| Minimum Devanagari ratio | Configurable (default 0.5) |
| Exact duplicate removal | SHA-256 hash |
| Near-duplicate removal | MinHash LSH (datasketch) |
| Cleaning validation (PDF) | 7 checks: empty, length ratio, Devanagari ratio, hallucination markers, repetition, prompt echo, language |

Records that failed cleaning validation were quarantined to:
```
data_ingestion/data/model_cleaned/pdf/rejected_model_outputs.parquet
```

---

## 7. Corpus Versioning

The corpus version `hindi_corpus_v001` is set in `data_ingestion/configs/ingestion_config.yaml`:

```yaml
project:
  corpus_version: hindi_corpus_v001
```

All output filenames, the manifest, and the profile embed this version string. To produce a new version (e.g., after adding more sources), bump to `hindi_corpus_v002` before re-running the pipeline.

---

## 8. Verifying Integrity

```python
import json, hashlib
from pathlib import Path

manifest = json.loads(Path("data_ingestion/data/final/hindi_corpus_v001_manifest.json").read_text())

for split_name, files in manifest["splits"].items():
    for entry in files:
        path = Path(entry["file_path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == entry["sha256"], f"Checksum mismatch: {path}"
        print(f"OK  {path.name}")
```

---

## 9. Re-running the Pipeline

If you need to regenerate the corpus (e.g., add more Wikipedia seeds or PDFs):

```bash
cd data_ingestion/

# Add sources to configs/ingestion_config.yaml, then:
./run_pipeline.sh --source all          # Linux/macOS
run_pipeline.bat --source all           # Windows

# Or a single source only:
./run_pipeline.sh --source wiki
```

Each run appends to the observability CSVs and produces a new `run_id`. Output files are overwritten in `data/final/` (versioned by corpus_version).

---

## 10. Next Phase Pointers

| Phase | Location | What it needs from here |
|---|---|---|
| Tokenizer Training | `tokenizer_training/` | `final_text` from all train shards |
| SLM Pretraining | `slm_training/` | Tokenized train/val splits; vocab from tokenizer |
