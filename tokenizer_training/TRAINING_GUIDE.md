# Tokenizer Training Guide

Step-by-step instructions for training the Hindi SLM tokenizer from scratch and publishing it to HuggingFace Hub. Written for someone who has the corpus ready and wants to reproduce or retrain.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Setup](#2-project-setup)
3. [Prepare Your Credentials](#3-prepare-your-credentials)
4. [Prepare the Corpus](#4-prepare-the-corpus)
5. [Configure the Training Run](#5-configure-the-training-run)
6. [Run the Pipeline](#6-run-the-pipeline)
7. [Run Individual Steps](#7-run-individual-steps)
8. [Inspect the Outputs](#8-inspect-the-outputs)
9. [Publish to HuggingFace Hub](#9-publish-to-huggingface-hub)
10. [Verify the Published Tokenizer](#10-verify-the-published-tokenizer)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

**Python:** 3.11 or higher

**Hardware:** No GPU required. Tokenizer training is CPU-only. A 5 GB corpus trains in ~2 hours on a modern laptop (8-core CPU).

**Disk space:**
- Input corpus: ~5 GB (Parquet shards)
- Sampled text file: ~5 GB
- Trained artifacts: ~10 MB
- Total: ~12 GB free space recommended

**Corpus:** Parquet files from the data ingestion pipeline. See `../data_ingestion/CORPUS_HANDOFF.md`. The files must be at:
```
../data_ingestion/data/final/parquet/train/*.parquet
```
Each file must have a `final_text` column containing clean Hindi text.

---

## 2. Project Setup

```bash
# Clone the repo (if you don't have it)
git clone https://github.com/vaibhavmaurya/SLM-HindiLanguage.git
cd SLM-HindiLanguage/tokenizer_training/

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install all dependencies
pip install -e ".[dev]"
```

Verify the install:
```bash
python -m hindi_tokenizer.orchestration.run_tokenizer --dry-run
# Expected output:
# Project: hindi-slm-tokenizer
# Config: configs/tokenizer_training_config.yaml
```

---

## 3. Prepare Your Credentials

The publish step requires a HuggingFace token with **write** access.

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — select **Write** role.

2. Create a `.env.local` file in `tokenizer_training/` (this file is gitignored and never committed):

```bash
# tokenizer_training/.env.local
HF_TOKEN=hf_your_token_here
```

The pipeline loads this file automatically at startup. You do **not** need to export it manually.

> **Security note:** `.env.local` is listed in `.gitignore`. Never rename it to `.env` or commit it. The `.env.example` file shows the expected format with placeholder values.

---

## 4. Prepare the Corpus

The corpus sampler reads Parquet shards and writes a single `.txt` file (one document per line) that the tokenizer trainer reads.

**Default config** samples 5 GB from the train split. To verify your corpus is readable before starting:

```bash
python -c "
import pandas as pd
from pathlib import Path

folder = Path('../data_ingestion/data/final/parquet/train')
files = sorted(folder.glob('*.parquet'))
print(f'Found {len(files)} parquet files')

# Read first file and check
df = pd.read_parquet(files[0])
print(f'Columns: {list(df.columns)}')
print(f'Rows in first shard: {len(df)}')
print(f'Sample text: {df[\"final_text\"].iloc[0][:100]}')
"
```

Expected output:
```
Found 47 parquet files
Columns: ['source', 'final_text', 'word_count', ...]
Rows in first shard: 12453
Sample text: भारत एक विशाल देश है...
```

---

## 5. Configure the Training Run

The main config file is `configs/tokenizer_training_config.yaml`. Key settings you may want to change:

```yaml
project:
  name: hindi-slm-tokenizer
  tokenizer_version: hindi_slm_tokenizer_v001   # change for a new version
  hf_repo_id: vaibhavmaurya/hindi-slm-tokenizer-v001  # your HF repo

input:
  parquet_train_folder: ../data_ingestion/data/final/parquet/train
  text_column: final_text

sampling:
  random_seed: 42
  smoke_test:
    target_size_gb: 0.5          # fast test run
    output_file: data/samples/smoke_test/...
  experiment:
    target_size_gb: 5.0          # full training run
    output_file: data/samples/experiment/...

tokenizer:
  vocab_sizes: [24000, 32000, 48000]   # all three are trained and compared
  default_vocab_size: 32000

text_filters:
  min_char_count: 30
  max_char_count: 5000
  min_devanagari_ratio: 0.60     # at least 60% Hindi characters
```

**To change the HuggingFace repo**, update `hf_repo_id` to `your-username/your-repo-name`.

---

## 6. Run the Pipeline

All commands are run from the `tokenizer_training/` directory.

### Option A: Full pipeline in one command

```bash
python -m hindi_tokenizer.orchestration.run_tokenizer
```

This runs all six steps in sequence: **sample → train → validate → compare → package → publish**

Expected total time: ~2.5 hours for a 5 GB corpus (24k + 32k + 48k variants).

### Option B: Smoke test first (recommended)

Run a quick smoke test on 0.5 GB before committing to the full 5 GB run:

```bash
python -m hindi_tokenizer.orchestration.run_tokenizer --smoke-test
```

This uses the `smoke_test` sampling profile (0.5 GB, ~15 minutes total). Check that validation passes before running the full experiment.

### Option C: Dry run (config validation only)

```bash
python -m hindi_tokenizer.orchestration.run_tokenizer --dry-run
```

Validates the config and prints a summary. No files are written.

---

## 7. Run Individual Steps

Each step can be run independently. Useful for resuming a failed run or re-running just one stage.

```bash
# Step 1 — Sample corpus text from Parquet shards
python -m hindi_tokenizer.orchestration.run_tokenizer --step sample

# Step 2 — Train tokenizer variants (24k, 32k, 48k)
python -m hindi_tokenizer.orchestration.run_tokenizer --step train

# Step 3 — Validate each trained variant
python -m hindi_tokenizer.orchestration.run_tokenizer --step validate

# Step 4 — Compare variants and select recommended
python -m hindi_tokenizer.orchestration.run_tokenizer --step compare

# Step 5 — Package the recommended variant into the final artifact dir
python -m hindi_tokenizer.orchestration.run_tokenizer --step package

# Step 6 — Publish to HuggingFace Hub
python -m hindi_tokenizer.orchestration.run_tokenizer --step publish
```

**To force retrain** (deletes existing artifacts and starts fresh):
```bash
python -m hindi_tokenizer.orchestration.run_tokenizer --step train --force-retrain
```

**To use a custom config:**
```bash
python -m hindi_tokenizer.orchestration.run_tokenizer --config configs/my_custom_config.yaml
```

---

## 8. Inspect the Outputs

After a successful run, the following files are created:

### Trained artifacts (all three variants)
```
data/artifacts/
├── vocab_24000/
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── vocab_32000/
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── vocab_48000/
    ├── tokenizer.json
    └── tokenizer_config.json
```

### Validation reports
```
data/reports/
├── validation_vocab_24000.json
├── validation_vocab_32000.json
├── validation_vocab_48000.json
└── tokenizer_comparison_report.md     ← open this to see which variant won
```

View the comparison report:
```bash
cat data/reports/tokenizer_comparison_report.md
```

Example output:
```
| variant                  | vocab_size | unk_rate | chars_per_token | tokens_per_word | passes |
|--------------------------|------------|----------|-----------------|-----------------|--------|
| hindi_unigram_24k_v001   | 24000      | 0.000000 | 4.660           | 1.159           | True   |
| hindi_unigram_32k_v001   | 32000      | 0.000000 | 4.788           | 1.128           | True   |
| hindi_unigram_48k_v001   | 48000      | 0.000000 | 4.904           | 1.102           | True   |

Recommended: hindi_unigram_32k_v001
```

### Final packaged artifact
```
data/final/hindi_slm_tokenizer_v001/
├── tokenizer.json                    ← load this with HuggingFace tokenizers
├── tokenizer_config.json
├── special_tokens_map.json
├── tokenizer_metadata.json
├── tokenizer_validation_report.json
├── tokenizer_comparison_report.md
├── tokenizer_training_config.yaml
├── checksums.json                    ← SHA-256 for every file
├── VERSION
└── README.md
```

Quick smoke test after training:
```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("data/final/hindi_slm_tokenizer_v001/tokenizer.json")
enc = tok.encode("नमस्ते, यह हिंदी टोकनाइज़र है।")
print(enc.tokens)
print(tok.decode(enc.ids))   # must match input exactly
```

### Observability logs
```
data/reports/pipeline_run_log.csv      ← every step with timing and record counts
data/reports/data_file_registry.csv   ← every file produced, with SHA-256
```

```bash
# Check if all steps completed successfully
python -c "
import pandas as pd
df = pd.read_csv('data/reports/pipeline_run_log.csv')
print(df[['phase', 'component', 'status', 'duration_seconds']].to_string())
"
```

---

## 9. Publish to HuggingFace Hub

### Before publishing — checklist

- [ ] `.env.local` exists with a valid `HF_TOKEN` (write access)
- [ ] Validation report shows `passes_thresholds: true`
- [ ] `data/final/hindi_slm_tokenizer_v001/` exists and contains `tokenizer.json`
- [ ] `README.md` in the artifact dir is up to date

### Create the HF repo (first time only)

The pipeline creates the repo automatically if it does not exist. The repo is created as **private** by default.

To create it manually via the HuggingFace website: go to [huggingface.co/new](https://huggingface.co/new) → select Model → set name → set Private.

### Publish

```bash
python -m hindi_tokenizer.orchestration.run_tokenizer --step publish
```

The pipeline:
1. Creates the repo at `hf_repo_id` if it does not exist (private)
2. Uploads all files from `data/final/hindi_slm_tokenizer_v001/` to the repo
3. Logs the event to `data/reports/pipeline_run_log.csv`

Expected output:
```
Run ID: a1b2c3d4-...
Uploading to vaibhavmaurya/hindi-slm-tokenizer-v001 ...
Done.
```

### Make the repo public (optional)

After publishing, go to the repo on HuggingFace → **Settings** → scroll to **Repository visibility** → click **Make public**.

Or via API:
```python
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv(".env.local")

api = HfApi(token=os.environ["HF_TOKEN"])
api.update_repo_visibility(
    repo_id="vaibhavmaurya/hindi-slm-tokenizer-v001",
    private=False,
)
```

### Update files on Hub without rerunning the full pipeline

To push only the README (e.g. after editing the model card):
```python
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv(".env.local")

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj="data/final/hindi_slm_tokenizer_v001/README.md",
    path_in_repo="README.md",
    repo_id="vaibhavmaurya/hindi-slm-tokenizer-v001",
    repo_type="model",
    commit_message="docs: update model card",
)
```

---

## 10. Verify the Published Tokenizer

After publishing, verify the tokenizer loads correctly from the Hub:

```python
import os
from dotenv import load_dotenv
load_dotenv("tokenizer_training/.env.local")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "vaibhavmaurya/hindi-slm-tokenizer-v001",
    token=os.environ["HF_TOKEN"],   # only needed while repo is private
)

# Check vocab size
print(tok.vocab_size)   # 32000

# Check special token IDs
print(tok.pad_token_id)   # 0
print(tok.unk_token_id)   # 1
print(tok.bos_token_id)   # 2
print(tok.eos_token_id)   # 3

# Encode and decode
ids = tok.encode("हिंदी भाषा में प्रशिक्षण")
print(ids)
print(tok.decode(ids, skip_special_tokens=True))

# Verify integrity of tokenizer.json on Hub
from huggingface_hub import hf_hub_download
import hashlib

path = hf_hub_download(
    repo_id="vaibhavmaurya/hindi-slm-tokenizer-v001",
    filename="tokenizer.json",
    token=os.environ["HF_TOKEN"],
)
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
expected = "fbe21c642a4a13030833be48733c1c6b78244e4c0bc077516422b22e7f046cd9"
print("Checksum OK" if sha == expected else f"MISMATCH: {sha}")
```

---

## 11. Troubleshooting

### `401 Unauthorized` when publishing

Your `HF_TOKEN` is invalid or expired.
- Check `tokenizer_training/.env.local` — the token must start with `hf_`
- Regenerate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### `404 RepositoryNotFound` when publishing

The `hf_repo_id` in `configs/tokenizer_training_config.yaml` must be in `username/repo-name` format — not just `repo-name`.

```yaml
# Wrong
hf_repo_id: hindi-slm-tokenizer-v001

# Correct
hf_repo_id: vaibhavmaurya/hindi-slm-tokenizer-v001
```

### Validation fails (`passes_thresholds: false`)

Check which metric failed:
```bash
cat data/reports/validation_vocab_32000.json
```

Common causes:
- `unk_rate > 0` — corpus has non-Hindi characters not covered by vocab. Check `min_devanagari_ratio` in config.
- `roundtrip_success_rate < 1.0` — look for special characters in `tests/fixtures/validation_sentences.txt` that may not round-trip (e.g. em dash `—` vs hyphen `-`).
- `chars_per_token < 3.0` — corpus is too small or too noisy. Increase `target_size_gb`.

### Training is very slow

Normal training time on a laptop CPU:
- 24k vocab, 5 GB corpus: ~35 min
- 32k vocab, 5 GB corpus: ~41 min
- 48k vocab, 5 GB corpus: ~48 min

If it is running much slower, check that no other heavy processes are using the CPU. The HuggingFace `tokenizers` library uses all available CPU cores automatically.

### `ModuleNotFoundError: No module named 'hindi_tokenizer'`

You are not in the `tokenizer_training/` directory, or the package is not installed.

```bash
cd tokenizer_training/
pip install -e ".[dev]"
```

### Corpus file not found

```
FileNotFoundError: .../parquet/train/*.parquet
```

The `parquet_train_folder` in config points to the data ingestion output. Run the data ingestion pipeline first, or update the path in `configs/tokenizer_training_config.yaml` to point to where your Parquet files actually are.

---

## Quick Reference

```bash
# Setup
pip install -e ".[dev]"
echo "HF_TOKEN=hf_your_token" > .env.local

# Smoke test (0.5 GB, ~15 min)
python -m hindi_tokenizer.orchestration.run_tokenizer --smoke-test

# Full run (5 GB, ~2.5 hours)
python -m hindi_tokenizer.orchestration.run_tokenizer

# Individual steps
python -m hindi_tokenizer.orchestration.run_tokenizer --step sample
python -m hindi_tokenizer.orchestration.run_tokenizer --step train
python -m hindi_tokenizer.orchestration.run_tokenizer --step validate
python -m hindi_tokenizer.orchestration.run_tokenizer --step compare
python -m hindi_tokenizer.orchestration.run_tokenizer --step package
python -m hindi_tokenizer.orchestration.run_tokenizer --step publish

# Force retrain from scratch
python -m hindi_tokenizer.orchestration.run_tokenizer --step train --force-retrain

# Dry run (no writes)
python -m hindi_tokenizer.orchestration.run_tokenizer --dry-run
```
