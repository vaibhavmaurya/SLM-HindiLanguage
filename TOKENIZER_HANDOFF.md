# TOKENIZER_HANDOFF.md

Authoritative reference for the frozen Hindi SLM tokenizer. Read this before starting the pretraining workstream.

---

## 1. Artifact Location

```
tokenizer_training/data/final/hindi_slm_tokenizer_v001/
├── tokenizer.json                  ← load this with HuggingFace tokenizers
├── tokenizer_config.json
├── special_tokens_map.json
├── tokenizer_metadata.json
├── tokenizer_validation_report.json
├── tokenizer_comparison_report.md
├── tokenizer_training_config.yaml
├── checksums.json                  ← SHA-256 for every file in this dir
└── VERSION                         ← "hindi_slm_tokenizer_v001"
```

**tokenizer.json SHA-256:** `fbe21c642a4a13030833be48733c1c6b78244e4c0bc077516422b22e7f046cd9`

Verify integrity before use:

```python
import hashlib, json
from pathlib import Path

checksums = json.loads(Path("checksums.json").read_text())
for name, expected in checksums.items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == expected, f"Checksum mismatch: {name}"
```

---

## 2. Tokenizer Specification

| Property | Value |
|---|---|
| Version | `hindi_slm_tokenizer_v001` |
| Algorithm | Unigram (HuggingFace `tokenizers` library) |
| Vocab size | 32,000 |
| Normalizer | NFKC |
| Pre-tokenizer | Metaspace (`▁` boundary marker, SentencePiece-compatible) |
| Training corpus | 5 GB Sangraha Hindi (`ai4bharat/sangraha`, `verified/hin`) |

---

## 3. Special Tokens (frozen — do not change)

| Token | ID | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown |
| `<s>` | 2 | BOS / sequence start |
| `</s>` | 3 | EOS / sequence end |
| `<\|system\|>` | 4 | Chat system turn |
| `<\|user\|>` | 5 | Chat user turn |
| `<\|assistant\|>` | 6 | Chat assistant turn |
| `<\|end\|>` | 7 | Chat turn end |

The SLM embedding matrix must be sized to `len(tokenizer)` = 32,000. These IDs are permanently frozen.

---

## 4. Validation Metrics (32k on 5 GB corpus)

| Metric | Value | Threshold | Pass |
|---|---|---|---|
| `unk_rate` | 0.00000 | < 0.001 | ✓ |
| `chars_per_token` | 4.788 | > 3.0 | ✓ |
| `tokens_per_word` | 1.128 | < 2.5 | ✓ |
| `roundtrip_success_rate` | 1.0000 | > 0.99 | ✓ |
| `devanagari_char_coverage` | 1.00000 | > 0.995 | ✓ |
| `special_token_split_failures` | 0 | == 0 | ✓ |

All 3 trained variants (24k, 32k, 48k) passed. 32k was selected as the recommended variant.

---

## 5. How to Load

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer_training/data/final/hindi_slm_tokenizer_v001/tokenizer.json"
)

# Encode
encoding = tokenizer.encode("हिंदी भाषा में प्रशिक्षण")
print(encoding.ids)      # token IDs
print(encoding.tokens)   # subword tokens

# Decode
text = tokenizer.decode(encoding.ids)
```

Or via the SDK wrapper:

```python
from hindi_tokenizer.sdk.loader import load_tokenizer
from hindi_tokenizer.sdk.encode import encode
from hindi_tokenizer.sdk.decode import decode

tok = load_tokenizer("tokenizer_training/data/final/hindi_slm_tokenizer_v001")
ids = encode(tok, "हिंदी में नमस्ते")
text = decode(tok, ids)
```

---

## 6. Embedding Matrix Size

```python
vocab_size = len(tokenizer.get_vocab())  # 32000
# Use this as nn.Embedding(vocab_size, d_model) in your model
```

---

## 7. Freezing Rules

Once SLM pretraining begins, the following are **permanently prohibited**:

- Modifying `tokenizer.json`, vocab, or merge rules
- Adding, removing, or renaming special tokens
- Changing token IDs
- Changing normalization (NFKC) or pre-tokenizer (Metaspace)
- Retraining or fine-tuning the tokenizer

The `checksums.json` SHA-256 values are the source of truth. Any artifact that does not match these checksums must be rejected.

---

## 8. Variant Comparison Summary

| Variant | Vocab | unk_rate | chars/token | tokens/word | Roundtrip | Selected |
|---|---|---|---|---|---|---|
| hindi_unigram_24k_v001 | 24,000 | 0.0 | 4.660 | 1.159 | 1.000 | |
| **hindi_unigram_32k_v001** | **32,000** | **0.0** | **4.788** | **1.128** | **1.000** | **✓** |
| hindi_unigram_48k_v001 | 48,000 | 0.0 | 4.904 | 1.102 | 1.000 | |

32k balances vocabulary coverage and model embedding size. All three variants passed all thresholds.

---

## 9. HuggingFace Hub Publishing

To publish (requires `HF_TOKEN` env var with write access):

```bash
cd tokenizer_training/
$env:HF_TOKEN="hf_your_token_here"
python -m hindi_tokenizer.orchestration.run_tokenizer --step publish
```

Target repo: `vaibhavmaurya/hindi-slm-tokenizer-v001` (private).

---

## 10. Related Files

| File | Purpose |
|---|---|
| `tokenizer_training/CLAUDE.md` | Tokenizer workstream context for Claude Code |
| `tokenizer_training/DevelopmentPlan.md` | Architecture and design decisions |
| `tokenizer_training/configs/tokenizer_training_config.yaml` | Training configuration used |
| `data_ingestion/CORPUS_HANDOFF.md` | Source corpus specification |
| `tokenizer_training/data/reports/pipeline_run_log.csv` | Full audit trail of all pipeline runs |
