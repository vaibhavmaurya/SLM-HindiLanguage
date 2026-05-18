# TOKENIZER_HANDOFF.md

Authoritative reference for the Hindi SLM tokenizer. Written for someone encountering this tokenizer for the first time — starts from first principles, covers every design decision, every test, and every artifact.

---

## Table of Contents

1. [What Is a Tokenizer?](#1-what-is-a-tokenizer)
2. [What Is a Vocabulary?](#2-what-is-a-vocabulary)
3. [Algorithm: Unigram Language Model](#3-algorithm-unigram-language-model)
4. [Why Unigram, Not BPE or WordPiece?](#4-why-unigram-not-bpe-or-wordpiece)
5. [Preprocessing: Normalization and Pre-tokenization](#5-preprocessing-normalization-and-pre-tokenization)
6. [Training Corpus](#6-training-corpus)
7. [Tokenizer Specification](#7-tokenizer-specification)
8. [Special Tokens](#8-special-tokens)
9. [Vocabulary: Size, Memory, and Token Distribution](#9-vocabulary-size-memory-and-token-distribution)
10. [Validation: What Was Tested and Why](#10-validation-what-was-tested-and-why)
11. [Test Suite (207 Automated Tests)](#11-test-suite-207-automated-tests)
12. [Variant Comparison (24k vs 32k vs 48k)](#12-variant-comparison-24k-vs-32k-vs-48k)
13. [Artifact Directory](#13-artifact-directory)
14. [What Is on HuggingFace Hub](#14-what-is-on-huggingface-hub)
15. [How to Load and Use](#15-how-to-load-and-use)
16. [Embedding Matrix Size for Pretraining](#16-embedding-matrix-size-for-pretraining)
17. [Freezing Rules](#17-freezing-rules)
18. [Integrity Verification](#18-integrity-verification)
19. [Related Files](#19-related-files)
20. [Reference Links](#20-reference-links)

---

## 1. What Is a Tokenizer?

A neural language model cannot operate directly on raw text — it operates on numbers. A tokenizer bridges that gap: it converts a string of text into a sequence of integer IDs, and converts integer IDs back into text.

```
Input text:  "हिंदी भाषा में सीखना"
Tokens:      ["▁हिंदी", "▁भाषा", "▁में", "▁सीखना"]
Token IDs:   [1423, 892, 45, 3201]
```

The inverse (IDs → text) is called decoding. A tokenizer must satisfy one critical property: **lossless round-trip** — decoding the token IDs produced by encoding must recover the original text exactly.

The tokenizer is trained *before* the language model and is frozen for the entire lifetime of the model. Changing the tokenizer after pretraining begins invalidates the embedding matrix — the learned correspondence between token IDs and their vector representations — which means you would have to retrain from scratch.

**Key insight for Hindi:** Hindi's Devanagari script is morphologically rich (words have many inflected forms) and each Devanagari character takes 3 UTF-8 bytes (Unicode range U+0900–U+097F). A good Hindi tokenizer must learn whole words and common morphological patterns as single tokens so the model sees meaning-carrying units rather than byte fragments.

---

## 2. What Is a Vocabulary?

The vocabulary is the fixed set of all tokens the tokenizer can produce. It is an ordered list:

```
ID 0  → "<pad>"
ID 1  → "<unk>"
ID 2  → "<s>"
...
ID 8  → "▁है"
ID 9  → "▁में"
...
ID 31999 → (last learned subword)
```

Every token in the vocabulary has:
- An **integer ID** — this is what the model sees.
- A **surface form** — the actual string (a word, subword, character, or special marker).
- A **score** (unigram probability) — used during tokenization to find the best segmentation.

The vocabulary is stored in `tokenizer.json` as a JSON array. This file is 2.21 MB. Each entry costs an average of **72 bytes** in the file (token string in JSON, score as float, type field).

When a text contains a character not in the vocabulary, the tokenizer emits `<unk>` (ID 1). Minimising the `unk_rate` is a primary training objective.

**Vocabulary size trade-off:**

| Smaller vocab | Larger vocab |
|---|---|
| Fewer parameters in embedding matrix | More parameters in embedding matrix |
| Shorter tokens → longer sequences → more compute | Longer tokens → shorter sequences → less compute |
| More `<unk>` for rare words | Fewer `<unk>`, better coverage |
| Less morphological precision | Better morphological precision |

For Hindi at the SLM scale (hundreds of millions of parameters), 32,000 is standard. It matches LLaMA, Mistral, and similar models' vocab sizes and is well-studied empirically.

---

## 3. Algorithm: Unigram Language Model

The Unigram algorithm (also called SentencePiece Unigram) was introduced by Kudo (2018) and is the algorithm used in LLaMA, Gemma, Mistral, and most modern open-source LLMs.

### Core idea

Rather than building up vocabulary by merging (as BPE does), Unigram starts with a large candidate vocabulary (~300,000 substrings) and iteratively *prunes* it down to the target size.

At each step, it assigns a probability to each token using the **unigram language model**:

```
P(sentence) = ∏ P(token_i)
```

where the best tokenization is found using the Viterbi algorithm (dynamic programming over all possible segmentations of the sentence). The training objective is to maximize the log-likelihood of the corpus under this model.

The pruning criterion is: for each vocabulary entry, compute how much the total log-likelihood would drop if that entry were removed. Entries with the smallest impact are removed first.

### Training steps

1. Seed a candidate vocabulary with all substrings up to `max_piece_length` (24 chars in our config) that appear in the corpus above a frequency threshold.
2. Run the EM (Expectation-Maximization) algorithm:
   - **E step:** For each sentence, use Viterbi to find the probability distribution over all valid segmentations.
   - **M step:** Update token probabilities using the expected counts from the E step.
3. Remove the bottom 20% of tokens by log-likelihood impact.
4. Repeat until the vocabulary reaches the target size.
5. Run `n_sub_iterations=2` EM steps between each pruning round to re-stabilize probabilities.

### Why EM + Viterbi instead of simple frequency counting?

Simple frequency counting (as in BPE) cannot model the *competition* between tokens. For example, the Hindi word "प्रशिक्षण" can be split as:
- `["प्र", "शिक्षण"]`
- `["प्रशिक्षण"]` (single token if in vocab)
- `["प्र", "शिक्", "षण"]`

BPE makes greedy decisions: once two subwords are merged, that merge is permanent. Unigram models *all* possible segmentations probabilistically and picks the maximum-probability one at inference time, which produces better segmentations for morphologically complex languages like Hindi.

### Inference: the Viterbi segmentation

At inference, given input text, the tokenizer runs the Viterbi algorithm to find the segmentation `[t_1, ..., t_n]` that maximizes `∑ log P(t_i)`. This is O(n × V) per sentence where n is character length and V is vocabulary size — fast in practice because the Rust implementation in the `tokenizers` library is highly optimized.

**Reference:** [Kudo (2018) — SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)

---

## 4. Why Unigram, Not BPE or WordPiece?

The three major subword algorithms in use today are BPE, WordPiece, and Unigram. Here is how they differ:

| Property | BPE | WordPiece | Unigram |
|---|---|---|---|
| Direction | Bottom-up (merge) | Bottom-up (merge) | Top-down (prune) |
| Selection criterion | Most frequent pair | Highest likelihood gain | Lowest likelihood loss |
| Segmentation | Deterministic, greedy | Deterministic, greedy | Probabilistic (Viterbi) |
| Reversibility | Yes (merge rules file) | Yes | Yes (scores in vocab) |
| Used by | GPT-2, GPT-4, LLaMA-2 | BERT, DistilBERT | LLaMA-3, Mistral, Gemma |
| Hindi suitability | Good | Decent | Best for morphology |

**Why Unigram for Hindi:**

Hindi has a highly productive morphology — a single root word can generate dozens of forms through suffixation, sandhi (sound merging at word boundaries), and compounding. BPE's greedy merges can fail on rare inflected forms, producing many small pieces. Unigram's probabilistic Viterbi decoding naturally handles unseen combinations by composing tokens it has seen.

The zero `unk_rate` on our 20-sentence validation set, and the `4.788 chars/token` metric, confirm that the Unigram model is producing long, meaningful tokens rather than fragmenting words.

**References:**
- [Sennrich et al. (2016) — BPE for NMT](https://arxiv.org/abs/1508.07909)
- [Schuster & Nakamura (2012) — WordPiece (Google internal, cited in BERT paper)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
- [Kudo & Richardson (2018) — SentencePiece](https://arxiv.org/abs/1808.06226)
- [HuggingFace Tokenizers docs — Unigram model](https://huggingface.co/docs/tokenizers/components#models)

---

## 5. Preprocessing: Normalization and Pre-tokenization

Before the Unigram algorithm sees any text, two preprocessing steps run:

### 5.1 NFKC Normalization

**What it does:** Converts text to Unicode NFKC (Normalization Form KC — Compatibility Decomposition followed by Canonical Composition).

**Why it matters for Hindi:**
- Devanagari has multiple Unicode representations for visually identical characters (e.g. different ways to encode anusvara, halant, nukta).
- NFKC canonicalizes all of these to a single form.
- Without NFKC, the same visual word could map to different token sequences depending on which keyboard or input method generated it — breaking the vocabulary.
- Compatibility decomposition handles special ligatures, half-forms, and presentation variants.

Example: the character `ﬁ` (U+FB01, fi ligature) → NFKC → `fi` (two chars). For Devanagari, characters like `ॐ` (Om) are preserved as-is since they are canonical.

**What NFKC does NOT do:** It does not remove characters, change word meanings, or alter the language. It is a purely structural normalization.

**Reference:** [Unicode Standard — Chapter 3.11 Normalization Forms](https://unicode.org/reports/tr15/)

### 5.2 Metaspace Pre-tokenizer

**What it does:** Replaces whitespace with a special prefix character `▁` (U+2581, LOWER ONE EIGHTH BLOCK, called "metaspace") and splits on whitespace.

Input: `"हिंदी भाषा में"`
After metaspace: `["▁हिंदी", "▁भाषा", "▁में"]`

**Why it matters:**
- The tokenizer must know which tokens appear at word boundaries. Without this, `"है"` at the start of a word and `"है"` in the middle of a word would be identical to the model.
- The `▁` prefix encodes the information "this token starts a new word" directly into the token surface form.
- This is identical to SentencePiece's approach, making the tokenizer **SentencePiece-compatible** — a practical advantage for tooling.

**Decoder:** The Metaspace decoder removes `▁` and converts back to spaces during decoding, ensuring lossless round-trip.

**Reference:** [Kudo (2018) SentencePiece paper, Section 3.4](https://arxiv.org/abs/1808.06226)

---

## 6. Training Corpus

| Property | Value |
|---|---|
| Source | [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) |
| Subset | `verified/hin` (human-curated Hindi, highest quality tier) |
| Corpus size | 5.0 GB sampled text |
| Full dataset size | ~10 GB for verified Hindi subset |
| Sampling method | Stratified random sampling across parquet shards |
| Random seed | 42 (reproducible) |
| Text column | `final_text` |
| Min char count | 30 (filter very short texts) |
| Max char count | 5000 (filter abnormally long texts) |
| Min Devanagari ratio | 60% (filter romanized or mixed texts) |
| Sample file | `data/samples/experiment/sangraha_tokenizer_sample_5gb.txt` |

**Why Sangraha?** Sangraha is the highest-quality Hindi corpus publicly available, compiled by AI4Bharat specifically for language model training. The `verified` tier underwent human annotation and quality checks. This is the same data used to train the Indic-series models.

**Corpus pipeline:** The data ingestion pipeline (see `data_ingestion/CORPUS_HANDOFF.md`) produced cleaned Parquet shards. The corpus sampler reads these shards and writes a single `.txt` file (one document per line) consumed by the tokenizer trainer.

**Note on corpus size for tokenizer training:** A 5 GB corpus is sufficient — tokenizer training saturates at a few billion characters. Increasing to 50 GB does not meaningfully change the vocabulary; it only changes training time. The 5 GB corpus was chosen to balance statistical coverage with training time (~2 hours on CPU for all 3 variants).

---

## 7. Tokenizer Specification

| Property | Value |
|---|---|
| Version identifier | `hindi_slm_tokenizer_v001` |
| Algorithm | Unigram Language Model |
| Library | HuggingFace `tokenizers` 0.21.x (Rust-backed) |
| Vocabulary size | 32,000 |
| Normalizer | NFKC |
| Pre-tokenizer | Metaspace (`▁`, add_prefix_space=True) |
| Decoder | Metaspace |
| Max piece length | 24 characters |
| EM sub-iterations per pruning round | 2 |
| Model max length (for downstream use) | 2048 tokens |
| Training corpus | 5 GB Sangraha Hindi |
| Training date | 2026-05-18 |

---

## 8. Special Tokens

These 8 tokens are **permanently frozen**. Their IDs must never change — the pretraining model's embedding matrix is indexed by these IDs.

| Token | ID | Type | Purpose |
|---|---|---|---|
| `<pad>` | 0 | Padding | Pads shorter sequences in a batch to equal length |
| `<unk>` | 1 | Unknown | Emitted when a character has no valid tokenization |
| `<s>` | 2 | BOS | Beginning-of-sequence marker |
| `</s>` | 3 | EOS | End-of-sequence marker |
| `<\|system\|>` | 4 | Chat | Marks the system prompt in chat format |
| `<\|user\|>` | 5 | Chat | Marks a user turn in chat format |
| `<\|assistant\|>` | 6 | Chat | Marks an assistant turn in chat format |
| `<\|end\|>` | 7 | Chat | Marks the end of a chat turn |

**Chat format example:**
```
<s><|system|>आप एक सहायक हैं।<|end|><|user|>नमस्ते<|end|><|assistant|>नमस्ते! मैं आपकी सहायता कर सकता हूँ।<|end|></s>
```

Token IDs 0–7 are reserved. The learned vocabulary begins at ID 8.

---

## 9. Vocabulary: Size, Memory, and Token Distribution

### 9.1 File size

| Artifact | Size |
|---|---|
| `tokenizer.json` (full tokenizer definition) | 2.21 MB (2,313,907 bytes) |
| Per-vocabulary-entry cost in file | ~72 bytes |
| `tokenizer_config.json` | 337 bytes |
| `special_tokens_map.json` | 213 bytes |
| Total artifact directory | ~2.3 MB |

The 72 bytes/entry in `tokenizer.json` breaks down as: ~30 bytes for the token string (JSON-encoded, most Hindi tokens are 1–4 Devanagari syllables = 3–12 bytes each but the surrounding JSON structure adds overhead), ~20 bytes for the score (float as string), and ~22 bytes for JSON structure (`{"content":"...","id":...,`).

### 9.2 Token byte-length distribution

Each Devanagari character is 3 bytes in UTF-8. A typical Hindi syllable (akshara) is 1–3 Devanagari characters = 3–9 bytes. The `▁` prefix is 3 bytes. So a token representing one whole syllable with a word-boundary marker = ~6–12 bytes.

| Token length (UTF-8 bytes) | Count | % of vocab | What this typically represents |
|---|---|---|---|
| 1–3 | 96 | 0.3% | ASCII chars, punctuation, digits |
| 4–6 | 2,143 | 6.7% | 1–2 char Latin/ASCII subwords |
| 9 (3 Devanagari chars) | ~2,283 | 7.1% | Single Devanagari syllable |
| 12 (4 Devanagari chars) | ~3,310 | 10.3% | Syllable with `▁` prefix |
| 15 (5 Devanagari chars) | ~4,409 | 13.8% | Common 1–2 syllable words |
| 18 (6 Devanagari chars) | ~4,350 | 13.6% | Common 2-syllable words |
| 21 (7 Devanagari chars) | ~3,534 | 11.0% | 2–3 syllable words |
| 24 (8 Devanagari chars) | ~2,395 | 7.5% | 3-syllable words |
| 27–48 bytes | ~9,630 | 30.1% | Longer words and compounds |

**Key observation:** 80% of the vocabulary (25,599 tokens) contain at least one Devanagari character. The remaining 20% covers numerals, Latin script, punctuation, and special tokens. This matches the target use case of a primarily Hindi model with English and numeric coverage.

### 9.3 Runtime memory (embedding matrix)

The single largest memory consumer at inference and training is the embedding matrix `nn.Embedding(32000, d_model)`.

| d_model | FP32 size | BF16/FP16 size | Notes |
|---|---|---|---|
| 256 | 31.2 MB | 15.6 MB | Very small model |
| 512 | 62.5 MB | 31.2 MB | Small SLM |
| 1024 | 125.0 MB | 62.5 MB | Medium SLM |
| 2048 | 250.0 MB | 125.0 MB | Large SLM |

For pretraining in BF16, a d_model=512 embedding matrix costs **31.2 MB**. There are actually two embedding matrices in a transformer (input embeddings + output projection/lm_head), so double these numbers for total embedding memory.

### 9.4 Token fertility (tokens per word)

**1.128 tokens per word** means that on average, each Hindi word is tokenized into just 1.128 pieces. This is excellent — close to the theoretical minimum of 1.0 (every word is a single token). By comparison, a character-level tokenizer would produce ~4–8 tokens per word for Hindi, massively inflating sequence length and compute.

For a 512-token context window, the model can cover approximately `512 / 1.128 ≈ 454` Hindi words — roughly a medium-length paragraph. With a 2048-token context, that is ~1,815 words.

---

## 10. Validation: What Was Tested and Why

Validation ran on 20 manually selected Hindi sentences covering:

| Sentence theme | Purpose |
|---|---|
| Simple declarative sentences about India | Common vocabulary baseline |
| Hindi as national language | Core domain vocabulary |
| Historical figures (Gandhi) | Named entities |
| Mixed Hindi+English abbreviations (UPI, GST, ISRO) | Code-switching coverage |
| Classical literature references (Ramayana, Mahabharata) | Literary vocabulary |
| Sacred geography (Ganga, Yamuna) | Cultural vocabulary |
| Quoted speech | Punctuation round-trip |
| Competitive exam sentences | Complex syntax |
| Historical dates | Numerals with Hindi context |
| Morphologically complex words (प्रशिक्षण/प्रशिक्षित) | Morphological test |
| Conversational Hindi with Devanagari punctuation (।॥) | Danda handling |
| Pure numerals (1234567890) | Number tokenization |
| Space agency names (NASA) | English proper nouns |
| Geographic sentences | Complex compounds |
| Long compound sentences (20+ words) | Sequence length test |

### Metric definitions

**`unk_rate`** — Fraction of token positions that are `<unk>`. Any non-zero value means the tokenizer cannot represent some input text. Target: < 0.001 (less than 0.1%). Achieved: **0.0** (zero unknown tokens across all sentences).

**`chars_per_token`** — Average number of characters per token. Higher is better — it means the tokenizer groups characters into longer, more meaningful pieces. Target: > 3.0. Achieved: **4.788**.

**`tokens_per_word`** — Average number of tokens per whitespace-delimited word. Lower is better — close to 1.0 means most words map to single tokens. Target: < 2.5. Achieved: **1.128**.

**`roundtrip_success_rate`** — Fraction of sentences where `decode(encode(text)) == text` (after stripping). Target: > 0.99. Achieved: **1.000** (20/20 sentences round-trip perfectly).

**`devanagari_char_coverage`** — Fraction of Devanagari characters in the validation set that map to non-`<unk>` tokens. Target: > 0.995. Achieved: **1.000** (every Devanagari character in the validation set is covered).

**`special_token_split_failures`** — Number of special tokens (e.g. `<|system|>`) that get split into multiple pieces during encoding. This would cause them to lose their special meaning. Target: 0. Achieved: **0**.

### Thresholds rationale

The `chars_per_token > 3.0` threshold comes from the observation that a Hindi character is 3 bytes in UTF-8, so a token must represent at least one full Hindi character on average to be useful. The `tokens_per_word < 2.5` threshold is a practical upper bound — beyond 2.5 fragments per word, the model would struggle to learn word-level semantics from token embeddings.

---

## 11. Test Suite (207 Automated Tests)

All 207 tests pass as of 2026-05-18. Coverage: 94%. Run with:

```bash
cd tokenizer_training/
pytest tests/ -v --cov=src/hindi_tokenizer --cov-report=term-missing
```

### Unit tests (19 test files)

| Test file | Module under test | What is tested |
|---|---|---|
| `test_settings.py` | `config/settings.py` | YAML loading, pydantic validation, env var overrides, missing fields |
| `test_records.py` | `corpus/records.py` | Pydantic corpus record schema validation |
| `test_corpus_sampler.py` | `corpus/corpus_sampler.py` | Parquet reading, Devanagari ratio filter, size-based sampling |
| `test_parquet_reader.py` | `corpus/parquet_reader.py` | Parquet shard discovery, column reading, filter application |
| `test_experiment_runner.py` | `training/experiment_runner.py` | Multi-vocab-size orchestration, force_retrain flag, skip logic |
| `test_tokenizer_trainer.py` | `training/tokenizer_trainer.py` | Unigram trainer configuration, special token insertion, model serialization |
| `test_tokenizer_validator.py` | `validation/tokenizer_validator.py` | All 6 metrics, threshold pass/fail, report JSON schema, logging events |
| `test_tokenizer_comparator.py` | `validation/tokenizer_comparator.py` | Multi-variant comparison, recommended variant selection, markdown table output |
| `test_artifact_packager.py` | `packaging/artifact_packager.py` | File copying, metadata writing, report inclusion, VERSION file |
| `test_checksum_generator.py` | `packaging/checksum_generator.py` | SHA-256 correctness, JSON output format, multi-file coverage |
| `test_tokenizer_publisher.py` | `publishing/tokenizer_publisher.py` | HF Hub API calls (mocked), repo creation, upload, env var token loading |
| `test_run_logger.py` | `observability/run_logger.py` | CSV append, run_id propagation, field types, atomic flush |
| `test_file_registry.py` | `observability/file_registry.py` | CSV append, SHA-256 registration, size_bytes capture |
| `test_sdk_loader.py` | `sdk/loader.py` | Tokenizer file discovery, from_file loading, error on missing path |
| `test_sdk_encode.py` | `sdk/encode.py` | Encoding output type, batch encoding, special token handling |
| `test_sdk_decode.py` | `sdk/decode.py` | Decoding ID sequences, skip_special_tokens, round-trip |
| `test_run_tokenizer.py` | `orchestration/run_tokenizer.py` | CLI flag parsing, step routing, dry_run behavior, smoke_test profile |
| `test_settings.py` (extended) | — | Additional edge cases: empty additional_special_tokens, yaml with missing optional fields |

### Integration tests (3 test files)

| Test file | Scope | What is tested |
|---|---|---|
| `test_corpus_pipeline.py` | Sampler → file output | Full sample → write → verify character count and Devanagari ratio on in-memory fixture corpus |
| `test_training_pipeline.py` | Trainer → validator | Train a tiny tokenizer on 5-sentence fixture corpus, validate metrics, assert report JSON is parseable |
| `test_full_pipeline.py` | End-to-end (all steps) | sample → train → validate → compare → package on tiny fixture; asserts artifact directory structure, checksums.json present and valid, VERSION file content |

### What is NOT tested

- Actual 5 GB corpus training (too slow for CI; covered by the production run on 2026-05-18)
- HuggingFace Hub network calls (mocked in `test_tokenizer_publisher.py`)
- GPU training (Unigram training is CPU-only; GPU irrelevant for tokenizer training)

---

## 12. Variant Comparison (24k vs 32k vs 48k)

Three vocab size variants were trained on the same 5 GB corpus:

| Variant | Vocab | unk_rate | chars/token | tokens/word | Roundtrip | Devanagari | All pass | Selected |
|---|---|---|---|---|---|---|---|---|
| `hindi_unigram_24k_v001` | 24,000 | 0.000 | 4.660 | 1.159 | 1.000 | 1.000 | Yes | |
| **`hindi_unigram_32k_v001`** | **32,000** | **0.000** | **4.788** | **1.128** | **1.000** | **1.000** | **Yes** | **✓** |
| `hindi_unigram_48k_v001` | 48,000 | 0.000 | 4.904 | 1.102 | 1.000 | 1.000 | Yes | |

All three variants scored perfectly on `unk_rate`, `roundtrip_success_rate`, and `devanagari_char_coverage`. The variants differ in the `chars/token` and `tokens/word` trade-off: larger vocabulary → longer tokens (fewer splits per word) → larger embedding matrix.

**Why 32k was selected:**
- All variants had `unk_rate = 0.0` so no tiebreaker was needed on that metric.
- 32k is the most widely used vocabulary size for SLMs and LLMs (LLaMA 1/2, Mistral 7B, and most Indic models use 32k).
- 32k provides a good balance: longer tokens than 24k (4.788 vs 4.660 chars/token) without the larger embedding cost of 48k.
- The 48k embedding matrix costs 33% more memory than 32k for no meaningful quality gain on this corpus.

Training times (all on CPU, 5 GB corpus):
- 24k variant: ~35 minutes
- 32k variant: ~41 minutes  
- 48k variant: ~48 minutes

---

## 13. Artifact Directory

Location: `tokenizer_training/data/final/hindi_slm_tokenizer_v001/`

```
tokenizer_training/data/final/hindi_slm_tokenizer_v001/
├── tokenizer.json                    2.21 MB  — full tokenizer definition (vocab + scores + model config)
├── tokenizer_config.json              337 B   — HuggingFace AutoTokenizer metadata
├── special_tokens_map.json            213 B   — maps token roles to surface strings
├── tokenizer_metadata.json            316 B   — algorithm/version/special_token summary
├── tokenizer_validation_report.json   728 B   — validation metrics for 32k variant
├── tokenizer_comparison_report.md     537 B   — side-by-side 24k/32k/48k comparison table
├── tokenizer_training_config.yaml    1.3 KB   — full YAML config used for training
├── checksums.json                     867 B   — SHA-256 for every file in this dir
├── VERSION                             24 B   — contains "hindi_slm_tokenizer_v001"
└── README.md                          283 B   — one-line repo description
```

### File descriptions

**`tokenizer.json`** — The only file you need to load the tokenizer. Contains the complete model: normalizer config, pre-tokenizer config, Unigram model (full vocab with scores), decoder config, and added special tokens. This file is the tokenizer. All other files are metadata.

**`tokenizer_config.json`** — HuggingFace `transformers` metadata. Allows `AutoTokenizer.from_pretrained(directory)` to find the correct tokenizer class and configuration. Sets `model_max_length`, `bos_token`, `eos_token`, `pad_token`, `unk_token`.

**`special_tokens_map.json`** — Explicitly maps the role names (`bos_token`, `eos_token`, etc.) to their string values. Used by HuggingFace pipelines and `transformers.PreTrainedTokenizer` wrappers.

**`tokenizer_metadata.json`** — Human-readable summary of algorithm, vocab size, normalizer, pre-tokenizer, corpus version, and special tokens list. Not consumed by any code — for documentation purposes.

**`tokenizer_validation_report.json`** — Machine-readable validation results for the 32k variant. All 6 metrics with their values, thresholds, pass/fail status, and special token integrity details.

**`tokenizer_comparison_report.md`** — Markdown table comparing all 3 trained variants. Produced by `TokenizerComparator`. Includes the recommended variant selection and its justification.

**`tokenizer_training_config.yaml`** — Exact copy of the YAML configuration used for training. Preserves reproducibility — you can re-run training with identical settings.

**`checksums.json`** — SHA-256 checksums for all files in this directory. The `tokenizer.json` checksum is the canonical integrity reference: `fbe21c642a4a13030833be48733c1c6b78244e4c0bc077516422b22e7f046cd9`.

**`VERSION`** — Plain text file containing `hindi_slm_tokenizer_v001`. Consumed by the artifact packager and publish pipeline for version tagging.

---

## 14. What Is on HuggingFace Hub

**Repo:** `vaibhavmaurya/hindi-slm-tokenizer-v001` (private)  
**Published:** 2026-05-18 08:08 UTC  
**Last commit SHA:** verified at publish time

The following 11 files are on the Hub (`.gitattributes` is auto-generated by HuggingFace):

| File on Hub | Local source | Purpose |
|---|---|---|
| `.gitattributes` | Auto-generated | HuggingFace LFS configuration |
| `tokenizer.json` | `data/final/hindi_slm_tokenizer_v001/tokenizer.json` | **Primary artifact** — load this |
| `tokenizer_config.json` | `data/final/hindi_slm_tokenizer_v001/tokenizer_config.json` | AutoTokenizer support |
| `special_tokens_map.json` | `data/final/hindi_slm_tokenizer_v001/special_tokens_map.json` | Special token mapping |
| `tokenizer_metadata.json` | `data/final/hindi_slm_tokenizer_v001/tokenizer_metadata.json` | Algorithm summary |
| `tokenizer_validation_report.json` | `data/final/hindi_slm_tokenizer_v001/tokenizer_validation_report.json` | Validation evidence |
| `tokenizer_comparison_report.md` | `data/final/hindi_slm_tokenizer_v001/tokenizer_comparison_report.md` | Variant comparison |
| `tokenizer_training_config.yaml` | `data/final/hindi_slm_tokenizer_v001/tokenizer_training_config.yaml` | Reproducibility |
| `checksums.json` | `data/final/hindi_slm_tokenizer_v001/checksums.json` | Integrity verification |
| `VERSION` | `data/final/hindi_slm_tokenizer_v001/VERSION` | Version identifier |
| `README.md` | `data/final/hindi_slm_tokenizer_v001/README.md` | Repo description |

**What is NOT on the Hub (intentionally):**
- Raw training corpus (5 GB — too large, and Sangraha is already public on HuggingFace)
- Intermediate training artifacts (24k, 48k variants) — only the selected 32k variant is published
- Test fixtures or development tooling
- `tokenizer_training/data/` data directory contents (corpus samples, intermediate artifacts)

**To load from Hub:**
```python
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="vaibhavmaurya/hindi-slm-tokenizer-v001",
    filename="tokenizer.json",
    token="hf_your_token_here",  # required while repo is private
)
tokenizer = Tokenizer.from_file(path)
```

Or with `transformers`:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "vaibhavmaurya/hindi-slm-tokenizer-v001",
    token="hf_your_token_here",
)
```

**To re-publish** (if you need to update any file):
```bash
cd tokenizer_training/
# set HF_TOKEN in environment or .env file
python -m hindi_tokenizer.orchestration.run_tokenizer --step publish
```

---

## 15. How to Load and Use

### Option A: Direct (HuggingFace `tokenizers` library — recommended)

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer_training/data/final/hindi_slm_tokenizer_v001/tokenizer.json"
)

# Encode a single sentence
encoding = tokenizer.encode("हिंदी भाषा में प्रशिक्षण")
print(encoding.ids)     # [1423, 892, 45, 3201, ...]
print(encoding.tokens)  # ['▁हिंदी', '▁भाषा', '▁में', '▁प्रशिक्षण']

# Decode back to text
text = tokenizer.decode(encoding.ids)
print(text)  # "हिंदी भाषा में प्रशिक्षण"

# Batch encode
batch = tokenizer.encode_batch(["वाक्य एक", "वाक्य दो"])

# Get vocabulary
vocab = tokenizer.get_vocab()         # dict[str, int]
vocab_size = tokenizer.get_vocab_size()  # 32000
```

### Option B: SDK wrapper (in this repo)

```python
from hindi_tokenizer.sdk.loader import load_tokenizer
from hindi_tokenizer.sdk.encode import encode, encode_batch
from hindi_tokenizer.sdk.decode import decode

tok = load_tokenizer("tokenizer_training/data/final/hindi_slm_tokenizer_v001")

# Single encode
ids = encode(tok, "नमस्ते")

# Batch encode
id_batches = encode_batch(tok, ["नमस्ते", "हिंदी"])

# Decode
text = decode(tok, ids)
```

### Option C: HuggingFace `transformers` (for pipeline integration)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "tokenizer_training/data/final/hindi_slm_tokenizer_v001"
)

# Returns PyTorch tensors directly
inputs = tok("हिंदी में नमस्ते", return_tensors="pt")
print(inputs["input_ids"])   # tensor([[2, 1423, 45, 892, 3]])
#                                        ^BOS                ^EOS
```

### Token IDs for special usage

```python
tokenizer = Tokenizer.from_file("...")
vocab = tokenizer.get_vocab()

pad_id = vocab["<pad>"]      # 0
unk_id = vocab["<unk>"]      # 1
bos_id = vocab["<s>"]        # 2
eos_id = vocab["</s>"]       # 3
sys_id = vocab["<|system|>"] # 4
usr_id = vocab["<|user|>"]   # 5
ast_id = vocab["<|assistant|>"] # 6
end_id = vocab["<|end|>"]    # 7
```

---

## 16. Embedding Matrix Size for Pretraining

When building the SLM model:

```python
import torch.nn as nn

vocab_size = 32000  # NEVER derive from len(tokenizer) at runtime — use this constant
d_model = 512       # your model dimension

# Input embeddings
embed = nn.Embedding(vocab_size, d_model, padding_idx=0)  # 0 = <pad>

# Output projection (language model head) — shares weights with input embeddings (optional)
lm_head = nn.Linear(d_model, vocab_size, bias=False)
# Weight tying: lm_head.weight = embed.weight
```

Memory for embedding matrix at various model sizes (BF16 training):

| Model d_model | Embedding only | Both embed + lm_head (tied weights) | Both (untied) |
|---|---|---|---|
| 256 | 15.6 MB | 15.6 MB | 31.2 MB |
| 512 | 31.2 MB | 31.2 MB | 62.5 MB |
| 1024 | 62.5 MB | 62.5 MB | 125.0 MB |
| 2048 | 125.0 MB | 125.0 MB | 250.0 MB |

**Recommendation:** Use tied weights (input embedding = output projection transposed). This halves the parameter count for the embedding component and is standard practice for small models.

---

## 17. Freezing Rules

Once SLM pretraining begins, the tokenizer is **permanently frozen**. The following are prohibited:

- **Modifying `tokenizer.json`** — any change invalidates all pretrained embeddings.
- **Adding or removing vocabulary entries** — changes all token IDs ≥ the insertion point.
- **Renaming special tokens** — the model has learned embeddings keyed by ID, not name; renaming causes semantic corruption.
- **Changing token IDs** — `<pad>=0` is used as `padding_idx` in `nn.Embedding`; changing it breaks all padding logic.
- **Changing normalization (NFKC)** — the same input text must always produce the same token IDs.
- **Changing the pre-tokenizer (Metaspace)** — word boundary information is encoded in `▁`; changing this changes all token boundaries.
- **Retraining the tokenizer** — a retrained tokenizer has different vocabulary entries and different scores, even on the same corpus.

The SHA-256 checksums in `checksums.json` are the source of truth. Any `tokenizer.json` that does not produce the checksum `fbe21c642a4a13030833be48733c1c6b78244e4c0bc077516422b22e7f046cd9` must be rejected.

---

## 18. Integrity Verification

**Quick check (Python):**
```python
import hashlib
from pathlib import Path

expected = "fbe21c642a4a13030833be48733c1c6b78244e4c0bc077516422b22e7f046cd9"
path = Path("tokenizer_training/data/final/hindi_slm_tokenizer_v001/tokenizer.json")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
assert actual == expected, f"tokenizer.json is corrupted or has been modified"
print("OK")
```

**Full directory verification:**
```python
import hashlib, json
from pathlib import Path

artifact_dir = Path("tokenizer_training/data/final/hindi_slm_tokenizer_v001")
checksums = json.loads((artifact_dir / "checksums.json").read_text())
for name, expected in checksums.items():
    actual = hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
    status = "OK" if actual == expected else "MISMATCH"
    print(f"{status}  {name}")
```

**All SHA-256 checksums:**

| File | SHA-256 |
|---|---|
| `tokenizer.json` | `fbe21c642a4a13030833be48733c1c6b78244e4c0bc077516422b22e7f046cd9` |
| `tokenizer_config.json` | `fd3dcca89c0ca509b16b27bd1152fc0df1509634ef672a7cbf70982b368e72cf` |
| `special_tokens_map.json` | `7c811104c5a2bad90d8d992de04d345d413cad6643f61956cf27247772ed2db2` |
| `tokenizer_metadata.json` | `f4fd93f01d9953fb1fac51b93a3691e59b88b11ae855c532cdaad084f3d83aaf` |
| `tokenizer_validation_report.json` | `710a1d3e441826d8c03f6d11b530a6f4440a95606c86d2d84fc5d1b3537b571f` |
| `tokenizer_training_config.yaml` | `223429af5ba40db476fd81876266136579366cae531c35d9db35bd56d9e11256` |
| `VERSION` | `1559432649e9cba76b31c919296aa2b3ec15864fa738d5e5b6679bddc6d6c601` |
| `README.md` | `28c9c7eaa72e57f7b9c2ed0373c980444a2df4df5723cb4f832ba38e21ae83bb` |
| `tokenizer_comparison_report.md` | `1ecc1d5a12eaa4b855e731aae60d66a163b7713c50089bcbded5fb026260cb17` |

---

## 19. Related Files

| File | Purpose |
|---|---|
| `tokenizer_training/CLAUDE.md` | Development context for Claude Code — workstream overview, commands |
| `tokenizer_training/DevelopmentPlan.md` | Full architecture, design decisions, component catalogue |
| `tokenizer_training/configs/tokenizer_training_config.yaml` | Training configuration (also copied into artifact dir) |
| `tokenizer_training/data/reports/pipeline_run_log.csv` | Audit trail: every pipeline step, timing, record counts |
| `tokenizer_training/data/reports/data_file_registry.csv` | Every file produced/consumed, with SHA-256 and size |
| `tokenizer_training/data/reports/tokenizer_comparison_report.md` | 24k/32k/48k comparison (also in artifact dir) |
| `data_ingestion/CORPUS_HANDOFF.md` | Corpus specification: how the 5 GB training data was produced |
| `tokenizer_training/tests/fixtures/validation_sentences.txt` | The 20 Hindi sentences used for validation |

---

## 20. Reference Links

### Core algorithm papers

- **Unigram LM tokenizer:** Taku Kudo (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL 2018. [arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959)
- **SentencePiece system:** Taku Kudo & John Richardson (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* EMNLP 2018. [arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
- **BPE (for comparison):** Rico Sennrich, Barry Haddow, Alexandra Birch (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

### Library documentation

- **HuggingFace `tokenizers` library (Rust-backed):** [huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers)
- **HuggingFace `tokenizers` — Unigram model:** [huggingface.co/docs/tokenizers/components#models](https://huggingface.co/docs/tokenizers/components#models)
- **HuggingFace `transformers` — AutoTokenizer:** [huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer)
- **HuggingFace `tokenizers` — Quicktour:** [huggingface.co/docs/tokenizers/quicktour](https://huggingface.co/docs/tokenizers/quicktour)

### Unicode/normalization

- **Unicode Normalization Forms (NFKC):** [unicode.org/reports/tr15](https://unicode.org/reports/tr15/)
- **Devanagari Unicode block (U+0900–U+097F):** [unicode.org/charts/PDF/U0900.pdf](https://unicode.org/charts/PDF/U0900.pdf)

### Training corpus

- **AI4Bharat Sangraha:** [huggingface.co/datasets/ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
- **Sangraha paper:** Sangraha: Verified, Clean, and Deduplicated Multilingual Dataset from the Web for Indic Languages (2024). [arxiv.org/abs/2309.11509](https://arxiv.org/abs/2309.11509)

### Tokenizer evaluation methodology

- **Fertility and other metrics:** Rust et al. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021. [arxiv.org/abs/2012.15613](https://arxiv.org/abs/2012.15613)

### Published tokenizer

- **HuggingFace Hub repo:** `vaibhavmaurya/hindi-slm-tokenizer-v001` (private — requires token)
