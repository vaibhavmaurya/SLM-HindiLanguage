# Hindi SLM — Training Guide

## Overview

This guide covers the end-to-end training pipeline for the Hindi Small Language Model (SLM): a
46.54M-parameter decoder-only transformer trained on Hindi text from `ai4bharat/sangraha`.

The pipeline runs in two modes:

| Mode | Data | Steps | Purpose |
|------|------|-------|---------|
| **TEST** | 10 MB synthetic | 100 | Smoke-test the full pipeline end-to-end |
| **FULL** | ~1 GB real Hindi | 50,000 | Actual training run |

---

## Quick Start

```powershell
# TEST mode (smoke test — completes in ~15 seconds)
cd slm_training
python run_test_pipeline.py

# Monitor training live
& "$env:APPDATA\Python\Python312\Scripts\tensorboard.exe" --logdir logs/tensorboard --port 6006
# then open http://localhost:6006
```

---

## Model Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 512 | Embedding and residual stream dimension |
| `num_layers` | 10 | Transformer decoder blocks |
| `num_attention_heads` | 8 | Query heads |
| `num_kv_heads` | 2 | Key/Value heads — 4× smaller KV cache (GQA) |
| `head_dim` | 64 | Per-head dimension |
| `intermediate_size` | 1536 | SwiGLU MLP hidden dimension |
| `max_seq_len` | 512 | Context window (tokens) |
| `vocab_size` | 32,000 | Hindi Unigram tokenizer |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `rope_base` | 10,000 | RoPE frequency base |
| `tie_embeddings` | True | lm_head shares embed_tokens weights |
| `dropout` | 0.1 | Applied at embed, residual, and attention |

**Total parameters: 46,541,312 (46.54M)**

Parameter breakdown:

| Component | Count | M params |
|-----------|-------|----------|
| Embedding | 16,384,000 | 16.38 |
| All layers (10×) | 30,156,800 | 30.16 |
| — Attention per layer | 655,360 | 0.66 |
| — MLP per layer | 2,359,296 | 2.36 |
| Final RMSNorm | 512 | ~0 |
| lm_head | 0 | 0 (tied) |

### Architecture Details

- **Attention**: Grouped Query Attention (GQA) — 8 query heads, 2 KV heads per layer
- **MLP**: SwiGLU — `gate × up → SiLU → down` (no bias)
- **Normalization**: RMSNorm pre-norm before every attention and MLP block
- **Position encoding**: Rotary Positional Embeddings (RoPE), precomputed cos/sin cache
- **Precision**: bfloat16 (native on Ada Lovelace; no GradScaler needed)

---

## Training Hyperparameters

All hyperparameters are saved to `configs/training_config_test.yaml` (TEST) or
`configs/training_config.yaml` (FULL) at the start of every run, and embedded inside every
checkpoint's `meta.json`.

### TEST Mode (current run)

| Hyperparameter | Value |
|----------------|-------|
| `learning_rate` | 3e-4 |
| `warmup_steps` | 10 |
| `max_steps` | 100 |
| `per_device_batch_size` | 2 |
| `gradient_accumulation_steps` | 1 |
| **Effective batch size** | **2 sequences × 512 tokens = 1,024 tokens/step** |
| `weight_decay` | 0.1 |
| `max_grad_norm` | 1.0 (gradient clipping) |
| `dtype` | bfloat16 |
| `log_every` | 5 steps |
| `eval_every` | 25 steps |
| `save_every` | 50 steps |

### FULL Mode (production run)

| Hyperparameter | Value |
|----------------|-------|
| `learning_rate` | 3e-4 |
| `warmup_steps` | 1,000 |
| `max_steps` | 50,000 |
| `per_device_batch_size` | 2 |
| `gradient_accumulation_steps` | 16 |
| **Effective batch size** | **32 sequences × 512 tokens = 16,384 tokens/step** |
| `weight_decay` | 0.1 |
| `max_grad_norm` | 1.0 |
| `dtype` | bfloat16 |
| `log_every` | 50 steps |
| `eval_every` | 500 steps |
| `save_every` | 1,000 steps |

### Optimizer

**AdamW** with decoupled weight decay:
- β₁ = 0.9, β₂ = 0.95, ε = 1e-8
- Weight decay (0.1) applied only to 2D+ parameters (weight matrices)
- No weight decay on biases, norms, or 1D parameters

### Learning Rate Schedule

**Warmup → Cosine Decay**:

```
step < warmup_steps  →  lr = base_lr × (step / warmup_steps)
step ≥ warmup_steps  →  lr = base_lr × 0.5 × (1 + cos(π × progress))
```

where `progress = (step - warmup) / (max_steps - warmup)`.

---

## Configuration Persistence

Every checkpoint saves a complete snapshot of all hyperparameters. Nothing is lost between
runs.

**Checkpoint directory layout:**
```
artifacts/checkpoints_test/
└── step_0000100/
    ├── model.pt          ← model weights (state_dict)
    ├── optimizer.pt      ← AdamW state (momentum, variance per param)
    └── meta.json         ← full model_config + train_config + step + loss
```

`meta.json` example:
```json
{
  "step": 100,
  "loss": 0.0,
  "model_config": { "hidden_size": 512, "num_layers": 10, ... },
  "train_config": { "learning_rate": 0.0003, "warmup_steps": 10, ... }
}
```

Training **automatically resumes** from the latest checkpoint on restart — no flags needed.

---

## Evaluation Metrics

### Primary Metric: Perplexity

**Perplexity = exp(average cross-entropy loss)** measured on held-out sequences.

```
PPL = exp( -1/N × Σ log P(token_i | context) )
```

Lower is better. A random model over 32,000 tokens has PPL = 32,000. A well-trained Hindi
LM should reach PPL < 50 after the full run.

| Split | TEST run (100 steps) | Expected (full run) |
|-------|---------------------|---------------------|
| Validation | 329.27 | < 50 |
| Test | 347.81 | < 60 |

> **Note**: The TEST run perplexity is high because only 100 steps were run on 29 training
> sequences — the model has not had enough exposure to learn Hindi structure.

### Secondary Metrics

| Metric | What it measures | TEST result |
|--------|-----------------|-------------|
| **Devanagari ratio** | Fraction of generated characters in the Devanagari Unicode block (U+0900–U+097F) | avg 0.697 |
| **UNK token rate** | Count of `<unk>` tokens in generated output — measures tokenizer coverage | 0.0 (perfect) |
| **Cross-entropy loss** | Raw training signal logged every N steps | Final: 28.42 |
| **Validation loss** | Loss on val split — monitors overfitting | logged every `eval_every` |
| **Gradient norm** | L2 norm of gradients before clipping — monitors training stability | clipped at 1.0 |

### Generation Quality Check

5 Hindi prompts are decoded at evaluation time using nucleus sampling (temperature=0.8, top_p=0.9):

| Prompt type | Prompt |
|-------------|--------|
| News | *"आज का समाचार यह है कि"* |
| Story | *"एक बार की बात है, जंगल में"* |
| Conversational | *"नमस्ते! आप कैसे हैं? मैं"* |
| Factual | *"भारत की राजधानी नई दिल्ली है। यहाँ"* |
| Poetry | *"चाँद की रोशनी में, नदी किनारे"* |

The evaluation report is saved to `artifacts/reports/evaluation_report.md` after every run.

---

## TensorBoard Monitoring

TensorBoard is active during every training run. Logs are written to `logs/tensorboard/`.

**Launch:**
```powershell
& "$env:APPDATA\Python\Python312\Scripts\tensorboard.exe" --logdir logs/tensorboard --port 6006
```
Open **http://localhost:6006** in any browser. Charts update live during training.

**Available charts:**

| Chart | Logged every | What to watch for |
|-------|-------------|-------------------|
| `train/loss` | `log_every` steps | Should decrease steadily |
| `train/lr` | `log_every` steps | Warmup ramp then cosine decay |
| `train/tokens_per_sec` | `log_every` steps | Training throughput |
| `train/vram_gb` | `log_every` steps | Should stay < 8 GB |
| `val/loss` | `eval_every` steps | Should track train/loss without diverging |
| `val/perplexity` | `eval_every` steps | Primary quality signal |

---

## OOM Recovery

If CUDA runs out of memory, the trainer automatically applies these strategies in order:

| Retry | Action |
|-------|--------|
| 1 | batch_size ÷ 2 |
| 2 | Enable gradient checkpointing (trades compute for memory) |
| 3 | num_workers → 2 |
| 4 | seq_len → 256 (halved) |
| 5 | Tier down: hidden=384, layers=8 (~28M params) |

---

## Data Pipeline

```
ai4bharat/sangraha (Hindi/verified)
    ↓ sentinel check (skip if already downloaded)
data/raw/sangraha_hin_*/
    ↓ NFKC normalize → length filter (50–50,000 chars)
    ↓ Devanagari ratio filter (≥ 60%)
    ↓ tokenize (Hindi SLM tokenizer v001, 32k Unigram)
    ↓ append EOS, concatenate, pack into 512-token windows
data/tokenized/{train, val, test}/   (98 / 1 / 1 split, seed=42)
```

Manifest saved to `configs/tokenized_dataset_manifest*.json` — includes split sizes, total
token count, and SHA-256 of the first training shard for reproducibility.

---

## File Map

```
slm_training/
├── run_test_pipeline.py          ← entry point (TEST mode)
├── src/slm_training/
│   ├── architecture.py           ← NAS calculator, param/memory estimators
│   ├── dataset.py                ← download, clean, tokenize, pack, split
│   ├── model.py                  ← HindiSLM, GQA, SwiGLU, RoPE, RMSNorm
│   ├── trainer.py                ← training loop, optimizer, LR schedule, OOM recovery
│   └── evaluator.py              ← perplexity, generation, Devanagari ratio
├── configs/
│   ├── optimal_model_schema.yaml ← model architecture (written at run time)
│   ├── training_config_test.yaml ← hyperparameters (written at run time)
│   ├── tokenizer_config.yaml     ← tokenizer metadata
│   └── tokenized_dataset_manifest_test.json
├── artifacts/
│   ├── checkpoints_test/step_N/  ← model.pt + optimizer.pt + meta.json
│   └── reports/evaluation_report.md
└── logs/tensorboard/             ← TensorBoard event files
```
