# Hindi SLM — Architecture & Training Reference

Complete reference document covering the model architecture, training configuration,
data pipeline, and all runtime details for the Hindi Small Language Model.

---

## Table of Contents

1. [What Is Being Built](#1-what-is-being-built)
2. [Model Architecture](#2-model-architecture)
3. [Parameter Breakdown](#3-parameter-breakdown)
4. [Memory & VRAM](#4-memory--vram)
5. [Training Configuration](#5-training-configuration)
6. [Steps vs Epochs — Explained](#6-steps-vs-epochs--explained)
7. [Learning Rate Schedule](#7-learning-rate-schedule)
8. [Data Pipeline](#8-data-pipeline)
9. [Training Signals — What to Watch](#9-training-signals--what-to-watch)
10. [Checkpointing & Resume](#10-checkpointing--resume)
11. [OOM Recovery](#11-oom-recovery)
12. [Model Tiers](#12-model-tiers)
13. [Post-Training Pipeline](#13-post-training-pipeline)
14. [File Map](#14-file-map)

---

## 1. What Is Being Built

A **decoder-only transformer** trained from scratch on Hindi text.

| Property | Value |
|---|---|
| Task | Next-token prediction (causal language modelling) |
| Language | Hindi (Devanagari script) |
| Training data | `ai4bharat/sangraha` Hindi verified subset |
| Parameters | **46.54M** (SMALL tier) |
| Context window | 512 tokens |
| Tokenizer | Hindi Unigram SentencePiece, 32,000 vocabulary |
| Target deployment | Raspberry Pi 8 GB (via GGUF Q4_K_M quantization) |
| Training hardware | NVIDIA RTX 3000 Ada Laptop GPU, 8.6 GB VRAM |

---

## 2. Model Architecture

The model is a **decoder-only transformer** — the same family as GPT-2, LLaMA, and Mistral.
"Decoder-only" means it uses a causal (left-to-right) attention mask: each token can only
attend to tokens before it, not after. This is standard for language generation.

### High-Level Forward Pass

```
input_ids [B, T]
    → Token Embedding  [B, T, 512]
    → Dropout (0.1)
    → TransformerBlock × 10
        each block:
          RMSNorm → GroupedQueryAttention (+RoPE) → Residual
          RMSNorm → SwiGLUMLP → Residual
    → Final RMSNorm
    → LM Head (tied to embedding)  [B, T, 32000]
    → logits
```

Where B = batch size, T = sequence length (up to 512).

---

### 2.1 RMSNorm (Root Mean Square Normalization)

Used instead of LayerNorm. Simpler and faster — no mean subtraction.

```
RMSNorm(x) = x / sqrt(mean(x²) + ε)  ×  γ
```

- `ε = 1e-6` (numerical stability)
- `γ` = learnable scale parameter, shape `[hidden_size]`, initialised to 1
- Applied **before** every attention block and every MLP block (Pre-Norm)
- Pre-Norm (normalise before the sub-layer) is more stable than Post-Norm for deep networks

**Why not LayerNorm?**  
LayerNorm computes mean and variance; RMSNorm skips the mean subtraction. Empirically
equivalent quality, lower compute — used in LLaMA, Mistral, Gemma.

---

### 2.2 Rotary Positional Embeddings (RoPE)

Tokens have no innate sense of position. RoPE injects position by rotating the query and
key vectors in attention — the rotation angle depends on the token's position in the sequence.

```
inv_freq[i] = 1 / (10000 ^ (2i / head_dim))   for i = 0..31

For position t:
  freqs = t × inv_freq
  [cos(freqs), cos(freqs)] and [sin(freqs), sin(freqs)]  →  applied to Q and K

q_rotated = q × cos + rotate_half(q) × sin
k_rotated = k × cos + rotate_half(k) × sin
```

- `head_dim = 64`, so `inv_freq` has 32 entries
- `rope_base = 10000` (standard; higher base → longer effective range)
- Cos/sin tables are **precomputed** up to `max_seq_len = 512` and cached as buffers
- RoPE is applied after Q and K projections, before attention scores

**Why RoPE over learned embeddings?**  
Learned position embeddings don't generalise beyond trained length. RoPE encodes relative
position algebraically — token A at position 5 and token B at position 8 have a fixed
geometric relationship regardless of where they appear.

---

### 2.3 Grouped Query Attention (GQA)

Standard multi-head attention has one K and V head per Q head.
GQA shares K/V heads across multiple Q heads — reducing KV cache size at inference.

```
Config:
  num_attention_heads (Q heads)  = 8
  num_kv_heads (K/V heads)       = 2
  head_dim                       = 64
  Groups: 8 / 2 = 4  →  each KV head is shared by 4 Q heads
```

**Projections per layer:**
| Projection | Shape | Parameters |
|---|---|---|
| Q (`q_proj`) | `512 → 8×64 = 512` | 512 × 512 = 262,144 |
| K (`k_proj`) | `512 → 2×64 = 128` | 512 × 128 = 65,536 |
| V (`v_proj`) | `512 → 2×64 = 128` | 512 × 128 = 65,536 |
| O (`o_proj`) | `8×64=512 → 512` | 512 × 512 = 262,144 |
| **Total** | | **655,360 / layer** |

No bias terms on any projection (standard modern practice).

**Attention computation:**
```
scores = softmax( (Q × Kᵀ) / √64 )   [causal mask applied]
output = scores × V
```

`√64 = 8` is the scaling factor — prevents dot products from growing too large with
high-dimensional vectors, which would push softmax into near-zero gradients.

Causal mask: a lower-triangular boolean mask sets future positions to `-inf` before softmax,
so the model cannot see tokens it hasn't generated yet.

Uses `F.scaled_dot_product_attention` — PyTorch's built-in implementation that automatically
uses **Flash Attention** when available (reduces memory from O(T²) to O(T)).

**KV expansion:** Before attention, K and V are expanded to match Q:
```
K: [B, 2, T, 64] → repeat_interleave(4) → [B, 8, T, 64]
V: [B, 2, T, 64] → repeat_interleave(4) → [B, 8, T, 64]
```

**Why GQA?**  
At inference, the KV cache grows with sequence length. With MHA (8 KV heads), the cache
is 4× larger than with GQA (2 KV heads). On Raspberry Pi with limited RAM, this matters:
- MHA KV cache at 512 ctx: ~10 MB
- GQA KV cache at 512 ctx: **2.5 MB** (4× smaller)

---

### 2.4 SwiGLU MLP

Each transformer block has a feed-forward network using the SwiGLU activation.

```
SwiGLU(x) = down_proj( SiLU(gate_proj(x))  ×  up_proj(x) )
```

Three linear projections per MLP block:
| Projection | Shape | Parameters |
|---|---|---|
| `gate_proj` | `512 → 1536` | 512 × 1536 = 786,432 |
| `up_proj` | `512 → 1536` | 512 × 1536 = 786,432 |
| `down_proj` | `1536 → 512` | 1536 × 512 = 786,432 |
| **Total** | | **2,359,296 / layer** |

No bias terms.

**SiLU (Sigmoid Linear Unit):**
```
SiLU(x) = x × sigmoid(x) = x / (1 + e^(-x))
```
Smooth, non-monotonic activation. Empirically outperforms ReLU and GeLU for transformers.

**Why SwiGLU over standard FFN?**  
Standard FFN: `down(activation(up(x)))` — one projection up, activation, one down.
SwiGLU gates the up-projection with the gate-projection — a multiplicative interaction.
Used in LLaMA, PaLM, Gemma. Consistently better quality for same parameter budget.

---

### 2.5 TransformerBlock

```
def forward(x):
    x = x + dropout(attn(rms_norm(x)))   # attention sub-layer
    x = x + dropout(mlp(rms_norm(x)))    # MLP sub-layer
    return x
```

- Pre-Norm: normalise input before passing to sub-layer
- Residual connections: add the original `x` back after each sub-layer
- Dropout (0.1): applied to the output of attention and MLP before residual addition

---

### 2.6 HindiSLM — Full Model

```
embed_tokens:  Embedding(32000, 512)   ← token ID → 512-dim vector
embed_drop:    Dropout(0.1)
layers[0..9]:  TransformerBlock × 10
norm:          RMSNorm(512)            ← final normalisation
lm_head:       Linear(512, 32000, bias=False)  ← tied to embed_tokens
```

**Tied embeddings:** `lm_head.weight = embed_tokens.weight`

The input embedding matrix and the output projection matrix are the same tensor.
This saves 16.38M parameters that would otherwise be a second copy of the vocabulary matrix.
The model learns one shared representation: "what does token X mean as input" = "how likely
is token X as output". This is standard for smaller LMs.

**Weight initialisation:**
- All `nn.Linear` weights: `N(0, 0.02)` (normal distribution, std=0.02)
- All `nn.Embedding` weights: `N(0, 0.02)`
- Biases: zeros (but no bias terms exist except none)

---

### 2.7 Loss Function

**Cross-entropy loss with causal shift:**

```python
# Predict token at position i+1 given tokens 0..i
shift_logits = logits[:, :-1, :]   # all positions except last
shift_labels = labels[:, 1:]       # all tokens except first

loss = cross_entropy(shift_logits, shift_labels)
```

For a sequence `[A, B, C, D]`:
- Input to model: `[A, B, C, D]`
- Model must predict: `[B, C, D, ?]`
- Loss measures: how well did `A` predict `B`? `B` predict `C`? `C` predict `D`?

**Initial loss:** With 32,000 tokens, random guessing gives `log(32000) ≈ 10.37` nats.
At step 364 the loss was 97.1 — this is high because:
- Warmup phase: learning rate is very low (only 364/1000 of warmup done)
- Model is initialised randomly (no pretrained weights)
- Loss will drop rapidly once LR reaches its peak at step 1000

---

## 3. Parameter Breakdown

**SMALL tier (currently training):**

| Component | Formula | Count | M params |
|---|---|---|---|
| Token embedding | 32,000 × 512 | 16,384,000 | 16.38 |
| Attention per layer | Q(262,144) + K(65,536) + V(65,536) + O(262,144) | 655,360 | 0.66 |
| MLP per layer | gate(786,432) + up(786,432) + down(786,432) | 2,359,296 | 2.36 |
| RMSNorm per layer | 2 × 512 (pre-attn + pre-mlp) | 1,024 | ~0 |
| All 10 layers | (655,360 + 2,359,296 + 1,024) × 10 | 30,156,800 | 30.16 |
| Final RMSNorm | 512 | 512 | ~0 |
| lm_head | 0 (tied to embedding) | 0 | 0 |
| **TOTAL** | | **46,541,312** | **46.54** |

---

## 4. Memory & VRAM

### Training VRAM (batch=2, bfloat16)

| Component | Formula | Size |
|---|---|---|
| Model weights | 46.54M × 2 bytes (bf16) | 93 MB |
| Gradients | 46.54M × 2 bytes (bf16) | 93 MB |
| AdamW states | 46.54M × 8 bytes (2 × fp32 moments) | 372 MB |
| Activations | 10 layers × 1024 tokens × 24,576 bytes/token | 252 MB |
| **Estimated total** | | **~810 MB** |

The VRAM log in training output (`0.19 GB after model load`) is measured **before** the
optimizer is created and before any forward pass — only weights are allocated at that point.
Actual VRAM during training is ~810 MB to ~1.2 GB.

### Inference KV Cache (Raspberry Pi deployment)

```
KV cache = 2 (K and V) × 10 layers × 512 tokens × 2 KV heads × 64 head_dim × 2 bytes
         = 2 × 10 × 512 × 2 × 64 × 2
         = 2,621,440 bytes = 2.5 MB
```

This is tiny — the 4× KV head reduction from GQA (8 Q heads → 2 KV heads) directly
reduces the KV cache 4×.

### Q4_K_M Quantized Model (for Pi)

| Component | Size |
|---|---|
| Model weights (Q4_K_M ~4.5 bits/param) | ~26 MB |
| KV cache at 512 ctx | 2.5 MB |
| Runtime overhead | ~100 MB |
| **Total on Pi** | **~130 MB** |

The model fits comfortably in Raspberry Pi 8 GB RAM.

---

## 5. Training Configuration

### Currently Running (FULL run)

| Parameter | Value | Meaning |
|---|---|---|
| `--tier` | SMALL | 46.54M param model (see Section 12 for other tiers) |
| `--max-steps` | 50,000 | Total optimizer update steps (see Section 6) |
| `--batch` | 2 | Sequences per GPU forward pass |
| `--grad-accum` | 16 | Accumulate gradients over 16 micro-steps before updating |
| **Effective batch** | **32** | 2 × 16 = 32 sequences per optimizer step |
| **Tokens per step** | **16,384** | 32 sequences × 512 tokens/sequence |
| `--lr` | 3e-4 | Peak learning rate (after warmup) |
| `--warmup` | 1,000 | Steps to linearly ramp LR from 0 to 3e-4 |
| `--weight-decay` | 0.1 | L2 regularisation on weight matrices |
| `--dtype` | bfloat16 | Mixed precision (native on Ada Lovelace) |
| `--log-every` | 50 | Print loss + LR + tokens/sec every 50 steps |
| `--eval-every` | 500 | Run validation loss every 500 steps |
| `--save-every` | 1,000 | Save checkpoint every 1,000 steps |

### Gradient Accumulation

With `--batch 2 --grad-accum 16`:

```
for micro_step in range(16):
    batch = load_next_batch()          # 2 sequences of 512 tokens
    loss = model(batch) / 16           # divide to average
    loss.backward()                    # accumulate gradients
optimizer.step()                       # one update using 16 batches combined
optimizer.zero_grad()
step += 1
```

This simulates a batch of 32 sequences without needing 32 sequences in VRAM at once.
Effective batch = 32 × 512 = 16,384 tokens per step — large enough for stable training.

### Optimizer: AdamW

```
Adam:    m = β₁m + (1-β₁)g          ← first moment (momentum)
         v = β₂v + (1-β₂)g²         ← second moment (variance)
         θ = θ - lr × m / (√v + ε)   ← parameter update

AdamW:   θ = θ × (1 - lr × λ)       ← weight decay decoupled from gradient
```

| β₁ | β₂ | ε | weight_decay |
|---|---|---|---|
| 0.9 | 0.95 | 1e-8 | 0.1 |

Weight decay (0.1) is applied **only** to 2D+ parameters (weight matrices, embedding).
Biases, norms (1D), and embedding norms are excluded from weight decay.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Before every optimizer step, the global gradient L2 norm is computed.
If it exceeds 1.0, all gradients are scaled down proportionally.
This prevents exploding gradients that can destabilise early training.

---

## 6. Steps vs Epochs — Explained

**A step** = one optimizer update (one call to `optimizer.step()`).

**An epoch** = one full pass through the entire training dataset.

### Dataset Size

```
21 source files × ~90,486 train sequences/file ≈ 1,900,000 train sequences
Each sequence = 512 tokens
Total tokens in dataset ≈ 1,900,000 × 512 ≈ 972,800,000 ≈ 973M tokens
```

### Steps Per Epoch

```
Effective batch = 32 sequences/step
Steps per epoch = 1,900,000 / 32 = ~59,375 steps
```

### What 50,000 Steps Means

```
50,000 steps × 32 sequences × 512 tokens = 819,200,000 tokens seen
50,000 / 59,375 ≈ 0.84 epochs
```

**50,000 steps = the model sees ~819M tokens ≈ 84% of one pass through the dataset.**

This is intentional — for language models, the number of tokens seen matters more
than how many "full passes" were done. 819M tokens is sufficient for a 46M parameter model
to converge to reasonable Hindi fluency. Scaling laws suggest roughly 1 token per parameter
per training token for good performance; 819M >> 46M.

### Time Estimate

```
~0.5 seconds per step on RTX 3000 Ada (SMALL tier, batch=32 effective)
50,000 steps × 0.5s = 25,000 seconds ≈ 6 hours 56 minutes
```

### Checkpoint Schedule

| Step | Action |
|---|---|
| Every 50 | Log loss, LR, tokens/sec to console and TensorBoard |
| Every 500 | Run validation loss on val split |
| Every 1,000 | Save checkpoint (50 checkpoints total over 50k steps) |
| Step 50,000 | Final checkpoint + training complete |

---

## 7. Learning Rate Schedule

**Linear warmup → Cosine decay**

```
step < 1000:    lr = 3e-4 × (step / 1000)        ← linear ramp
step ≥ 1000:    lr = 3e-4 × 0.5 × (1 + cos(π × p))
                where p = (step - 1000) / (50000 - 1000)
```

| Step | LR |
|---|---|
| 0 | 0 |
| 500 | 1.5e-4 |
| 1,000 | 3e-4 (peak) |
| 10,000 | ~2.8e-4 |
| 25,000 | ~1.5e-4 |
| 40,000 | ~0.3e-4 |
| 50,000 | ~0 |

**Why warmup?**  
At step 0 the model weights are random. A large LR immediately would cause explosive
gradient updates. Warmup lets the optimizer stabilise before full-speed training.

**Why cosine decay?**  
Gradually reducing LR as training converges prevents overshooting the loss minimum.
Cosine is smooth — no sudden drops that could destabilise late training.

---

## 8. Data Pipeline

### Source Data

`ai4bharat/sangraha` — Hindi verified subset  
21 files, each ~109–110 MB compressed (`.jsonl.gz`), ~2.3 GB total compressed  
Estimated ~15 GB uncompressed text

### Per-File Processing (single streaming pass)

Each `.jsonl.gz` file is processed independently — no loading all files into memory at once.

```
for each line in gzip file:
    parse JSON → extract "text" field
    NFKC normalize (Unicode canonical form)
    collapse whitespace
    length filter: discard if chars < 50 or > 50,000
    Devanagari ratio filter: discard if < 60% of chars are Devanagari (U+0900–U+097F)
    tokenize: SentencePiece Unigram → token IDs
    append EOS token
    extend token buffer
    when buffer ≥ 5,120,000 tokens: pack into 512-token windows → flush to sequence list
end
final flush: pack remaining buffer
```

### Packing

Instead of padding short sequences to 512, consecutive token streams are concatenated
and sliced into exactly-512-token windows:

```
[doc1_tokens ... eos, doc2_tokens ... eos, doc3_tokens ...]
→ [t1..t512] [t513..t1024] [t1025..t1536] ...
```

This maximises token utilisation — no wasted padding tokens in any training batch.

### Splits (per file, 98/1/1)

```
packed sequences → 98% train / 1% val / 1% test
                   (minimum 4 sequences guaranteed for val+test pool)
                   seed = 42 (deterministic)
```

### Output Structure

```
data/tokenized/
├── train/
│   ├── part_0000/    ← Arrow dataset from file 0 (~90,486 sequences)
│   ├── part_0001/    ← Arrow dataset from file 1 (~90,935 sequences)
│   └── ...           ← part_0002 through part_0020 (21 files total)
├── val/
│   ├── part_0000/    ← ~923 sequences
│   └── ...
└── test/
    ├── part_0000/    ← ~923 sequences
    └── ...
```

**Incremental processing:** If `part_NNNN/` already exists, it is skipped on re-run.
Re-running `run_tokenize.py` only processes new files.

### Dataset Statistics (projected, 21 files)

| Split | Sequences | Tokens |
|---|---|---|
| Train | ~1,900,000 | ~973M |
| Val | ~19,390 | ~9.9M |
| Test | ~19,390 | ~9.9M |
| **Total** | **~1,938,780** | **~992M** |

---

## 9. Training Signals — What to Watch

### Loss Curve

| Training stage | Expected loss range | Notes |
|---|---|---|
| Step 0 (random) | ~10.37 | log(32000) — completely random |
| Step 1–1000 (warmup) | 8–10 | Very high, LR is still ramping up |
| Step 1000–5000 | 5–7 | LR at peak, rapid descent |
| Step 5000–20000 | 3–5 | Model learning Hindi structure |
| Step 20000–50000 | 2–3.5 | Converging |
| Step 50000 (final) | < 3.0 | Target |

Currently at step 364 with loss=97.1 — this is **warmup phase** (LR=1.09e-4, only 36% of
warmup done). The loss will drop steeply once LR reaches 3e-4 at step 1000.

### Perplexity Targets (val split)

```
PPL = exp(val_loss)
```

| PPL range | Interpretation |
|---|---|
| > 1000 | Early training / warmup phase |
| 200–1000 | Model learning basic structure |
| 100–200 | Fair — keep training |
| 50–100 | Good |
| < 50 | Excellent |

### TensorBoard Charts

Launch: `tensorboard --logdir artifacts/tb_logs`

| Chart | Logged every | What to watch |
|---|---|---|
| `train/loss` | 50 steps | Steady decrease; should halve every ~5k steps |
| `train/lr` | 50 steps | Ramp to 3e-4 at step 1000, then cosine decay |
| `train/tokens_per_sec` | 50 steps | ~16,000–20,000 tok/s expected on RTX 3000 Ada |
| `train/vram_gb` | 50 steps | Should stabilise around 0.8–1.2 GB |
| `val/loss` | 500 steps | Should track train/loss; if it diverges = overfitting |
| `val/perplexity` | 500 steps | Primary quality metric |

### GPU Monitoring

```cmd
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits
```

Expected during training:
- **GPU utilisation:** 20–50% (low because DataLoader is CPU-bound with num_workers=0)
- **VRAM used:** ~1,900–2,200 MB
- **Power draw:** 20–35W

Low GPU utilisation is expected — the bottleneck is the CPU loading batches sequentially
(num_workers=0 is required; no subprocesses). The GPU computes fast, then waits for CPU.

---

## 10. Checkpointing & Resume

### Checkpoint Contents

```
artifacts/checkpoints/
└── step_0001000/
    ├── model.pt       ← model weights (state_dict, all 46.54M params, bfloat16)
    ├── optimizer.pt   ← AdamW state (momentum + variance for every param, fp32)
    └── meta.json      ← complete snapshot of everything
```

`meta.json` contains:
```json
{
  "step": 1000,
  "loss": 4.2156,
  "model_config": {
    "hidden_size": 512, "num_layers": 10, "num_attention_heads": 8,
    "num_kv_heads": 2, "head_dim": 64, "intermediate_size": 1536,
    "max_seq_len": 512, "vocab_size": 32000, "rms_norm_eps": 1e-06,
    "rope_base": 10000.0, "tie_embeddings": true, "dropout": 0.1
  },
  "train_config": {
    "learning_rate": 0.0003, "warmup_steps": 1000, "max_steps": 50000,
    "per_device_batch_size": 2, "gradient_accumulation_steps": 16, ...
  }
}
```

### Auto-Resume

```bash
# Stop training at step 5000 with Ctrl+C
# Re-run the same command — it automatically picks up from step_0005000
python train.py
```

On startup, the trainer scans `artifacts/checkpoints/` for the highest `step_*` directory
and loads both `model.pt` and `optimizer.pt`. Training resumes from that exact step with
the same LR schedule position.

### Storage per Checkpoint

```
model.pt:      46.54M params × 2 bytes (bfloat16)  ≈  93 MB
optimizer.pt:  46.54M params × 8 bytes (fp32 ×2)   ≈ 372 MB
meta.json:     ~1 KB
Total per checkpoint: ~465 MB
50 checkpoints × 465 MB = ~22 GB total disk usage
```

Manually delete intermediate checkpoints to save disk space — only the final one and a
few recent ones are needed for recovery.

---

## 11. OOM Recovery

If CUDA runs out of memory, the trainer automatically retries with progressively more
aggressive memory reduction:

| Retry | Action | Memory saved |
|---|---|---|
| 1 | batch_size ÷ 2 (2 → 1) | ~50% activation memory |
| 2 | Enable gradient checkpointing | Recompute activations during backward; ~60% less activation memory |
| 3 | seq_len → 256 | Quadratic reduction in attention memory |
| 4 | Tier down: hidden=384, layers=8 | ~60% of original model size |
| 5 (exhausted) | Stop training | Log error and exit |

With SMALL tier on an 8.6 GB GPU, OOM should not trigger — estimated training VRAM is
well under 2 GB.

---

## 12. Model Tiers

| Tier | hidden | layers | Q heads | KV heads | intermediate | seq_len | Params | Use case |
|---|---|---|---|---|---|---|---|---|
| CPU_ONLY | 256 | 6 | 4 | 2 | 768 | 256 | ~12M | No GPU — CPU only |
| MICRO | 384 | 8 | 6 | 2 | 1024 | 512 | ~27M | Quick tests, 2 GB GPU |
| **SMALL** | **512** | **10** | **8** | **2** | **1536** | **512** | **~46M** | **Currently training; Pi target** |
| MEDIUM | 768 | 12 | 12 | 4 | 2048 | 1024 | ~117M | 12 GB GPU |
| LARGE | 1024 | 16 | 16 | 4 | 2752 | 2048 | ~290M | 24 GB GPU |

All tiers use the same architecture (RoPE, GQA, SwiGLU, RMSNorm) — only dimensions differ.
The SMALL tier was chosen as the Raspberry Pi MVP: sufficient quality at 46M params while
fitting comfortably in 8 GB RAM even with Q4_K_M quantization.

---

## 13. Post-Training Pipeline

After `train.py` completes (step 50,000):

### Step 1 — Evaluate

```powershell
cd slm_training
python evaluate.py
```

Produces:
- Val perplexity + test perplexity
- 5 Hindi generation samples with quality metrics
- `artifacts/reports/evaluation_step_0050000.md`

### Step 2 — Export

```powershell
python export.py
```

Produces:
```
artifacts/models/hindi_slm_v001/           ← HuggingFace format
artifacts/models/hindi_slm_v001_f16.gguf  ← GGUF F16 (~186 MB)
artifacts/models/hindi_slm_v001_q4_k_m.gguf  ← GGUF Q4_K_M (~26 MB)
```

GGUF conversion requires `llama.cpp` installed separately.

### Step 3 — Raspberry Pi Deployment

Copy `hindi_slm_v001_q4_k_m.gguf` to Pi and serve with `llama.cpp`:

```bash
./llama-server -m hindi_slm_v001_q4_k_m.gguf --ctx-size 512 --port 8080
```

---

## 14. File Map

```
slm_training/
├── run_tokenize.py          ← Tokenize all 21 .jsonl.gz files → Arrow parts
├── train.py                 ← Main training entry point
├── validate.py              ← Quick perplexity check on a checkpoint
├── evaluate.py              ← Full evaluation: perplexity + generation + report
├── export.py                ← HF format + GGUF F16 + Q4_K_M
│
├── src/slm_training/
│   ├── architecture.py      ← ModelConfig, TIER_CONFIGS, ParameterCounter, MemoryEstimator
│   ├── dataset.py           ← Text cleaning, per-file tokenization, DataLoader
│   ├── model.py             ← RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock, HindiSLM
│   ├── trainer.py           ← Training loop, AdamW, LR schedule, OOM recovery, checkpointing
│   ├── evaluator.py         ← Perplexity, Hindi generation, Devanagari ratio
│   └── exporter.py          ← HF export, GGUF conversion, Pi compatibility check
│
├── configs/
│   ├── training_config.yaml              ← Written at training start
│   ├── tokenized_dataset_manifest.json  ← Written after tokenization
│   └── optimal_model_schema.yaml        ← Model config + param counts
│
├── data/
│   ├── sangraha_verified_hin_10gb/      ← Raw .jsonl.gz source files (21 files)
│   └── tokenized/
│       ├── train/part_0000/ .. part_0020/
│       ├── val/part_0000/ .. part_0020/
│       └── test/part_0000/ .. part_0020/
│
└── artifacts/
    ├── checkpoints/step_NNNNNNN/        ← model.pt + optimizer.pt + meta.json
    ├── models/hindi_slm_v001/           ← HF format (post-training)
    ├── models/*.gguf                    ← GGUF exports (post-training)
    ├── reports/                         ← Evaluation markdown reports
    └── tb_logs/                         ← TensorBoard event files
```
