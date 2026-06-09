# Action Items — Hindi SLM Training

## Pre-Run Checklist

### Environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- [ ] Close Ollama and any other GPU processes before training
- [ ] Ensure at least **10 GB free disk** in `slm_training/` directory

### HuggingFace Token (for tokenizer download if not cached locally)
- [ ] Run `huggingface-cli login` and paste your HF read token
- [ ] Or set env var: `HUGGING_FACE_HUB_TOKEN=hf_...` in your shell

### llama.cpp (for Section 9 GGUF export)
Install with CUDA support:
```bash
# Option A: Pre-built wheel (CUDA 12.x)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Option B: Build from source with CUDA
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# After building, set path for exporter.py to find convert script:
# The script looks for llama.cpp/convert_hf_to_gguf.py in:
#   ./llama.cpp/   or   ~/llama.cpp/
```

---

## Per-Stage Success Criteria & Failure Guide

### Section 0 — Environment Setup
**Success:** GPU detected, VRAM ≥ 4 GB, bfloat16=True, `DEVICE=cuda`  
**Failure:**
- No GPU → training will run on CPU (~100× slower). Consider Google Colab with T4.
- `torch` import fails → check CUDA version compatibility with PyTorch wheel.

### Section 1 — System Profiling
**Success:** `configs/system_profile.yaml` written, tier shown (expect `SMALL` for RTX 3000 Ada)  
**Failure:** `psutil` not installed → install with `pip install psutil`; profiler will still work without it.

### Section 2 — Architecture Deduction
**Success:** Total params ≈ 68M (assertion checks 60M–120M), `configs/optimal_model_schema.yaml` written  
**Failure:** Assertion fails → `TARGET_TIER` changed to an incompatible tier; force `TARGET_TIER = "SMALL"`.

### Section 3 — Data Download
**Success:** `data/raw/sangraha_hin_1gb/` exists, sentinel file present, row count ≥ 100k  
**Failure:**
- 429 rate limit from HuggingFace → run `huggingface-cli login`; try again after a few minutes.
- Disk full → free space or change `DATA_RAW_DIR` to a drive with more space.
- Dataset config error → `ai4bharat/sangraha` may require `trust_remote_code=True` (already set).

### Section 4 — Tokenizer Freezing
**Success:** `vocab_size == 32000`, all round-trips PASS, no UNK tokens in test sentences  
**Failure:**
- Tokenizer not found locally → will auto-fall-back to HuggingFace download (requires login).
- Round-trip FAIL for a sentence → acceptable if the sentence contains rare Unicode not in training corpus. Check `unk_count` per sentence.

### Section 5 — Clean + Tokenize + Pack
**Success:** `configs/tokenized_dataset_manifest.json` written, train example `input_ids` length == 512  
**Failure:**
- OOM during `datasets.map()` → reduce `num_proc=6` to `num_proc=2`.
- Very few sequences after cleaning → lower `MIN_DEVANAGARI_RATIO` in `dataset.py` from 0.60 to 0.50.
- Already tokenized? → delete `configs/tokenized_dataset_manifest.json` to re-run.

### Section 7 — Training
**Success:** Loss decreases from ~10+ toward <4, checkpoints appear in `artifacts/checkpoints/`  
**Failure:**
- CUDA OOM on first step → OOM recovery will auto-trigger. Watch for "Retry N:" messages.
- If all 5 retries fail → manually set `TRAIN_CFG.per_device_batch_size = 1` and `gradient_accumulation_steps = 32`.
- Loss explodes (NaN) → check `max_grad_norm` (should be 1.0); reduce `learning_rate` to 1e-4.
- Checkpoint not resuming → verify `artifacts/checkpoints/step_*/meta.json` exists.
- Training too slow → normal for 50k steps on laptop. Estimate: ~3–7 days continuous. Interrupt safely (Ctrl+C saves checkpoint).

### Section 8 — Evaluation
**Success:** Perplexity < 200 (untrained baseline ~32000; good threshold after 10k steps: < 100)  
**Failure:**
- Generated text is garbage → expected before 5k steps. Resume training.
- Many `<unk>` tokens in output → tokenizer round-trip issue; recheck Section 4.

### Section 9 — Edge Export
**Success:** `artifacts/models/hindi_slm_v001/` exists, Pi compatibility PASS  
**Failure:**
- GGUF convert fails → check that `llama.cpp/convert_hf_to_gguf.py` exists in the search paths.
- `llama-quantize` not found → add llama.cpp `build/bin/` to PATH.
- Pi FAIL (total > 6 GB) → unexpected for 68M model; verify `MODEL_CFG.vocab_size == 32000` (not larger).

---

## Hardware Tuning Knobs

| Setting | Default | When to change |
|---------|---------|----------------|
| `per_device_batch_size` | 2 | Reduce to 1 if OOM persists after all retries |
| `gradient_accumulation_steps` | 16 | Increase to 32 if batch reduced to 1 |
| `num_workers` (DataLoader) | 6 | Reduce to 2 if CPU throttles or RAM pressure |
| `num_proc` (datasets.map) | 6 | Reduce to 2 if tokenization OOMs |
| `max_steps` | 50,000 | Reduce for a quick smoke run (e.g., 1000) |
| `eval_every` | 500 | Reduce to 100 for faster feedback early in training |
| `save_every` | 1000 | Reduce to 500 if you want finer checkpoint granularity |

---

## Quick Smoke Run (10 minutes)

To verify the full pipeline without waiting days:

```python
# In Section 7, change:
TRAIN_CFG = TrainingConfig(
    max_steps=500,
    warmup_steps=50,
    per_device_batch_size=2,
    gradient_accumulation_steps=4,
    log_every=10,
    eval_every=100,
    save_every=100,
)
```

Then run Sections 0–9. Loss will be high but pipeline correctness is verified.

---

## Post-Training Tasks

- [ ] Upload to HuggingFace Hub:
  ```bash
  huggingface-cli upload <your-username>/hindi-slm-v001 artifacts/models/hindi_slm_v001/
  ```
- [ ] Copy GGUF to Raspberry Pi and test:
  ```bash
  scp artifacts/models/gguf/hindi_slm_v001_q4_k_m.gguf pi@raspberrypi:~/models/
  # On Pi:
  ./llama-server -m ~/models/hindi_slm_v001_q4_k_m.gguf --host 0.0.0.0 --port 8080
  ```
- [ ] Record metrics in project wiki: test perplexity, tokens/sec on Pi, generation quality
- [ ] Open `logs/tensorboard/` in TensorBoard to visualize training curves:
  ```bash
  tensorboard --logdir slm_training/logs/tensorboard
  ```

---

## Future Extensions

| Extension | Estimated effort | Notes |
|-----------|-----------------|-------|
| Increase context to 1024 tokens | Low | Rerun Section 5 with `seq_len=1024`; recheck Pi KV cache |
| Train 300M parameter variant | Medium | Change `TARGET_TIER="MEDIUM"`; needs gradient checkpointing throughout |
| Instruction tuning | High | Requires labeled Hindi QA dataset; add instruct-format tokens |
| Multi-GPU training | High | Use `accelerate launch` with FSDP; not needed for 68M model |
| Flash Attention 2 | Low | `pip install flash-attn`; replace `F.scaled_dot_product_attention` call |
| Upload GGUF to HF Hub | Low | `huggingface-cli upload ... *.gguf` |
