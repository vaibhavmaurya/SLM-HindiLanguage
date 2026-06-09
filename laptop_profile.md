# Laptop Hardware Profile — Hindi SLM Development Machine

**Profiled:** 2026-05-17  
**Purpose:** Determine viable SLM training scale, batch sizes, and training strategies for the Hindi SLM project.

---

## Hardware Specifications

### CPU — Intel Core i7-13700H (13th Gen, Raptor Lake-H)

| Property | Value |
|---|---|
| Model | Intel Core i7-13700H |
| Architecture | Raptor Lake-H (13th Gen) |
| Total cores | 14 (6 P-cores + 8 E-cores) |
| Total threads | 20 |
| Base clock | 2.4 GHz |
| Max turbo boost | ~5.0 GHz (P-cores) |
| L2 cache | 11.5 MB |
| L3 cache | 24 MB |
| TDP | 45W (laptop) |
| Instruction sets | AVX-512, AVX2, SSE4.2 |

**Notes:** The P-cores (Performance) handle compute-intensive tasks; E-cores (Efficiency) handle background work. PyTorch CPU operations benefit from AVX2 for fallback ops. DataLoader workers should use `num_workers` ≤ 6 on this machine to avoid throttling.

---

### GPU — NVIDIA RTX 3000 Ada Generation Laptop GPU

| Property | Value |
|---|---|
| Model | NVIDIA RTX 3000 Ada Generation Laptop GPU |
| Architecture | Ada Lovelace |
| VRAM | **8 GB GDDR6** (8188 MiB) |
| CUDA version | 13.1 |
| Driver version | 591.86 |
| Max TDP | 50W (laptop mode) |
| Tensor Cores | 4th Gen (Ada) — native bfloat16 and FP8 support |
| Current VRAM usage | 0 MiB (idle; Ollama using N/A shared) |

**Notes:** Ada Lovelace architecture has native bfloat16 Tensor Cores — use `torch.bfloat16` for training, not `float16`, to avoid gradient underflow. The 50W TDP cap is a hard limit in laptop mode; sustained GPU compute may throttle clock speeds.

---

### RAM — 32 GB DDR5

| Property | Value |
|---|---|
| Total installed | **32 GB** |
| Configuration | 1 × 32 GB DIMM (single-channel) |
| Speed | 5600 MHz |
| Type | DDR5 |
| Slots used | 1 of 2 |

**Notes:** Single-channel DDR5 limits memory bandwidth. A second 32 GB DIMM (dual-channel) would roughly double bandwidth and improve data-loading throughput significantly. CPU offloading (`device_map="auto"`) can use RAM for model layers that don't fit in VRAM.

---

### Integrated GPU — Intel UHD Graphics

| Property | Value |
|---|---|
| Model | Intel UHD Graphics |
| VRAM (shared) | 2 GB (shared from system RAM) |
| Role | Display only — do NOT use for ML training |

---

### Storage — NVMe SSD

| Property | Value |
|---|---|
| Drive | C: |
| Total | ~953 GB |
| Used | ~620 GB |
| Free | **~332 GB** |

**Notes:** NVMe SSD is fast enough for DataLoader I/O and Parquet streaming. Keep training data and checkpoints on C: (not external drives) to avoid I/O bottlenecks.

---

## SLM Training Capacity Analysis

### VRAM Budget (8 GB)

The VRAM consumed during training (full precision, Adam optimizer):

| Component | Memory (per parameter) |
|---|---|
| Model weights (bfloat16) | 2 bytes |
| Gradients (bfloat16) | 2 bytes |
| Adam optimizer states (fp32) | 8 bytes |
| **Total per parameter** | **12–16 bytes** |
| Activations (depends on batch size) | variable |

| Model Size | Weights Only (bf16) | + Gradients + Adam | Feasibility |
|---|---|---|---|
| 125M params | 0.25 GB | ~2.0 GB | ✅ Comfortable — full training |
| 300M params | 0.6 GB | ~4.5 GB | ✅ Training with small batch |
| 500M params | 1.0 GB | ~7.5 GB | ⚠️ Tight — need gradient checkpointing |
| 700M params | 1.4 GB | ~10.5 GB | ❌ Exceeds VRAM — need CPU offload or FSDP |
| 1B params | 2.0 GB | ~15 GB | ❌ Requires DeepSpeed ZeRO-3 or CPU offload |
| 3B params | 6.0 GB | ~45 GB | ❌ Inference only with quantization |
| 7B params | 14 GB | ~95 GB | ❌ Inference only with 4-bit quantization |

### Recommended Training Target

**300M–500M parameters** is the optimal range for this machine:
- Fits in 8 GB VRAM with gradient checkpointing + bfloat16 mixed precision
- Batch size 4–8 with gradient accumulation steps 8–16 gives effective batch 32–128
- Feasible training time on 10 GB Hindi corpus: estimated 3–10 days continuous

### Practical Techniques Required

| Technique | Purpose | When to use |
|---|---|---|
| **bfloat16 mixed precision** | Halve VRAM, faster Tensor Cores | Always — use `torch.bfloat16` |
| **Gradient checkpointing** | Trade compute for VRAM (~30% VRAM reduction) | Models > 300M |
| **Gradient accumulation** | Simulate large batch without VRAM cost | Always (steps=8–16) |
| **DataLoader pin_memory=True** | Faster CPU→GPU transfer | Always on dedicated GPU |
| **`torch.compile()`** | Speed boost on Ada Tensor Cores | After baseline works |
| **DeepSpeed ZeRO-2** | Shard optimizer states across CPU+GPU | If > 500M params |
| **4-bit quantization (QLoRA)** | Fine-tune large models on small GPU | Inference/fine-tuning only |
| **Flash Attention 2** | Memory-efficient attention | When sequence length > 512 |

### Inference Capacity (no training overhead)

| Model Size | Precision | Fits in VRAM? | Tokens/sec (est.) |
|---|---|---|---|
| 1B | bfloat16 | ⚠️ Borderline (2 GB) | ~50–80 |
| 3B | bfloat16 | ❌ No | — |
| 3B | int4 | ✅ Yes (~1.8 GB) | ~30–50 |
| 7B | int4 | ⚠️ Borderline (4 GB) | ~15–25 |
| 13B | int4 | ❌ No | — |

---

## Tokenizer Training Capacity

Tokenizer training is CPU-bound and RAM-bound (no GPU needed):

| Stage | Resource | This Machine |
|---|---|---|
| 5 GB corpus sample | RAM during training | ~8–12 GB peak → fits in 32 GB |
| 10 GB corpus sample | RAM during training | ~15–20 GB peak → fits in 32 GB |
| Training time (32k vocab, 5 GB) | CPU time | ~10–30 minutes |
| Training time (32k vocab, 10 GB) | CPU time | ~20–60 minutes |

The HuggingFace `tokenizers` library uses Rust under the hood — it will use all 20 threads efficiently.

---

## Recommended SLM Architecture for This Machine

Based on the hardware profile, the recommended model size for the Hindi SLM:

```
Architecture: GPT-style decoder-only transformer
Parameters:   300M–500M
Hidden size:  768–1024
Layers:       12–16
Attention heads: 12–16
Vocab size:   32,000 (from tokenizer)
Max sequence: 1024–2048 tokens
Precision:    bfloat16
Batch size:   4 (per GPU) × 16 (gradient accumulation) = 64 effective
```

**Why not larger?**
- 700M+ parameters will exceed VRAM budget even with gradient checkpointing
- Laptop TDP cap (50W GPU) means sustained performance is lower than desktop equivalent

**Why not smaller?**
- 125M is too small for useful Hindi language modeling with 32k vocab
- 300M is the minimum where the model can learn complex Hindi grammar

---

## How to Re-Profile This Machine

Run these commands to regenerate this profile:

### GPU (NVIDIA)
```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv
```

### GPU (via Python / PyTorch)
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"CUDA version: {torch.version.cuda}")
print(f"bfloat16 supported: {torch.cuda.is_bf16_supported()}")
```

### CPU and RAM (PowerShell)
```powershell
# CPU
Get-WmiObject -Class Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, L2CacheSize, L3CacheSize

# RAM
Get-WmiObject -Class Win32_PhysicalMemory | Select-Object DeviceLocator, Capacity, Speed, MemoryType

# RAM total
(Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB
```

### Disk
```powershell
Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{N='Used(GB)';E={[math]::Round($_.Used/1GB,1)}}, @{N='Free(GB)';E={[math]::Round($_.Free/1GB,1)}}
```

### Full PyTorch benchmark
```python
import torch, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Matrix multiply benchmark (TFLOPS estimate)
N = 4096
a = torch.randn(N, N, dtype=torch.bfloat16, device=device)
b = torch.randn(N, N, dtype=torch.bfloat16, device=device)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

flops = 2 * N**3 * 10
tflops = flops / elapsed / 1e12
print(f"bfloat16 matmul throughput: {tflops:.2f} TFLOPS")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
```

---

## Summary

| Component | Spec | Training Implication |
|---|---|---|
| GPU | RTX 3000 Ada 8 GB VRAM | Max ~500M params with gradient checkpointing |
| CPU | i7-13700H 14-core | Fast tokenizer training; 6 DataLoader workers |
| RAM | 32 GB DDR5 (single-channel) | CPU offload possible; add 2nd DIMM for bandwidth |
| Storage | ~332 GB free NVMe | Sufficient for corpus + checkpoints |
| **Verdict** | — | **Train a 300M–500M Hindi SLM natively on this machine** |
