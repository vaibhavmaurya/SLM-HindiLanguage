"""
End-to-end TEST mode pipeline runner.
Equivalent to running the notebook with RUN_MODE = "TEST".
Sections 0-8 (skips Section 9 GGUF export — needs llama.cpp).
"""

import sys, os, json, time, multiprocessing
from pathlib import Path

# Force UTF-8 stdout on Windows so Devanagari characters print correctly.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Required on Windows: prevents worker processes from re-executing this script.
# Without this guard, datasets.map() and DataLoader workers that use spawn
# would re-import this file and run the entire pipeline again.
multiprocessing.freeze_support()
if __name__ != "__main__":
    sys.exit(0)

# ── Path setup ──────────────────────────────────────────────────────────────
NOTEBOOK_DIR      = Path(__file__).parent / "notebooks"
SLM_TRAINING_ROOT = Path(__file__).parent
PROJECT_ROOT      = SLM_TRAINING_ROOT.parent

SRC_DIR = SLM_TRAINING_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIGS_DIR   = SLM_TRAINING_ROOT / "configs"
DATA_RAW_DIR  = SLM_TRAINING_ROOT / "data" / "raw"
ARTIFACTS_DIR = SLM_TRAINING_ROOT / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
REPORTS_DIR   = ARTIFACTS_DIR / "reports"
TB_DIR        = SLM_TRAINING_ROOT / "logs" / "tensorboard"

# ── TEST mode parameters ─────────────────────────────────────────────────────
RUN_MODE      = "TEST"
_TARGET_BYTES = 10 * 1024 * 1024   # 10 MB
_DATASET_NAME = "sangraha_hin_test"
_MAX_STEPS    = 100
_WARMUP_STEPS = 10
_GRAD_ACCUM   = 1
_BATCH_SIZE   = 2
_LOG_EVERY    = 5
_EVAL_EVERY   = 25
_SAVE_EVERY   = 50
_NUM_PROC     = None  # None = single-threaded; avoids Windows multiprocess spawn issue
_NUM_WORKERS  = 0     # 0 = load in main process; avoids Windows spawn issue
DATA_TOK_DIR  = SLM_TRAINING_ROOT / "data" / "tokenized_test"
CKPT_DIR      = ARTIFACTS_DIR / "checkpoints_test"
MANIFEST_PATH = CONFIGS_DIR / "tokenized_dataset_manifest_test.json"

for d in [CONFIGS_DIR, DATA_RAW_DIR, DATA_TOK_DIR, ARTIFACTS_DIR, CKPT_DIR,
          MODELS_DIR, REPORTS_DIR, TB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# datasets (PyArrow) must be imported before torch on Windows to avoid DLL conflict
import datasets as _datasets_preload  # noqa: F401

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

t_start = time.time()
print("=" * 62)
print("  TEST MODE — end-to-end pipeline smoke test")
print(f"  Data   : {_TARGET_BYTES/1e6:.0f} MB  |  Steps: {_MAX_STEPS}")
print("=" * 62)

# ── Section 0: GPU check ─────────────────────────────────────────────────────
section("Section 0 — Environment")
import torch
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA      : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM      : {props.total_memory / 1e9:.2f} GB")
    print(f"bfloat16  : {torch.cuda.is_bf16_supported()}")
    DEVICE = torch.device("cuda")
    DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    print("WARNING: No GPU — running on CPU (slow)")
    DEVICE = torch.device("cpu")
    DTYPE  = torch.float32
print(f"Device={DEVICE}  dtype={DTYPE}")

# ── Section 1: System profiling ───────────────────────────────────────────────
section("Section 1 — System Profiling")
from slm_training.profiler import detect_profile, save_profile, print_profile_table
profile = detect_profile(disk_path=str(SLM_TRAINING_ROOT))
save_profile(profile, CONFIGS_DIR / "system_profile.yaml")
print_profile_table(profile)
TRAINING_TIER = profile.training_tier
print(f"Training tier: {TRAINING_TIER}")

# ── Section 2: Architecture deduction ────────────────────────────────────────
section("Section 2 — Architecture Deduction")
from slm_training.architecture import (
    NASCalculator, ParameterCounter, MemoryEstimator, KVCacheEstimator,
    save_model_schema,
)
TARGET_TIER = "SMALL"
MODEL_CFG   = NASCalculator.select(TARGET_TIER)
NASCalculator.print_summary(MODEL_CFG)

param_counts = ParameterCounter.count(MODEL_CFG)
total_params = param_counts["total"]
print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
assert 30_000_000 < total_params < 120_000_000, f"Unexpected param count: {total_params/1e6:.2f}M"
save_model_schema(MODEL_CFG, param_counts, CONFIGS_DIR / "optimal_model_schema.yaml")
print("PASS: param count in expected range")

# ── Section 3: Load local Sangraha data ──────────────────────────────────────
section("Section 3 — Load Local Sangraha Data")
from slm_training.dataset import load_local_sangraha

LOCAL_SANGRAHA_DIR = SLM_TRAINING_ROOT / "data" / "sangraha_verified_hin_10gb"
_MAX_FILES = 1   # TEST: load only 1 jsonl.gz file (~500 MB compressed, ~1 GB text)

raw_ds = load_local_sangraha(LOCAL_SANGRAHA_DIR, max_files=_MAX_FILES)
print(f"Rows loaded : {len(raw_ds):,}")
print(f"Columns     : {raw_ds.column_names}")
for i in range(min(2, len(raw_ds))):
    txt = raw_ds[i]["text"]
    print(f"  [{i}] {txt[:100].encode('ascii', errors='replace').decode()}...")
assert len(raw_ds) > 0, "No rows loaded from local data!"
print("PASS: local data loaded")

# ── Section 4: Tokenizer ──────────────────────────────────────────────────────
section("Section 4 — Tokenizer Freezing")
import yaml
from transformers import AutoTokenizer

LOCAL_TOK = PROJECT_ROOT / "tokenizer_training" / "data" / "final" / "hindi_slm_tokenizer_v001"
HF_TOK    = "vaibhavmaurya/hindi-slm-tokenizer-v001"

if LOCAL_TOK.exists():
    print(f"Loading from local: {LOCAL_TOK}")
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_TOK))
else:
    print(f"Loading from HF Hub: {HF_TOK}")
    tokenizer = AutoTokenizer.from_pretrained(HF_TOK)

print(f"Vocab size: {tokenizer.vocab_size}")
assert tokenizer.vocab_size == 32000, f"Expected 32000, got {tokenizer.vocab_size}"
for tok in ["<pad>", "<unk>", "<s>", "</s>"]:
    assert tok in tokenizer.get_vocab(), f"Missing: {tok}"
print(f"Special tokens: {tokenizer.all_special_tokens}")

# Round-trip
test_sents = [
    "नमस्ते! भारत एक महान देश है।",
    "हिंदी भाषा बहुत सुंदर है।",
    "विज्ञान और तकनीक में भारत ने प्रगति की है।",
]
ok_count = 0
for sent in test_sents:
    ids = tokenizer.encode(sent, add_special_tokens=False)
    dec = tokenizer.decode(ids, skip_special_tokens=True)
    unk = ids.count(tokenizer.unk_token_id)
    passed = dec.strip() == sent.strip() and unk == 0
    safe = sent[:40].encode("ascii", errors="replace").decode()
    print(f"  [{'PASS' if passed else 'FAIL'}] tokens={len(ids)}  unk={unk}  '{safe}'")
    if passed:
        ok_count += 1
assert ok_count == len(test_sents), "Round-trip test failures!"

MODEL_CFG.vocab_size = tokenizer.vocab_size
tok_cfg = {"tokenizer_name": HF_TOK, "vocab_size": tokenizer.vocab_size,
           "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
with open(CONFIGS_DIR / "tokenizer_config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(tok_cfg, f, default_flow_style=False, allow_unicode=True)
print("PASS: tokenizer OK")

# ── Section 5: Tokenize + Pack ────────────────────────────────────────────────
section("Section 5 — Clean + Tokenize + Pack")
from slm_training.dataset import build_tokenized_splits

if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    print(f"Tokenized splits already exist — skipping.")
    print(f"  train={manifest['splits']['train']:,}  val={manifest['splits']['val']:,}  test={manifest['splits']['test']:,}")
else:
    manifest = build_tokenized_splits(
        raw_dataset_or_path=raw_ds,
        tokenized_dir=DATA_TOK_DIR,
        manifest_path=MANIFEST_PATH,
        tokenizer=tokenizer,
        seq_len=MODEL_CFG.max_seq_len,
        seed=42,
        num_proc=_NUM_PROC,
    )

print(f"\nManifest:")
for k, v in manifest.items():
    print(f"  {k}: {v}")

# Verify packed sequences
from datasets import load_from_disk
train_ds = load_from_disk(str(DATA_TOK_DIR / "train"))
sample = train_ds[0]
assert len(sample["input_ids"]) == MODEL_CFG.max_seq_len, "Packed length mismatch!"
print(f"\nTrain examples  : {len(train_ds):,}")
print(f"input_ids length: {len(sample['input_ids'])}")
decoded_preview = tokenizer.decode(sample["input_ids"][:30], skip_special_tokens=False)
print(f"First 30 tokens : {decoded_preview}")
print("PASS: tokenized dataset verified")

# ── Section 6: Model architecture ─────────────────────────────────────────────
section("Section 6 — Model Architecture")
from slm_training.model import HindiSLM

model = HindiSLM(MODEL_CFG)
n_params = model.count_parameters()
print(f"Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")

# Forward pass smoke test
model.eval()
rand_ids = torch.randint(0, MODEL_CFG.vocab_size, (2, MODEL_CFG.max_seq_len))
with torch.no_grad():
    logits, loss = model(rand_ids, labels=rand_ids)
assert logits.shape == (2, MODEL_CFG.max_seq_len, MODEL_CFG.vocab_size), f"Bad logits shape: {logits.shape}"
assert loss is not None
print(f"Forward pass: logits={logits.shape}  loss={loss.item():.4f}")

# Dropout verification
model.train()
with torch.no_grad():
    logits_train, _ = model(rand_ids)
print(f"Train mode logits shape: {logits_train.shape}")
print(f"embed_drop   : {model.embed_drop}")
print(f"resid_drop[0]: {model.layers[0].resid_drop}")
print(f"attn_drop[0] : {model.layers[0].attn.attn_dropout}")
print("PASS: model architecture verified")

# ── Section 7: Training ───────────────────────────────────────────────────────
section("Section 7 — Training (TEST mode: 100 steps)")
from slm_training.trainer import TrainingConfig, save_training_config, train
from slm_training.dataset import make_dataloader

TRAIN_CFG = TrainingConfig(
    learning_rate=3e-4,
    weight_decay=0.1,
    max_steps=_MAX_STEPS,
    warmup_steps=_WARMUP_STEPS,
    per_device_batch_size=_BATCH_SIZE,
    gradient_accumulation_steps=_GRAD_ACCUM,
    max_grad_norm=1.0,
    num_workers=_NUM_WORKERS,
    log_every=_LOG_EVERY,
    eval_every=_EVAL_EVERY,
    save_every=_SAVE_EVERY,
    dtype="bfloat16",
    max_oom_retries=5,
)
save_training_config(TRAIN_CFG, CONFIGS_DIR / "training_config_test.yaml")
print(f"Max steps       : {TRAIN_CFG.max_steps}")
print(f"Effective batch : {TRAIN_CFG.per_device_batch_size} × {TRAIN_CFG.gradient_accumulation_steps}")
print(f"Warmup steps    : {TRAIN_CFG.warmup_steps}")

train_loader = make_dataloader(DATA_TOK_DIR / "train", batch_size=_BATCH_SIZE,
                               num_workers=_NUM_WORKERS, shuffle=True)
val_loader   = make_dataloader(DATA_TOK_DIR / "val",   batch_size=_BATCH_SIZE,
                               num_workers=min(2, _NUM_WORKERS), shuffle=False)

print(f"Train batches   : {len(train_loader):,}")
print(f"Val batches     : {len(val_loader):,}")

# Verify batch shape
sample_batch = next(iter(train_loader))
print(f"Batch shape     : {sample_batch['input_ids'].shape}")

try:
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=str(TB_DIR))
    print(f"TensorBoard writer: {TB_DIR}")
except Exception as e:
    tb_writer = None
    print(f"TensorBoard unavailable: {e}")

t_train_start = time.time()
train(
    model=model,
    model_cfg=MODEL_CFG,
    train_cfg=TRAIN_CFG,
    train_loader=train_loader,
    val_loader=val_loader,
    ckpt_dir=CKPT_DIR,
    device=DEVICE,
    tb_writer=tb_writer,
)
t_train = time.time() - t_train_start
print(f"\nTraining time: {t_train:.1f}s ({t_train/60:.1f} min)")
print("PASS: training completed")

# ── Section 8: Evaluation ──────────────────────────────────────────────────────
section("Section 8 — Evaluation")
from slm_training.trainer import _find_latest_checkpoint, _load_checkpoint, _build_optimizer
from slm_training.evaluator import compute_perplexity, generate_samples, print_samples, write_evaluation_report

latest_ckpt = _find_latest_checkpoint(CKPT_DIR)
if latest_ckpt:
    optimizer = _build_optimizer(model, TRAIN_CFG)
    step = _load_checkpoint(latest_ckpt, model, optimizer)
    model = model.to(DEVICE)
    print(f"Loaded checkpoint: {latest_ckpt.name}  (step {step})")
else:
    step = 0
    model = model.to(DEVICE)
    print("No checkpoint — evaluating current model weights")

val_ppl = compute_perplexity(model, val_loader, DEVICE, DTYPE, max_batches=50)
print(f"Validation perplexity: {val_ppl:.2f}")

test_loader = make_dataloader(DATA_TOK_DIR / "test", batch_size=_BATCH_SIZE,
                              num_workers=_NUM_WORKERS, shuffle=False)
test_ppl = compute_perplexity(model, test_loader, DEVICE, DTYPE, max_batches=50)
print(f"Test perplexity      : {test_ppl:.2f}")

samples = generate_samples(model, tokenizer, DEVICE, max_new_tokens=80,
                           temperature=0.8, top_p=0.9)
print_samples(samples)

write_evaluation_report(
    report_path=REPORTS_DIR / "evaluation_report.md",
    val_perplexity=val_ppl,
    test_perplexity=test_ppl,
    samples=samples,
    step=step,
)
print(f"Evaluation report: {REPORTS_DIR / 'evaluation_report.md'}")
print("PASS: evaluation completed")

# ── Summary ───────────────────────────────────────────────────────────────────
t_total = time.time() - t_start
section(f"PIPELINE COMPLETE — {t_total:.1f}s ({t_total/60:.1f} min)")
print(f"Val perplexity : {val_ppl:.2f}")
print(f"Test perplexity: {test_ppl:.2f}")
print(f"Checkpoints    : {CKPT_DIR}")
print(f"Eval report    : {REPORTS_DIR / 'evaluation_report.md'}")
print("\nAll sections PASSED — end-to-end pipeline is healthy.")
