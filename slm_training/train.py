"""
CLI: train the Hindi SLM from tokenized Arrow splits.

Usage:
  python train.py                                  # full 50k-step run (SMALL tier)
  python train.py --max-steps 1000                 # short test run
  python train.py --tier MICRO                     # smaller model
  python train.py --ckpt-dir artifacts/ckpts_v2   # custom checkpoint dir
  python train.py --tb-dir artifacts/tb_v2         # custom TensorBoard dir
"""

import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# ── Windows: force UTF-8 output ──────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
SLM_TRAINING_ROOT = Path(__file__).parent
SRC_DIR = SLM_TRAINING_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sep(title: str = ""):
    bar = "─" * 64
    if title:
        print(f"\n{bar}\n  {title}\n{bar}", flush=True)
    else:
        print(bar, flush=True)


def _find_latest_ckpt(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return ckpts[-1] if ckpts else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Import dataset module early — datasets library must init before PyTorch/CUDA
    from slm_training.dataset import make_dataloader  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Train the Hindi SLM from tokenized Arrow splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenized-dir", default=str(SLM_TRAINING_ROOT / "data" / "tokenized"),
                        help="Directory with train/val/test Arrow splits.")
    parser.add_argument("--ckpt-dir", default=str(SLM_TRAINING_ROOT / "artifacts" / "checkpoints"),
                        help="Where to save (and auto-resume) checkpoints.")
    parser.add_argument("--tb-dir", default=str(SLM_TRAINING_ROOT / "artifacts" / "tb_logs"),
                        help="TensorBoard log directory.")
    parser.add_argument("--tier", default="SMALL",
                        choices=["CPU_ONLY", "MICRO", "SMALL", "MEDIUM", "LARGE"],
                        help="Model size tier.")
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=2,
                        help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch × grad_accum).")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only, no subprocesses).")
    parser.add_argument("--pack-factor", type=int, default=1,
                        help="Concatenate N consecutive sequences (e.g. 2 turns 512-token data into 1024).")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    tokenized_dir = Path(args.tokenized_dir)
    ckpt_dir = Path(args.ckpt_dir)
    tb_dir = Path(args.tb_dir)

    sep("Hindi SLM — Training")
    log(f"Tokenized dir  : {tokenized_dir}")
    log(f"Checkpoint dir : {ckpt_dir}")
    log(f"TensorBoard    : {tb_dir}")
    log(f"Tier           : {args.tier}")
    log(f"Max steps      : {args.max_steps:,}")
    log(f"Batch size     : {args.batch}  (effective = {args.batch * args.grad_accum})")
    log(f"Grad accum     : {args.grad_accum}")
    log(f"Learning rate  : {args.lr}")
    log(f"Warmup steps   : {args.warmup}")
    log(f"dtype          : {args.dtype}")
    log(f"Log every      : {args.log_every} steps")
    log(f"Eval every     : {args.eval_every} steps")
    log(f"Save every     : {args.save_every} steps")

    # ── Pre-flight: verify tokenized splits exist ─────────────────────────────
    sep("Pre-flight checks")
    for split in ("train", "val"):
        split_path = tokenized_dir / split
        if not split_path.exists():
            log(f"ERROR: '{split}' split not found: {split_path}")
            log(f"       Run run_tokenize.py first.")
            raise SystemExit(1)
        log(f"  {split:5s} split : {split_path}  OK")

    manifest_path = SLM_TRAINING_ROOT / "configs" / "tokenized_dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        log(f"  Manifest      : train={manifest['splits']['train']:,}  "
            f"val={manifest['splits']['val']:,} sequences")
    else:
        log("  (No manifest found — using splits directly)")

    # ── Device ────────────────────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = props.total_memory / 1e9
        vram_free_gb = (props.total_memory - torch.cuda.memory_allocated()) / 1e9
        log(f"  Device        : {props.name}  ({vram_total_gb:.1f} GB total, {vram_free_gb:.1f} GB free)")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            log("  WARNING: bfloat16 not supported on this GPU — switching to float16")
            args.dtype = "float16"
    else:
        device = torch.device("cpu")
        vram_total_gb = 0.0
        log("  WARNING: No CUDA device found. Training on CPU will be extremely slow.")

    # ── Model config ──────────────────────────────────────────────────────────
    sep("Model Architecture")
    from slm_training.architecture import TIER_CONFIGS, ParameterCounter, MemoryEstimator
    model_cfg = TIER_CONFIGS[args.tier]
    param_counts = ParameterCounter.count(model_cfg)
    mem_est = MemoryEstimator.estimate_training_vram_gb(model_cfg, batch_size=args.batch)

    log(f"  hidden_size       : {model_cfg.hidden_size}")
    log(f"  num_layers        : {model_cfg.num_layers}")
    log(f"  num_attn_heads    : {model_cfg.num_attention_heads}  (kv_heads={model_cfg.num_kv_heads})")
    log(f"  intermediate_size : {model_cfg.intermediate_size}")
    log(f"  max_seq_len       : {model_cfg.max_seq_len}")
    log(f"  vocab_size        : {model_cfg.vocab_size:,}")
    log(f"  Total parameters  : {param_counts['total'] / 1e6:.2f}M")
    log(f"  Est. train VRAM   : {mem_est['total_estimated_gb']:.2f} GB")
    if torch.cuda.is_available():
        if mem_est['total_estimated_gb'] > vram_total_gb * 0.95:
            log(f"  WARNING: estimated VRAM ({mem_est['total_estimated_gb']:.1f} GB) "
                f"is close to / exceeds device ({vram_total_gb:.1f} GB)")
            log(f"           OOM recovery will trigger. Consider --tier MICRO or --batch 1.")
        else:
            log(f"  VRAM check OK  ({mem_est['total_estimated_gb']:.2f} GB est. / {vram_total_gb:.1f} GB available)")

    # ── Existing checkpoints / auto-resume ────────────────────────────────────
    sep("Checkpoint / Resume")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    existing_ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    start_step = 0
    if existing_ckpts:
        latest_ckpt = existing_ckpts[-1]
        with open(latest_ckpt / "meta.json") as f:
            ckpt_meta = json.load(f)
        start_step = ckpt_meta["step"]
        log(f"  Found {len(existing_ckpts)} checkpoint(s)")
        log(f"  Latest  : {latest_ckpt.name}  (step {start_step:,}, loss {ckpt_meta['loss']:.4f})")
        log(f"  Training will RESUME from step {start_step:,}")
        if args.max_steps <= start_step:
            log(f"  Already at or past max_steps={args.max_steps:,}. Nothing to train.")
            log(f"  Pass --max-steps {start_step + 10000} to continue, or run evaluate.py.")
            raise SystemExit(0)
        log(f"  Steps remaining: {args.max_steps - start_step:,}")
    else:
        log("  No existing checkpoints found — training from scratch.")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    sep("DataLoaders")
    from slm_training.dataset import make_dataloader
    log("  Building train DataLoader (shuffle=True) ...")
    train_loader = make_dataloader(
        tokenized_dir / "train",
        batch_size=args.batch,
        shuffle=True,
        pack_factor=args.pack_factor,
    )
    log("  Building val DataLoader (shuffle=False) ...")
    val_loader = make_dataloader(
        tokenized_dir / "val",
        batch_size=args.batch,
        shuffle=False,
        pack_factor=args.pack_factor,
    )
    eff_batch = args.batch * args.grad_accum
    tokens_per_step = eff_batch * model_cfg.max_seq_len
    steps_remaining = args.max_steps - start_step
    total_tokens = tokens_per_step * steps_remaining
    log(f"  Train batches/epoch : {len(train_loader):,}")
    log(f"  Val batches         : {len(val_loader):,}")
    log(f"  Effective batch     : {eff_batch} sequences × {model_cfg.max_seq_len} tokens")
    log(f"  Tokens per step     : {tokens_per_step:,}")
    log(f"  Tokens to train     : ~{total_tokens / 1e9:.2f}B  ({steps_remaining:,} steps)")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    sep("TensorBoard")
    tb_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        log(f"  Writer ready  : {tb_dir}")
        log(f"  Launch with   : tensorboard --logdir {tb_dir}")
        log(f"  Tracks        : train/loss, train/lr, train/tokens_per_sec, val/loss, val/perplexity")
    except ImportError:
        tb_writer = None
        log("  WARNING: tensorboard not installed — no TensorBoard logging.")
        log("           pip install tensorboard")

    # ── Build model ───────────────────────────────────────────────────────────
    sep("Building Model")
    from slm_training.model import HindiSLM
    log(f"  Instantiating HindiSLM ({args.tier} tier) ...")
    model = HindiSLM(model_cfg)
    actual_params = model.count_parameters()
    log(f"  Actual parameters : {actual_params / 1e6:.2f}M")
    log(f"  Moving model to {device} ...")
    model = model.to(device)
    if torch.cuda.is_available():
        log(f"  VRAM after model load : {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Training config ───────────────────────────────────────────────────────
    sep("Training Config")
    from slm_training.trainer import TrainingConfig, save_training_config
    train_cfg = TrainingConfig(
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        per_device_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        dtype=args.dtype,
    )
    cfg_path = SLM_TRAINING_ROOT / "configs" / "training_config.yaml"
    save_training_config(train_cfg, cfg_path)
    log(f"  Config saved : {cfg_path}")

    # Rough ETA: assume ~0.5s/step on RTX 3000 Ada
    eta_sec = steps_remaining * 0.5
    eta_h = int(eta_sec // 3600)
    eta_m = int((eta_sec % 3600) // 60)
    log(f"  Rough ETA    : ~{eta_h}h {eta_m}m  (estimate based on 0.5s/step; actual varies)")
    log(f"  Ctrl+C stops training cleanly — resume by re-running this script.")

    # ── Start training ────────────────────────────────────────────────────────
    sep("Training — Starting")
    t_start = time.time()
    from slm_training.trainer import train
    try:
        train(
            model=model,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            ckpt_dir=ckpt_dir,
            device=device,
            tb_writer=tb_writer,
        )
    except KeyboardInterrupt:
        log("Ctrl+C — stopping.")

    if tb_writer:
        tb_writer.close()

    elapsed = time.time() - t_start
    sep(f"Training complete — {elapsed / 3600:.2f}h elapsed")
    final_ckpt = _find_latest_ckpt(ckpt_dir)
    if final_ckpt:
        with open(final_ckpt / "meta.json") as f:
            final_meta = json.load(f)
        log(f"  Final checkpoint : {final_ckpt}")
        log(f"  Final step       : {final_meta['step']:,}")
        log(f"  Final loss       : {final_meta['loss']:.4f}")
    log(f"  Next step: run evaluate.py to compute perplexity and generate Hindi samples.")


if __name__ == "__main__":
    main()
