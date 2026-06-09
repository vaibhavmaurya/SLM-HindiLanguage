"""
CLI: compute validation or test loss on a saved checkpoint.

Lightweight check — runs evaluate_loss() only, no generation.
Use this to quickly inspect a checkpoint mid-training or compare checkpoints.

Usage:
  python validate.py                                      # latest checkpoint, val split
  python validate.py --split test                         # test split instead
  python validate.py --ckpt artifacts/checkpoints/step_0005000   # specific checkpoint
  python validate.py --max-batches 50                    # fast check (50 batches only)
  python validate.py --all-checkpoints                   # run on every checkpoint
"""

import sys
import argparse
import json
import math
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


def _load_model_from_ckpt(ckpt_path: Path, device):
    import torch
    from slm_training.architecture import ModelConfig
    from slm_training.model import HindiSLM

    meta_file = ckpt_path / "meta.json"
    if not meta_file.exists():
        log(f"ERROR: meta.json not found in {ckpt_path}")
        raise SystemExit(1)

    with open(meta_file) as f:
        meta = json.load(f)

    cfg_dict = meta["model_config"]
    model_cfg = ModelConfig(**cfg_dict)
    log(f"  Model config from checkpoint:")
    log(f"    hidden={model_cfg.hidden_size}, layers={model_cfg.num_layers}, "
        f"heads={model_cfg.num_attention_heads}, seq_len={model_cfg.max_seq_len}")

    model = HindiSLM(model_cfg)
    weights = ckpt_path / "model.pt"
    log(f"  Loading weights from {weights} ...")
    state_dict = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    log(f"  Model loaded and moved to {device}")
    return model, model_cfg, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Import dataset module early — datasets library must init before PyTorch/CUDA
    from slm_training.dataset import make_dataloader  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Compute val/test loss on a saved checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenized-dir", default=str(SLM_TRAINING_ROOT / "data" / "tokenized"),
                        help="Directory with train/val/test Arrow splits.")
    parser.add_argument("--ckpt-dir", default=str(SLM_TRAINING_ROOT / "artifacts" / "checkpoints"),
                        help="Checkpoint directory to scan for step_* folders.")
    parser.add_argument("--ckpt", default=None,
                        help="Path to a specific checkpoint folder (e.g. artifacts/checkpoints/step_0005000). "
                             "If omitted, uses the latest checkpoint in --ckpt-dir.")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Which split to evaluate.")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for evaluation.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Max batches to evaluate (None = full split). Use 50 for a quick check.")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Run evaluation on every checkpoint in --ckpt-dir and print a comparison table.")
    args = parser.parse_args()

    tokenized_dir = Path(args.tokenized_dir)
    ckpt_dir = Path(args.ckpt_dir)

    sep("Hindi SLM — Validate")
    log(f"Tokenized dir  : {tokenized_dir}")
    log(f"Checkpoint dir : {ckpt_dir}")
    log(f"Split          : {args.split}")
    log(f"Batch size     : {args.batch}")
    log(f"Max batches    : {args.max_batches if args.max_batches else 'full split'}")
    log(f"All checkpoints: {args.all_checkpoints}")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    sep("Pre-flight checks")
    split_path = tokenized_dir / args.split
    if not split_path.exists():
        log(f"ERROR: '{args.split}' split not found: {split_path}")
        log(f"       Run tokenize.py first.")
        raise SystemExit(1)
    log(f"  {args.split} split : {split_path}  OK")

    if not ckpt_dir.exists():
        log(f"ERROR: checkpoint directory does not exist: {ckpt_dir}")
        log(f"       Run train.py first.")
        raise SystemExit(1)

    all_ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not all_ckpts:
        log(f"ERROR: no step_* checkpoints found in {ckpt_dir}")
        log(f"       Run train.py first.")
        raise SystemExit(1)
    log(f"  Found {len(all_ckpts)} checkpoint(s) in {ckpt_dir}")
    for ck in all_ckpts:
        log(f"    {ck.name}")

    # ── Device ────────────────────────────────────────────────────────────────
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"  Device : {torch.cuda.get_device_name(0)}")
    else:
        log("  Device : CPU")

    # ── Decide which checkpoints to run ───────────────────────────────────────
    if args.all_checkpoints:
        ckpts_to_run = all_ckpts
    elif args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            log(f"ERROR: specified checkpoint not found: {ckpt_path}")
            raise SystemExit(1)
        ckpts_to_run = [ckpt_path]
    else:
        ckpts_to_run = [all_ckpts[-1]]

    log(f"  Will evaluate {len(ckpts_to_run)} checkpoint(s)")

    # ── DataLoader ────────────────────────────────────────────────────────────
    sep(f"DataLoader — {args.split} split")
    from slm_training.dataset import make_dataloader
    loader = make_dataloader(
        split_path,
        batch_size=args.batch,
        shuffle=False,
    )
    total_batches = len(loader)
    batches_used = min(args.max_batches, total_batches) if args.max_batches else total_batches
    log(f"  Total batches in split : {total_batches:,}")
    log(f"  Batches to evaluate    : {batches_used:,}")

    # ── Evaluate each checkpoint ──────────────────────────────────────────────
    results = []
    for ck_idx, ckpt_path in enumerate(ckpts_to_run):
        sep(f"Checkpoint [{ck_idx+1}/{len(ckpts_to_run)}] — {ckpt_path.name}")
        t0 = time.time()
        model, model_cfg, meta = _load_model_from_ckpt(ckpt_path, device)
        dtype_str = meta.get("train_config", {}).get("dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        log(f"  Computing {args.split} loss ...")
        from slm_training.trainer import evaluate_loss
        loss = _evaluate_with_limit(model, loader, device, dtype, args.max_batches)
        ppl = math.exp(loss)
        elapsed = time.time() - t0

        log(f"  Step         : {meta['step']:,}")
        log(f"  Train loss   : {meta['loss']:.4f}  (last logged during training)")
        log(f"  {args.split:4s} loss    : {loss:.4f}")
        log(f"  Perplexity   : {ppl:.2f}")
        log(f"  Elapsed      : {elapsed:.1f}s")
        results.append({"ckpt": ckpt_path.name, "step": meta["step"], "loss": loss, "ppl": ppl})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(results) > 1:
        sep("Comparison — All Checkpoints")
        log(f"  {'Checkpoint':<25}  {'Step':>7}  {args.split+' loss':>10}  {'Perplexity':>12}")
        log(f"  {'─'*25}  {'─'*7}  {'─'*10}  {'─'*12}")
        best = min(results, key=lambda r: r["loss"])
        for r in results:
            marker = " <-- best" if r["ckpt"] == best["ckpt"] else ""
            log(f"  {r['ckpt']:<25}  {r['step']:>7,}  {r['loss']:>10.4f}  {r['ppl']:>12.2f}{marker}")
    else:
        sep("Result")
        r = results[0]
        log(f"  Checkpoint : {r['ckpt']}")
        log(f"  Step       : {r['step']:,}")
        log(f"  {args.split:4s} loss  : {r['loss']:.4f}")
        log(f"  Perplexity : {r['ppl']:.2f}")
        log(f"")
        log(f"  Next step: run evaluate.py for full report with Hindi generation samples.")


@torch.no_grad()
def _evaluate_with_limit(model, loader, device, dtype, max_batches):
    import torch
    import math
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            _, loss = model(input_ids, labels=input_ids.clone())
        total_loss += loss.item()
        n += 1
        if n % 50 == 0:
            running_ppl = math.exp(total_loss / n)
            print(f"[validate]   batch {n:,} / {max_batches or '?'} — running ppl={running_ppl:.2f}", flush=True)
    model.train()
    return total_loss / max(n, 1)


import torch  # needed for the decorator above


if __name__ == "__main__":
    main()
