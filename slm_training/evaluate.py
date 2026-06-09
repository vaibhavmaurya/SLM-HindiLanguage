"""
CLI: full evaluation — val + test perplexity, Hindi generation samples, markdown report.

Usage:
  python evaluate.py                                          # latest checkpoint
  python evaluate.py --ckpt artifacts/checkpoints/step_0005000
  python evaluate.py --max-batches 100                       # faster (100 batches each)
  python evaluate.py --temperature 0.9 --top-p 0.95          # sampling params
  python evaluate.py --report-dir artifacts/reports          # custom report dir
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


def _load_tokenizer(tokenizer_arg: str):
    from transformers import AutoTokenizer
    default_local = (
        SLM_TRAINING_ROOT.parent
        / "tokenizer_training" / "data" / "final" / "hindi_slm_tokenizer_v001"
    )
    candidate = Path(tokenizer_arg)
    if candidate.exists():
        log(f"  Tokenizer : local  {candidate}")
        return AutoTokenizer.from_pretrained(str(candidate))
    if default_local.exists():
        log(f"  Tokenizer : local  {default_local}")
        return AutoTokenizer.from_pretrained(str(default_local))
    log(f"  Tokenizer : HF Hub  {tokenizer_arg}")
    return AutoTokenizer.from_pretrained(tokenizer_arg)


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
    log(f"    params ≈ {model.count_parameters() / 1e6:.2f}M")
    weights = ckpt_path / "model.pt"
    log(f"  Loading weights from {weights} ...")
    import torch as _torch
    state_dict = _torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    log(f"  Model on {device}")
    return model, model_cfg, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Import dataset module early — datasets library must init before PyTorch/CUDA
    from slm_training.dataset import make_dataloader  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Full evaluation: perplexity on val+test, Hindi generation, markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenized-dir", default=str(SLM_TRAINING_ROOT / "data" / "tokenized"),
                        help="Directory with train/val/test Arrow splits.")
    parser.add_argument("--ckpt-dir", default=str(SLM_TRAINING_ROOT / "artifacts" / "checkpoints"),
                        help="Checkpoint directory (used to find latest if --ckpt not given).")
    parser.add_argument("--ckpt", default=None,
                        help="Specific checkpoint folder path. If omitted, uses latest in --ckpt-dir.")
    parser.add_argument("--tokenizer", default="vaibhavmaurya/hindi-slm-tokenizer-v001",
                        help="Tokenizer: local path or HF Hub name.")
    parser.add_argument("--report-dir", default=str(SLM_TRAINING_ROOT / "artifacts" / "reports"),
                        help="Directory to write the evaluation report markdown.")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for perplexity computation.")
    parser.add_argument("--max-batches", type=int, default=200,
                        help="Max batches per split for perplexity (200 ≈ fast but representative).")
    parser.add_argument("--max-new-tokens", type=int, default=80,
                        help="Max tokens to generate per sample.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    tokenized_dir = Path(args.tokenized_dir)
    ckpt_dir = Path(args.ckpt_dir)
    report_dir = Path(args.report_dir)

    sep("Hindi SLM — Evaluate")
    log(f"Tokenized dir  : {tokenized_dir}")
    log(f"Checkpoint dir : {ckpt_dir}")
    log(f"Report dir     : {report_dir}")
    log(f"Tokenizer      : {args.tokenizer}")
    log(f"Batch size     : {args.batch}")
    log(f"Max batches    : {args.max_batches}")
    log(f"Max new tokens : {args.max_new_tokens}")
    log(f"Temperature    : {args.temperature}")
    log(f"Top-p          : {args.top_p}")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    sep("Pre-flight checks")
    for split in ("val", "test"):
        split_path = tokenized_dir / split
        if not split_path.exists():
            log(f"ERROR: '{split}' split not found: {split_path}")
            log(f"       Run tokenize.py first.")
            raise SystemExit(1)
        log(f"  {split:4s} split : {split_path}  OK")

    if not ckpt_dir.exists():
        log(f"ERROR: checkpoint directory does not exist: {ckpt_dir}")
        log(f"       Run train.py first.")
        raise SystemExit(1)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            log(f"ERROR: specified checkpoint not found: {ckpt_path}")
            raise SystemExit(1)
    else:
        ckpt_path = _find_latest_ckpt(ckpt_dir)
        if ckpt_path is None:
            log(f"ERROR: no step_* checkpoints found in {ckpt_dir}")
            log(f"       Run train.py first.")
            raise SystemExit(1)

    log(f"  Checkpoint : {ckpt_path}")
    log("Pre-flight OK")

    # ── Device ────────────────────────────────────────────────────────────────
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"  Device : {torch.cuda.get_device_name(0)}")
    else:
        log("  Device : CPU")

    # ── Load model ────────────────────────────────────────────────────────────
    sep("Loading Model")
    model, model_cfg, meta = _load_model_from_ckpt(ckpt_path, device)
    dtype_str = meta.get("train_config", {}).get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    log(f"  dtype          : {dtype_str}")
    log(f"  Training step  : {meta['step']:,}")
    log(f"  Train loss     : {meta['loss']:.4f}  (last recorded)")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    sep("Loading Tokenizer")
    tokenizer = _load_tokenizer(args.tokenizer)
    log(f"  Vocab size : {tokenizer.vocab_size:,}")
    if tokenizer.vocab_size != 32000:
        log(f"  WARNING: expected 32000, got {tokenizer.vocab_size}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    sep("DataLoaders")
    from slm_training.dataset import make_dataloader
    log("  Building val DataLoader ...")
    val_loader = make_dataloader(tokenized_dir / "val", batch_size=args.batch, shuffle=False)
    log("  Building test DataLoader ...")
    test_loader = make_dataloader(tokenized_dir / "test", batch_size=args.batch, shuffle=False)
    log(f"  Val  batches : {len(val_loader):,}  (using {min(args.max_batches, len(val_loader)):,})")
    log(f"  Test batches : {len(test_loader):,}  (using {min(args.max_batches, len(test_loader)):,})")

    # ── Val perplexity ────────────────────────────────────────────────────────
    sep("Step 1 / 3 — Val Perplexity")
    log("  Computing val perplexity ...")
    t0 = time.time()
    val_ppl = _compute_ppl_verbose(model, val_loader, device, dtype, args.max_batches, label="val")
    log(f"  Val perplexity  : {val_ppl:.2f}  ({time.time()-t0:.1f}s)")

    # ── Test perplexity ───────────────────────────────────────────────────────
    sep("Step 2 / 3 — Test Perplexity")
    log("  Computing test perplexity ...")
    t0 = time.time()
    test_ppl = _compute_ppl_verbose(model, test_loader, device, dtype, args.max_batches, label="test")
    log(f"  Test perplexity : {test_ppl:.2f}  ({time.time()-t0:.1f}s)")

    # ── Hindi generation samples ──────────────────────────────────────────────
    sep("Step 3 / 3 — Hindi Generation Samples")
    from slm_training.evaluator import generate_samples, print_samples
    log(f"  Generating {5} samples (max_new_tokens={args.max_new_tokens}, "
        f"temp={args.temperature}, top_p={args.top_p}) ...")

    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    log(f"  Generation done. Printing samples:")
    print("", flush=True)
    print_samples(samples)

    # ── Quality signals ───────────────────────────────────────────────────────
    sep("Quality Signals")
    avg_dev_ratio = sum(s["devanagari_ratio"] for s in samples) / len(samples)
    avg_unk = sum(s["unk_count"] for s in samples) / len(samples)
    log(f"  Val perplexity          : {val_ppl:.2f}")
    log(f"  Test perplexity         : {test_ppl:.2f}")
    log(f"  Avg Devanagari ratio    : {avg_dev_ratio:.3f}  (good >= 0.7)")
    log(f"  Avg UNK tokens/sample   : {avg_unk:.1f}  (good = 0)")
    if val_ppl < 50:
        log("  Perplexity assessment   : EXCELLENT  (< 50)")
    elif val_ppl < 100:
        log("  Perplexity assessment   : GOOD  (50–100)")
    elif val_ppl < 200:
        log("  Perplexity assessment   : FAIR  (100–200, keep training)")
    else:
        log("  Perplexity assessment   : POOR  (> 200, model needs more training)")

    # ── Write report ──────────────────────────────────────────────────────────
    sep("Writing Report")
    from slm_training.evaluator import write_evaluation_report
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"evaluation_step_{meta['step']:07d}.md"
    report_path = report_dir / report_name
    write_evaluation_report(
        report_path=report_path,
        val_perplexity=val_ppl,
        test_perplexity=test_ppl,
        samples=samples,
        step=meta["step"],
    )
    log(f"  Report written : {report_path}")

    sep(f"Evaluation complete — step {meta['step']:,}")
    log(f"  Val perplexity  : {val_ppl:.2f}")
    log(f"  Test perplexity : {test_ppl:.2f}")
    log(f"  Report          : {report_path}")


@torch.no_grad()
def _compute_ppl_verbose(model, loader, device, dtype, max_batches, label=""):
    import torch
    import math
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            _, loss = model(input_ids, labels=input_ids.clone())
        total_loss += loss.item()
        n += 1
        if n % 25 == 0:
            running_ppl = math.exp(total_loss / n)
            print(f"[evaluate]   {label} batch {n} / {max_batches} — ppl={running_ppl:.2f}", flush=True)
    model.train()
    avg = total_loss / max(n, 1)
    return math.exp(avg)


import torch  # needed for the decorator above


if __name__ == "__main__":
    main()
