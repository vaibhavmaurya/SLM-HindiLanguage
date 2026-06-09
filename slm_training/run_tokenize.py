"""
CLI: tokenize each Sangraha .jsonl.gz file independently into its own Arrow part.

Each input file produces:
    data/tokenized/train/part_NNNN/
    data/tokenized/val/part_NNNN/
    data/tokenized/test/part_NNNN/

Already-completed parts are skipped on re-run (incremental).
Use --force to redo all parts from scratch.

Usage:
  python run_tokenize.py                         # all files
  python run_tokenize.py --max-files 2           # 2 files only  (quick test)
  python run_tokenize.py --out-dir data/tokenized_test --max-files 2
  python run_tokenize.py --force                 # re-tokenize everything
"""

import sys
import argparse
import json
import shutil
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
        log(f"Tokenizer    : local  {candidate}")
        return AutoTokenizer.from_pretrained(str(candidate))
    if default_local.exists():
        log(f"Tokenizer    : local  {default_local}")
        return AutoTokenizer.from_pretrained(str(default_local))

    log(f"Tokenizer    : HF Hub  {tokenizer_arg}")
    return AutoTokenizer.from_pretrained(tokenizer_arg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize Sangraha Hindi files into per-file Arrow parts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=str(SLM_TRAINING_ROOT / "data" / "sangraha_verified_hin_10gb"),
        help="Directory with sangraha_verified_hin_part_*.jsonl.gz files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SLM_TRAINING_ROOT / "data" / "tokenized"),
        help="Output dir. Each file creates train/part_NNNN/, val/part_NNNN/, test/part_NNNN/.",
    )
    parser.add_argument(
        "--manifest",
        default=str(SLM_TRAINING_ROOT / "configs" / "tokenized_dataset_manifest.json"),
        help="Manifest JSON path.",
    )
    parser.add_argument(
        "--tokenizer",
        default="vaibhavmaurya/hindi-slm-tokenizer-v001",
        help="Tokenizer: local dir path or HuggingFace Hub name.",
    )
    parser.add_argument(
        "--seq-len", type=int, default=512,
        help="Pack sequences to exactly this many tokens.",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Max .jsonl.gz files to process (None = all).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing parts and re-tokenize all files from scratch.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    t_start = time.time()

    sep("Hindi SLM — Tokenize + Pack (per-file)")
    log(f"Data dir     : {data_dir}")
    log(f"Output dir   : {out_dir}")
    log(f"Manifest     : {manifest_path}")
    log(f"Seq length   : {args.seq_len}")
    log(f"Max files    : {args.max_files if args.max_files is not None else 'all'}")
    log(f"Seed         : {args.seed}")
    log(f"Force redo   : {args.force}")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    sep("Pre-flight checks")
    if not data_dir.exists():
        log(f"ERROR: data directory does not exist: {data_dir}")
        log(f"       Run download_sangraha_10gb.py first, or pass --data-dir <path>")
        raise SystemExit(1)

    gz_files = sorted(data_dir.glob("*.jsonl.gz"))
    if not gz_files:
        log(f"ERROR: no .jsonl.gz files found in {data_dir}")
        raise SystemExit(1)

    files_to_process = gz_files if args.max_files is None else gz_files[:args.max_files]
    total_compressed_mb = sum(f.stat().st_size for f in files_to_process) / 1e6

    log(f"Found {len(gz_files)} .jsonl.gz file(s) in data dir")
    log(f"Will process : {len(files_to_process)} file(s)  ({total_compressed_mb:.0f} MB compressed)")
    for gf in files_to_process:
        log(f"  {gf.name}  ({gf.stat().st_size / 1e6:.0f} MB)")
    log(f"Est. uncompressed : ~{total_compressed_mb * 5:.0f}–{total_compressed_mb * 8:.0f} MB")
    log("Pre-flight OK")

    # ── Force: wipe existing parts ────────────────────────────────────────────
    if args.force:
        sep("Force — Clearing existing tokenized parts")
        for split in ("train", "val", "test"):
            split_dir = out_dir / split
            if split_dir.exists():
                shutil.rmtree(str(split_dir))
                log(f"  Deleted {split_dir}")
        if manifest_path.exists():
            manifest_path.unlink()
            log(f"  Deleted {manifest_path}")

    # Import dataset module early — datasets library must init before SentencePiece tokenizer
    from slm_training.dataset import build_tokenized_splits

    # ── Load tokenizer ────────────────────────────────────────────────────────
    sep("Load Tokenizer")
    tokenizer = _load_tokenizer(args.tokenizer)
    log(f"Vocab size   : {tokenizer.vocab_size:,}")
    if tokenizer.vocab_size != 32000:
        log(f"WARNING: expected vocab_size=32000, got {tokenizer.vocab_size}")

    # ── Tokenize + Pack (per file) ────────────────────────────────────────────
    sep("Tokenize + Pack — per file")

    manifest = build_tokenized_splits(
        gz_files=files_to_process,
        tokenized_dir=out_dir,
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    sep(f"DONE — {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    log(f"Files processed : {len(files_to_process)}")
    log(f"Train parts     : {len(files_to_process)} parts in {out_dir}/train/")
    log(f"Train sequences : {manifest['splits']['train']:,}")
    log(f"Val sequences   : {manifest['splits']['val']:,}")
    log(f"Test sequences  : {manifest['splits']['test']:,}")
    log(f"Total sequences : {manifest['total_sequences']:,}")
    log(f"Manifest SHA    : {manifest.get('train_first_shard_sha256', 'n/a')[:16]}...")
    log(f"Output          : {out_dir}")
    log(f"Manifest        : {manifest_path}")


if __name__ == "__main__":
    main()
