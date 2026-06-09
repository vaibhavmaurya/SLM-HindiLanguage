"""Stages 3 + 5: Data loading, cleaning, tokenization, and packing.

Each .jsonl.gz source file is tokenized independently and saved as its own
Arrow part under train/part_NNNN/, val/part_NNNN/, test/part_NNNN/.

Pipeline per file (single streaming pass, no datasets.map):
  gzip stream -> clean -> tokenize -> pack into seq_len windows -> save Arrow.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path
from typing import Optional

# Import datasets at module level so it initialises before SentencePiece is loaded.
# On Windows/WSL (datasets 4.8.x) importing datasets after a SentencePiece tokenizer
# causes a segfault; keeping this import here makes the order deterministic.
import datasets as _datasets_preload  # noqa: F401


# ---------- Constants ----------

MIN_CHARS = 50
MAX_CHARS = 50_000
MIN_DEVANAGARI_RATIO = 0.60

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

# Flush packing buffer every this many tokens (bounds RAM per file)
_TOKEN_FLUSH = 512 * 10_000   # ~10k sequences worth


# ---------- Text cleaning ----------

def _devanagari_ratio(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    dev = sum(1 for ch in text if DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END)
    return dev / total


def _clean_text(text: str) -> Optional[str]:
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    if _devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
        return None
    return text


# ---------- Per-file tokenization (single streaming pass) ----------

def _tokenize_and_pack_file(gz_file: Path, tokenizer, seq_len: int) -> "Dataset":
    """Stream one .jsonl.gz, clean+tokenize+pack in one pass.

    No intermediate Dataset objects or datasets.map calls — avoids segfaults
    from datasets 4.8.x workers on Windows/WSL.
    Returns Dataset of {input_ids: [int × seq_len]}.
    """
    from datasets import Dataset

    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")

    all_seqs: list[list[int]] = []
    token_buf: list[int] = []
    n_raw = 0
    n_kept = 0

    print(f"[tokenize] {gz_file.name}: streaming ...", flush=True)

    with gzip.open(gz_file, "rt", encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if not text:
                continue
            n_raw += 1

            cleaned = _clean_text(text)
            if cleaned is None:
                continue
            n_kept += 1

            ids = tokenizer.encode(cleaned, add_special_tokens=False)
            ids.append(eos_id)
            token_buf.extend(ids)

            # Periodic flush: pack full windows, keep the leftover tail
            if len(token_buf) >= _TOKEN_FLUSH:
                n_seqs = len(token_buf) // seq_len
                all_seqs.extend(
                    token_buf[i * seq_len:(i + 1) * seq_len] for i in range(n_seqs)
                )
                token_buf = token_buf[n_seqs * seq_len:]
                print(
                    f"[tokenize]   {n_raw:,} raw  {n_kept:,} kept  {len(all_seqs):,} seqs",
                    flush=True,
                )

    # Final flush
    n_seqs = len(token_buf) // seq_len
    all_seqs.extend(
        token_buf[i * seq_len:(i + 1) * seq_len] for i in range(n_seqs)
    )

    print(
        f"[tokenize] {gz_file.name}: done — "
        f"{n_raw:,} raw rows, {n_kept:,} kept, {len(all_seqs):,} packed sequences",
        flush=True,
    )

    return Dataset.from_dict({"input_ids": all_seqs})


# ---------- Build splits across all files ----------

def build_tokenized_splits(
    gz_files: list[Path],
    tokenized_dir: Path,
    manifest_path: Path,
    tokenizer,
    seq_len: int = 512,
    seed: int = 42,
) -> dict:
    """Tokenize each .jsonl.gz independently, split 98/1/1, save as part_NNNN/.

    Output structure:
        tokenized_dir/train/part_0000/   (Arrow dataset dir)
        tokenized_dir/train/part_0001/
        tokenized_dir/val/part_0000/
        tokenized_dir/val/part_0001/
        tokenized_dir/test/part_0000/
        tokenized_dir/test/part_0001/

    Already-completed parts (train/part_NNNN/ exists) are skipped — incremental.
    Returns a manifest dict with aggregate counts.
    """
    from datasets import load_from_disk

    tokenized_dir.mkdir(parents=True, exist_ok=True)

    total_train = total_val = total_test = 0

    for fi, gz_file in enumerate(gz_files):
        part_name = f"part_{fi:04d}"
        train_part = tokenized_dir / "train" / part_name

        if train_part.exists():
            existing = load_from_disk(str(train_part))
            n = len(existing)
            print(
                f"[tokenize] {part_name} already exists ({n:,} train seqs) — skipping",
                flush=True,
            )
            total_train += n
            for split in ("val", "test"):
                p = tokenized_dir / split / part_name
                if p.exists():
                    n_split = len(load_from_disk(str(p)))
                    if split == "val":
                        total_val += n_split
                    else:
                        total_test += n_split
            continue

        print(f"\n[tokenize] === File {fi+1}/{len(gz_files)}: {gz_file.name} ===", flush=True)

        packed = _tokenize_and_pack_file(gz_file, tokenizer, seq_len)

        if len(packed) == 0:
            print(f"[tokenize] {gz_file.name}: no sequences produced, skipping", flush=True)
            continue

        # 98 / 1 / 1 split; guarantee at least 4 rows in the val+test pool
        n_total = len(packed)
        val_test_count = max(4, int(n_total * 0.02))
        val_test_frac = val_test_count / n_total

        split1 = packed.train_test_split(test_size=val_test_frac, seed=seed)
        split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)

        train_ds = split1["train"]
        val_ds = split2["train"]
        test_ds = split2["test"]

        print(
            f"[tokenize] {part_name} split — "
            f"train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}",
            flush=True,
        )

        train_ds.save_to_disk(str(tokenized_dir / "train" / part_name), num_shards=1)
        val_ds.save_to_disk(str(tokenized_dir / "val" / part_name), num_shards=1)
        test_ds.save_to_disk(str(tokenized_dir / "test" / part_name), num_shards=1)
        print(f"[tokenize] {part_name} saved.", flush=True)

        total_train += len(train_ds)
        total_val += len(val_ds)
        total_test += len(test_ds)

    manifest = _build_manifest(total_train, total_val, total_test, seq_len, tokenized_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[tokenize] Manifest written to {manifest_path}", flush=True)

    return manifest


def _build_manifest(train_n, val_n, test_n, seq_len, tokenized_dir) -> dict:
    import datetime

    first_arrow = next((tokenized_dir / "train").glob("part_*/data-*.arrow"), None)
    sha256 = "n/a"
    if first_arrow and first_arrow.exists():
        h = hashlib.sha256()
        with open(first_arrow, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    return {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "seq_len": seq_len,
        "total_sequences": train_n + val_n + test_n,
        "splits": {
            "train": train_n,
            "val": val_n,
            "test": test_n,
        },
        "train_first_shard_sha256": sha256,
        "schema": "input_ids: List[int] of length seq_len",
    }


# ---------- DataLoader helper ----------

def make_dataloader(dataset_path: Path, batch_size: int = 2, shuffle: bool = True):
    """Return a PyTorch DataLoader for a tokenized split directory.

    Supports both the new per-file structure (part_NNNN/ subdirs) and the old
    single-dataset structure for backwards compatibility.
    """
    import torch
    from datasets import concatenate_datasets, load_from_disk
    from torch.utils.data import DataLoader

    split_dir = Path(dataset_path)
    part_dirs = sorted(split_dir.glob("part_*/"))

    if part_dirs:
        datasets = [load_from_disk(str(p)) for p in part_dirs]
        ds = concatenate_datasets(datasets)
        print(
            f"[dataloader] {split_dir.name}: {len(part_dirs)} part(s) -> {len(ds):,} sequences",
            flush=True,
        )
    else:
        ds = load_from_disk(str(split_dir))
        print(
            f"[dataloader] {split_dir.name}: {len(ds):,} sequences",
            flush=True,
        )

    ds.set_format("torch", columns=["input_ids"])

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
