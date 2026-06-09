import os
import json
import gzip
import argparse
from pathlib import Path
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError(
        "Missing Hugging Face token. Please create a .env file with HF_TOKEN=hf_your_token_here"
    )


GIB = 1024 ** 3
MIB = 1024 ** 2


POSSIBLE_TEXT_COLUMNS = [
    "text",
    "content",
    "sentence",
    "document",
    "data",
]


def extract_text(example: dict) -> str:
    """
    Extract text from a Hugging Face dataset row.

    Sangraha is text-generation corpus data, but column names can vary.
    This function first checks common text column names.
    If none are found, it falls back to joining all string fields.
    """

    for column in POSSIBLE_TEXT_COLUMNS:
        value = example.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()

    string_values = []

    for key, value in example.items():
        if isinstance(value, str) and value.strip():
            string_values.append(value.strip())

    return "\n".join(string_values).strip()


def open_gzip_writer(output_dir: Path, chunk_index: int):
    file_path = output_dir / f"sangraha_verified_hin_part_{chunk_index:05d}.jsonl.gz"
    writer = gzip.open(file_path, "wb")
    return file_path, writer


def write_manifest(output_dir: Path, manifest: dict):
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Stream and download a limited-size subset of ai4bharat/sangraha."
    )

    parser.add_argument(
        "--dataset",
        default="ai4bharat/sangraha",
        help="Hugging Face dataset name.",
    )

    parser.add_argument(
        "--data-dir",
        default="verified/hin",
        help="Subset and language path. Example: verified/hin",
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream. Usually train.",
    )

    parser.add_argument(
        "--out-dir",
        default="./data/sangraha_verified_hin_10gb",
        help="Output directory for local chunks.",
    )

    parser.add_argument(
        "--max-gib",
        type=float,
        default=10.0,
        help="Maximum uncompressed UTF-8 text size to write, in GiB.",
    )

    parser.add_argument(
        "--chunk-mib",
        type=float,
        default=512.0,
        help="Approximate uncompressed size per output chunk, in MiB.",
    )

    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=20,
        help="Skip records where extracted text is shorter than this.",
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="Optional streaming shuffle buffer. Use 0 to disable.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for streaming shuffle.",
    )

    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing Hugging Face token.",
    )

    args = parser.parse_args()

    hf_token = os.environ.get(args.token_env)

    if not hf_token:
        raise RuntimeError(
            f"Missing Hugging Face token. Please set {args.token_env} environment variable."
        )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_gib * GIB)
    chunk_bytes = int(args.chunk_mib * MIB)

    print("Starting Sangraha streaming download")
    print(f"Dataset       : {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split         : {args.split}")
    print(f"Output dir    : {output_dir}")
    print(f"Max size      : {args.max_gib} GiB uncompressed text")
    print(f"Chunk size    : {args.chunk_mib} MiB uncompressed text")

    dataset = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        streaming=True,
        token=hf_token,
    )

    if args.shuffle_buffer and args.shuffle_buffer > 0:
        dataset = dataset.shuffle(
            seed=args.seed,
            buffer_size=args.shuffle_buffer,
        )

    chunk_index = 0
    total_rows_seen = 0
    total_rows_written = 0
    total_text_bytes = 0
    current_chunk_bytes = 0
    chunks = []

    current_file_path, writer = open_gzip_writer(output_dir, chunk_index)

    try:
        progress = tqdm(unit="rows")

        for example in dataset:
            total_rows_seen += 1

            text = extract_text(example)

            if len(text) < args.min_text_chars:
                progress.update(1)
                continue

            record = {
                "text": text
            }

            line = json.dumps(record, ensure_ascii=False) + "\n"
            line_bytes = line.encode("utf-8")
            line_size = len(line_bytes)

            if total_text_bytes + line_size > max_bytes:
                break

            if current_chunk_bytes + line_size > chunk_bytes and current_chunk_bytes > 0:
                writer.close()

                chunks.append(
                    {
                        "file": str(current_file_path),
                        "uncompressed_text_bytes": current_chunk_bytes,
                    }
                )

                chunk_index += 1
                current_file_path, writer = open_gzip_writer(output_dir, chunk_index)
                current_chunk_bytes = 0

            writer.write(line_bytes)

            total_rows_written += 1
            total_text_bytes += line_size
            current_chunk_bytes += line_size

            progress.set_postfix(
                {
                    "written_gib": round(total_text_bytes / GIB, 3),
                    "rows_written": total_rows_written,
                    "chunk": chunk_index,
                }
            )

            progress.update(1)

        progress.close()

    finally:
        writer.close()

    if current_chunk_bytes > 0:
        chunks.append(
            {
                "file": str(current_file_path),
                "uncompressed_text_bytes": current_chunk_bytes,
            }
        )

    manifest = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "split": args.split,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "max_requested_gib": args.max_gib,
        "total_rows_seen": total_rows_seen,
        "total_rows_written": total_rows_written,
        "total_uncompressed_text_bytes": total_text_bytes,
        "total_uncompressed_text_gib": round(total_text_bytes / GIB, 4),
        "output_format": "jsonl.gz",
        "chunks": chunks,
    }

    write_manifest(output_dir, manifest)

    print("\nDownload complete")
    print(f"Rows seen                : {total_rows_seen}")
    print(f"Rows written             : {total_rows_written}")
    print(f"Uncompressed text size   : {total_text_bytes / GIB:.4f} GiB")
    print(f"Number of chunks         : {len(chunks)}")
    print(f"Manifest                 : {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()