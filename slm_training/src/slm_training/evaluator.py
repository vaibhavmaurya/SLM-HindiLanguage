"""Stage 8: Evaluation — perplexity, Hindi generation samples, script quality."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Optional

import torch

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

HINDI_PROMPTS = [
    "आज का समाचार यह है कि",
    "एक बार की बात है, जंगल में",
    "नमस्ते! आप कैसे हैं? मैं",
    "भारत की राजधानी नई दिल्ली है। यहाँ",
    "चाँद की रोशनी में, नदी किनारे",
]


def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    dev = sum(1 for ch in text if DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END)
    return dev / len(text)


@torch.no_grad()
def compute_perplexity(model, loader, device: torch.device, dtype: torch.dtype, max_batches: int = 200) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        device_type = input_ids.device.type
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            _, loss = model(input_ids, labels=labels)
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / max(n, 1)
    return math.exp(avg_loss)


@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    device: torch.device,
    prompts: Optional[list] = None,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> list[dict]:
    if prompts is None:
        prompts = HINDI_PROMPTS
    model.eval()
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        unk_count = output_ids[0].tolist().count(tokenizer.unk_token_id or 0)
        dev_ratio = _devanagari_ratio(generated)
        results.append({
            "prompt": prompt,
            "generated": generated,
            "total_tokens": output_ids.shape[1],
            "unk_count": unk_count,
            "devanagari_ratio": round(dev_ratio, 3),
        })
    return results


def print_samples(samples: list[dict]) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        for s in samples:
            console.print(Panel(
                f"[bold yellow]{s['prompt']}[/bold yellow][white]{s['generated']}[/white]\n"
                f"[dim]tokens={s['total_tokens']}  unk={s['unk_count']}  dev_ratio={s['devanagari_ratio']:.2f}[/dim]",
                title="Generation Sample",
                border_style="green",
            ))
    except ImportError:
        for s in samples:
            print(f"\nPrompt : {s['prompt']}")
            print(f"Output : {s['generated']}")
            print(f"Metrics: tokens={s['total_tokens']}  unk={s['unk_count']}  dev_ratio={s['devanagari_ratio']:.2f}")


def write_evaluation_report(
    report_path: Path,
    val_perplexity: float,
    test_perplexity: float,
    samples: list[dict],
    step: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    avg_dev_ratio = sum(s["devanagari_ratio"] for s in samples) / max(len(samples), 1)
    avg_unk = sum(s["unk_count"] for s in samples) / max(len(samples), 1)

    lines = [
        "# Evaluation Report",
        "",
        f"**Checkpoint step:** {step}",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Validation Perplexity | {val_perplexity:.2f} |",
        f"| Test Perplexity | {test_perplexity:.2f} |",
        f"| Avg Devanagari Ratio (generated) | {avg_dev_ratio:.3f} |",
        f"| Avg UNK tokens per sample | {avg_unk:.1f} |",
        "",
        "## Generation Samples",
        "",
    ]
    for i, s in enumerate(samples, 1):
        lines += [
            f"### Sample {i}",
            f"**Prompt:** {s['prompt']}",
            f"",
            f"**Generated:** {s['generated']}",
            f"",
            f"_tokens={s['total_tokens']}  unk={s['unk_count']}  dev_ratio={s['devanagari_ratio']:.3f}_",
            "",
        ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[eval] Report written to {report_path}")
