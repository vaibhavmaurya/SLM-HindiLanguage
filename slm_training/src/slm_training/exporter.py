"""Stage 9: Export — HuggingFace format, GGUF (F16 + Q4_K_M), Raspberry Pi validation."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .architecture import ModelConfig
from .model import HindiSLM


# ---------- HuggingFace export ----------

def export_to_hf(
    model: HindiSLM,
    tokenizer,
    model_cfg: ModelConfig,
    output_dir: Path,
) -> Path:
    """Save model + tokenizer in HuggingFace format."""
    import torch
    from transformers import AutoConfig, GPT2Config, PreTrainedModel

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw weights as PyTorch bin (HF-compatible naming)
    weights_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), weights_path)

    # Save model config as config.json
    config_dict = {
        "architectures": ["HindiSLM"],
        "model_type": "hindi_slm",
        **asdict(model_cfg),
        "torch_dtype": "bfloat16",
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Save generation config
    gen_config = {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))

    print(f"[export] HuggingFace model saved to {output_dir}")
    print(f"  Files: {[p.name for p in output_dir.iterdir()]}")
    return output_dir


# ---------- GGUF conversion via llama.cpp ----------

def _find_llama_convert_script() -> Optional[Path]:
    """Locate convert_hf_to_gguf.py from llama.cpp installation."""
    candidates = [
        Path("llama.cpp") / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        candidates.insert(0, pkg_dir / "convert_hf_to_gguf.py")
    except ImportError:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def convert_to_gguf_f16(hf_model_dir: Path, gguf_output_dir: Path) -> Optional[Path]:
    """Convert HF model directory to GGUF F16 using llama.cpp converter."""
    gguf_output_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_output_dir / "hindi_slm_v001_f16.gguf"

    convert_script = _find_llama_convert_script()
    if convert_script is None:
        print("[export] WARNING: llama.cpp convert script not found.")
        print("  Install llama.cpp and set LLAMA_CPP_PATH or add to PATH.")
        print("  Skipping GGUF conversion.")
        return None

    cmd = [
        sys.executable,
        str(convert_script),
        str(hf_model_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16",
    ]
    print(f"[export] Converting to GGUF F16 ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[export] GGUF conversion failed:\n{result.stderr}")
        return None
    print(f"[export] GGUF F16 saved to {gguf_path}")
    return gguf_path


def quantize_gguf_q4km(gguf_f16_path: Path, gguf_output_dir: Path) -> Optional[Path]:
    """Quantize GGUF F16 to Q4_K_M using llama-quantize."""
    if gguf_f16_path is None or not gguf_f16_path.exists():
        print("[export] Skipping quantization — F16 GGUF not found.")
        return None

    q4_path = gguf_output_dir / "hindi_slm_v001_q4_k_m.gguf"

    # Try llama-quantize from PATH
    quantize_bin = "llama-quantize"
    cmd = [quantize_bin, str(gguf_f16_path), str(q4_path), "Q4_K_M"]
    print("[export] Quantizing to Q4_K_M ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[export] Quantization failed (is llama-quantize in PATH?):\n{result.stderr}")
        return None
    print(f"[export] Q4_K_M GGUF saved to {q4_path}")
    return q4_path


# ---------- Raspberry Pi compatibility validation ----------

def validate_raspberry_pi_compatibility(
    hf_model_dir: Path,
    model_cfg: ModelConfig,
    gguf_q4_path: Optional[Path] = None,
) -> dict:
    """Estimate memory footprint and validate Pi 8 GB compatibility."""
    PI_VRAM_GB = 8.0
    PI_RAM_SAFETY_FACTOR = 0.75  # leave 25% for OS + overhead

    # Q4_K_M: ~4.5 bits per weight on average
    from .architecture import ParameterCounter, KVCacheEstimator
    counts = ParameterCounter.count(model_cfg)
    params = counts["total"]

    model_q4_gb = params * 4.5 / 8 / 1e9
    kv_cache_gb = KVCacheEstimator.estimate_inference_gb(model_cfg)
    total_gb = model_q4_gb + kv_cache_gb
    fits_pi = total_gb < (PI_VRAM_GB * PI_RAM_SAFETY_FACTOR)

    # Disk size of GGUF
    gguf_size_mb = gguf_q4_path.stat().st_size / 1e6 if (gguf_q4_path and gguf_q4_path.exists()) else None

    report = {
        "parameters": params,
        "parameters_M": round(params / 1e6, 2),
        "model_q4_size_gb": round(model_q4_gb, 3),
        "kv_cache_gb_at_ctx512": round(kv_cache_gb, 4),
        "total_estimated_gb": round(total_gb, 3),
        "pi_vram_budget_gb": PI_VRAM_GB,
        "fits_raspberry_pi_8gb": fits_pi,
        "gguf_disk_size_mb": round(gguf_size_mb, 1) if gguf_size_mb else "N/A",
        "recommended_context": model_cfg.max_seq_len,
    }

    print("\n[export] Raspberry Pi 8 GB Compatibility Report")
    print(f"  Model (Q4_K_M): {report['model_q4_size_gb']:.3f} GB")
    print(f"  KV cache:       {report['kv_cache_gb_at_ctx512']:.4f} GB")
    print(f"  Total:          {report['total_estimated_gb']:.3f} GB / {PI_VRAM_GB} GB budget")
    status = "PASS" if fits_pi else "FAIL"
    print(f"  Status:         {status}")

    return report


def write_export_report(report_path: Path, hf_dir: Path, gguf_f16: Optional[Path], gguf_q4: Optional[Path], pi_report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Export Report",
        "",
        "## HuggingFace Model",
        f"- Directory: `{hf_dir}`",
        f"- Files: {[p.name for p in hf_dir.iterdir() if hf_dir.exists()]}",
        "",
        "## GGUF Exports",
        f"- F16: `{gguf_f16}`" if gguf_f16 else "- F16: not generated",
        f"- Q4_K_M: `{gguf_q4}`" if gguf_q4 else "- Q4_K_M: not generated",
        "",
        "## Raspberry Pi 8 GB Compatibility",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in pi_report.items():
        lines.append(f"| {k} | {v} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[export] Report written to {report_path}")
