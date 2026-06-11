"""
Export the trained Hindi SLM checkpoint to:
  1. HuggingFace LlamaForCausalLM format  (artifacts/models/hindi_slm_v001/)
  2. GGUF F16                             (artifacts/models/hindi_slm_v001_f16.gguf)
  3. GGUF Q4_K_M                          (artifacts/models/hindi_slm_v001_q4_k_m.gguf)

GGUF steps 2 and 3 require the `gguf` package and llama.cpp quantizer.
Install: pip install gguf

Usage:
  cd slm_training
  python export.py                          # exports latest checkpoint
  python export.py --ckpt artifacts/checkpoints/step_0050000
  python export.py --skip-gguf             # HF format only
"""

import sys
import argparse
import json
from pathlib import Path

# datasets must load before PyTorch/CUDA on Windows
import datasets as _ds_preload  # noqa: F401

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SLM_TRAINING_ROOT = Path(__file__).parent
SRC_DIR = SLM_TRAINING_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def log(msg: str):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sep(title: str = ""):
    bar = "─" * 64
    print(f"\n{bar}\n  {title}\n{bar}" if title else bar, flush=True)


# ── Weight remapping ──────────────────────────────────────────────────────────

def _remap_to_llama(state_dict: dict, num_layers: int) -> dict:
    """Remap custom HindiSLM weight names to LlamaForCausalLM names.

    Our architecture is identical to LLaMA in structure; only key names differ.
    """
    new = {}

    new["model.embed_tokens.weight"] = state_dict["embed_tokens.weight"]

    for i in range(num_layers):
        p = f"layers.{i}"
        q = f"model.layers.{i}"
        new[f"{q}.input_layernorm.weight"]          = state_dict[f"{p}.attn_norm.weight"]
        new[f"{q}.self_attn.q_proj.weight"]         = state_dict[f"{p}.attn.q_proj.weight"]
        new[f"{q}.self_attn.k_proj.weight"]         = state_dict[f"{p}.attn.k_proj.weight"]
        new[f"{q}.self_attn.v_proj.weight"]         = state_dict[f"{p}.attn.v_proj.weight"]
        new[f"{q}.self_attn.o_proj.weight"]         = state_dict[f"{p}.attn.o_proj.weight"]
        new[f"{q}.post_attention_layernorm.weight"] = state_dict[f"{p}.mlp_norm.weight"]
        new[f"{q}.mlp.gate_proj.weight"]            = state_dict[f"{p}.mlp.gate_proj.weight"]
        new[f"{q}.mlp.up_proj.weight"]              = state_dict[f"{p}.mlp.up_proj.weight"]
        new[f"{q}.mlp.down_proj.weight"]            = state_dict[f"{p}.mlp.down_proj.weight"]

    new["model.norm.weight"] = state_dict["norm.weight"]
    # lm_head is tied to embed_tokens — copy explicitly so HF save_pretrained works
    new["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    return new


# ── HF export ─────────────────────────────────────────────────────────────────

def export_hf(ckpt_path: Path, out_dir: Path, tokenizer_id: str) -> Path:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

    sep("Stage 1 — HuggingFace LlamaForCausalLM")

    with open(ckpt_path / "meta.json") as f:
        meta = json.load(f)
    cfg = meta["model_config"]
    log(f"  Checkpoint step : {meta['step']:,}  loss={meta['loss']:.4f}")

    llama_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_kv_heads"],
        max_position_embeddings=cfg["max_seq_len"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_base"],
        tie_word_embeddings=False,   # we copy lm_head explicitly
        hidden_act="silu",
        architectures=["LlamaForCausalLM"],
        model_type="llama",
    )

    log("  Loading checkpoint weights ...")
    raw = torch.load(str(ckpt_path / "model.pt"), map_location="cpu", weights_only=True)
    remapped = _remap_to_llama(raw, cfg["num_layers"])

    log("  Building LlamaForCausalLM and loading remapped weights ...")
    hf_model = LlamaForCausalLM(llama_cfg)
    missing, unexpected = hf_model.load_state_dict(remapped, strict=True)
    if missing:
        log(f"  WARNING: missing keys: {missing}")
    if unexpected:
        log(f"  WARNING: unexpected keys: {unexpected}")

    # Convert to bfloat16 to keep file size small
    hf_model = hf_model.to(torch.bfloat16)
    log(f"  Params: {sum(p.numel() for p in hf_model.parameters())/1e6:.2f}M (bfloat16)")

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"  Saving model → {out_dir} ...")
    hf_model.save_pretrained(str(out_dir), safe_serialization=True)

    log("  Loading tokenizer ...")
    # try local first, fall back to HF Hub
    local_tok = (
        SLM_TRAINING_ROOT.parent
        / "tokenizer_training" / "data" / "final" / "hindi_slm_tokenizer_v001"
    )
    tok_src = str(local_tok) if local_tok.exists() else tokenizer_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    tokenizer.save_pretrained(str(out_dir))
    log(f"  Tokenizer saved (vocab={tokenizer.vocab_size:,})")

    # generation_config
    import json as _json
    gen_cfg = {
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 150,
        "repetition_penalty": 1.1,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    with open(out_dir / "generation_config.json", "w") as f:
        _json.dump(gen_cfg, f, indent=2)
    log("  generation_config.json written")

    # Quick sanity check
    log("  Smoke test: forward pass ...")
    import torch as _t
    ids = _t.tensor([[tokenizer.bos_token_id or 1] * 8])
    with _t.no_grad():
        out = hf_model(ids)
    assert out.logits.shape == (1, 8, cfg["vocab_size"]), f"Unexpected shape {out.logits.shape}"
    log(f"  Smoke test passed — logits shape {tuple(out.logits.shape)}")

    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    log(f"  Export size : {size_mb:.1f} MB")
    log(f"  HF export complete → {out_dir}")
    return out_dir


# ── GGUF export ───────────────────────────────────────────────────────────────

def export_gguf_f16(hf_dir: Path, out_path: Path) -> bool:
    """Write GGUF F16 using the gguf Python package. Returns True on success."""
    sep("Stage 2 — GGUF F16")
    try:
        import gguf
    except ImportError:
        log("  gguf package not installed — skipping GGUF export.")
        log("  To install: pip install gguf")
        log("  Then re-run: python export.py")
        return False

    import torch
    import numpy as np
    from transformers import LlamaForCausalLM, LlamaConfig
    import json

    log(f"  Loading HF model from {hf_dir} ...")
    with open(hf_dir / "config.json") as f:
        cfg = json.load(f)

    writer = gguf.GGUFWriter(str(out_path), "llama")

    # Architecture metadata
    writer.add_name("hindi-slm-v001")
    writer.add_description("Hindi SLM SMALL tier — 46M params, trained on Sangraha")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_uint32("llama.context_length",    cfg["max_position_embeddings"])
    writer.add_uint32("llama.embedding_length",  cfg["hidden_size"])
    writer.add_uint32("llama.block_count",       cfg["num_hidden_layers"])
    writer.add_uint32("llama.feed_forward_length", cfg["intermediate_size"])
    writer.add_uint32("llama.rope.dimension_count",
                      cfg["hidden_size"] // cfg["num_attention_heads"])
    writer.add_uint32("llama.attention.head_count",    cfg["num_attention_heads"])
    writer.add_uint32("llama.attention.head_count_kv", cfg["num_key_value_heads"])
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    writer.add_float32("llama.rope.freq_base", cfg.get("rope_theta", 10000.0))
    writer.add_uint32("general.file_type", gguf.LlamaFileType.MOSTLY_F16)

    # Tokenizer metadata
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
    vocab = [tokenizer.convert_ids_to_tokens(i) or f"[{i}]" for i in range(tokenizer.vocab_size)]
    scores = [0.0] * tokenizer.vocab_size
    token_types = [gguf.TokenType.NORMAL] * tokenizer.vocab_size
    for special_id in (tokenizer.bos_token_id, tokenizer.eos_token_id,
                       tokenizer.pad_token_id, tokenizer.unk_token_id):
        if special_id is not None and 0 <= special_id < tokenizer.vocab_size:
            token_types[special_id] = gguf.TokenType.CONTROL

    writer.add_tokenizer_model("llama")
    writer.add_token_list(vocab)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(tokenizer.bos_token_id or 1)
    writer.add_eos_token_id(tokenizer.eos_token_id or 2)
    writer.add_unk_token_id(tokenizer.unk_token_id or 0)

    # Write header
    writer.write_header_to_file()
    writer.write_kv_data_to_file()

    log("  Writing tensors (F16) ...")

    # Load weights
    import safetensors.torch as st
    shard = next(hf_dir.glob("*.safetensors"))
    weights = st.load_file(str(shard))

    n_layers = cfg["num_hidden_layers"]

    def _f16(t):
        return t.to(torch.float16).numpy().astype(np.float16)

    def _f32(t):
        return t.to(torch.float32).numpy().astype(np.float32)

    writer.add_tensor("token_embd.weight",  _f16(weights["model.embed_tokens.weight"]))
    writer.add_tensor("output_norm.weight", _f32(weights["model.norm.weight"]))
    writer.add_tensor("output.weight",      _f16(weights["lm_head.weight"]))

    for i in range(n_layers):
        p = f"model.layers.{i}"
        writer.add_tensor(f"blk.{i}.attn_norm.weight",  _f32(weights[f"{p}.input_layernorm.weight"]))
        writer.add_tensor(f"blk.{i}.attn_q.weight",     _f16(weights[f"{p}.self_attn.q_proj.weight"]))
        writer.add_tensor(f"blk.{i}.attn_k.weight",     _f16(weights[f"{p}.self_attn.k_proj.weight"]))
        writer.add_tensor(f"blk.{i}.attn_v.weight",     _f16(weights[f"{p}.self_attn.v_proj.weight"]))
        writer.add_tensor(f"blk.{i}.attn_output.weight",_f16(weights[f"{p}.self_attn.o_proj.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",   _f32(weights[f"{p}.post_attention_layernorm.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight",   _f16(weights[f"{p}.mlp.gate_proj.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_up.weight",     _f16(weights[f"{p}.mlp.up_proj.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_down.weight",   _f16(weights[f"{p}.mlp.down_proj.weight"]))
        if (i + 1) % 3 == 0:
            log(f"    layer {i+1}/{n_layers} written")

    writer.write_tensors_to_file()
    writer.close()

    size_mb = out_path.stat().st_size / 1e6
    log(f"  GGUF F16 written → {out_path}  ({size_mb:.1f} MB)")
    return True


def _try_quantize_q4km(f16_path: Path, q4_path: Path) -> bool:
    """Try Q4_K_M quantization via llama-quantize binary."""
    sep("Stage 3 — GGUF Q4_K_M Quantization")
    import shutil, subprocess

    quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
    if quantize_bin is None:
        # Try common install locations
        candidates = [
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
            Path.home() / "llama.cpp" / "llama-quantize",
            Path("llama-quantize"),
        ]
        for c in candidates:
            if c.exists():
                quantize_bin = str(c)
                break

    if quantize_bin is None:
        log("  llama-quantize not found — skipping Q4_K_M.")
        log("")
        log("  To quantize manually, install llama.cpp and run:")
        log(f"    llama-quantize {f16_path} {q4_path} Q4_K_M")
        log("")
        log("  Quick llama.cpp install (Linux/WSL):")
        log("    git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        log("    cmake -B ~/llama.cpp/build ~/llama.cpp -DGGML_CUDA=ON")
        log("    cmake --build ~/llama.cpp/build --config Release -j$(nproc)")
        return False

    log(f"  Found quantizer: {quantize_bin}")
    log(f"  Running Q4_K_M quantization ...")
    result = subprocess.run(
        [quantize_bin, str(f16_path), str(q4_path), "Q4_K_M"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"  ERROR: quantization failed:\n{result.stderr}")
        return False

    size_mb = q4_path.stat().st_size / 1e6
    log(f"  Q4_K_M written → {q4_path}  ({size_mb:.1f} MB)")
    return True


# ── Pi readiness check ────────────────────────────────────────────────────────

def _pi_check(q4_path: Path, hf_dir: Path):
    sep("Raspberry Pi Readiness")
    import json

    with open(hf_dir / "config.json") as f:
        cfg = json.load(f)

    model_mb = q4_path.stat().st_size / 1e6 if q4_path.exists() else None
    kv_bytes = (
        2 * cfg["num_hidden_layers"]
        * cfg["max_position_embeddings"]
        * cfg["num_key_value_heads"]
        * (cfg["hidden_size"] // cfg["num_attention_heads"])
        * 2  # fp16
    )
    kv_mb = kv_bytes / 1e6
    runtime_mb = 50  # llama.cpp overhead estimate

    if model_mb:
        total_mb = model_mb + kv_mb + runtime_mb
        log(f"  Q4_K_M model  : {model_mb:.1f} MB")
        log(f"  KV cache      : {kv_mb:.1f} MB  (ctx={cfg['max_position_embeddings']} fp16)")
        log(f"  Runtime est.  : {runtime_mb} MB")
        log(f"  Total RAM est.: {total_mb:.1f} MB  (Pi 8 GB has ~7,500 MB usable)")
        if total_mb < 1000:
            log(f"  Pi assessment : FITS COMFORTABLY (< 1 GB of 8 GB)")
        elif total_mb < 4000:
            log(f"  Pi assessment : FITS  ({total_mb:.0f} MB < 4 GB)")
        else:
            log(f"  Pi assessment : TIGHT — check available RAM")
    else:
        log("  Q4_K_M not available — skipping RAM estimate.")
        log(f"  KV cache estimate: {kv_mb:.1f} MB at ctx={cfg['max_position_embeddings']}")

    log("")
    log("  To run on Raspberry Pi:")
    if q4_path.exists():
        log(f"    scp {q4_path} pi@raspberrypi.local:~/")
        log(f"    ./llama-cli -m ~/{q4_path.name} -p 'आज का समाचार' -n 100")
    else:
        log("    (quantize first, then copy the .gguf to the Pi)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Hindi SLM checkpoint → HF format + GGUF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-dir", default=str(SLM_TRAINING_ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--ckpt", default=None, help="Specific checkpoint path; uses latest if omitted.")
    parser.add_argument("--out-dir", default=str(SLM_TRAINING_ROOT / "artifacts" / "models"))
    parser.add_argument("--name", default="hindi_slm_v001", help="Model name prefix for output files.")
    parser.add_argument("--tokenizer", default="vaibhavmaurya/hindi-slm-tokenizer-v001")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion.")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)

    sep("Hindi SLM — Export")

    # Find checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        if not ckpts:
            log(f"ERROR: no checkpoints found in {ckpt_dir}")
            raise SystemExit(1)
        ckpt_path = ckpts[-1]

    log(f"  Checkpoint : {ckpt_path}")
    log(f"  Output dir : {out_dir}")

    hf_out   = out_dir / args.name
    f16_path = out_dir / f"{args.name}_f16.gguf"
    q4_path  = out_dir / f"{args.name}_q4_k_m.gguf"

    # Stage 1: HF format
    export_hf(ckpt_path, hf_out, args.tokenizer)

    if not args.skip_gguf:
        # Stage 2: GGUF F16
        f16_ok = export_gguf_f16(hf_out, f16_path)
        # Stage 3: Q4_K_M quantize
        if f16_ok:
            _try_quantize_q4km(f16_path, q4_path)
    else:
        log("\n  --skip-gguf set — skipping GGUF stages.")

    # Pi readiness
    _pi_check(q4_path, hf_out)

    sep("Export Summary")
    log(f"  HF model    : {hf_out}")
    if f16_path.exists():
        log(f"  GGUF F16    : {f16_path}  ({f16_path.stat().st_size/1e6:.1f} MB)")
    if q4_path.exists():
        log(f"  GGUF Q4_K_M : {q4_path}  ({q4_path.stat().st_size/1e6:.1f} MB)")
    log("")
    log("  Next steps:")
    log("    1. streamlit run app.py                        # test the UI")
    if q4_path.exists():
        log(f"    2. Copy {q4_path.name} to Raspberry Pi")
        log("    3. llama-cli -m hindi_slm_v001_q4_k_m.gguf -p 'आज का समाचार' -n 100")
    else:
        log("    2. Install llama.cpp and quantize to Q4_K_M (see instructions above)")
        log("    3. Copy .gguf to Raspberry Pi")


if __name__ == "__main__":
    main()
