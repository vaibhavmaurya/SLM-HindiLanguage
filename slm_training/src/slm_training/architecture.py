"""Stage 2: Architecture deduction — deterministic parameter/memory/KV calculators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class ModelConfig:
    hidden_size: int = 512
    num_layers: int = 10
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 1536
    max_seq_len: int = 512
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.1


# ---------- Predefined tiers ----------

TIER_CONFIGS: Dict[str, ModelConfig] = {
    "CPU_ONLY": ModelConfig(
        hidden_size=256, num_layers=6, num_attention_heads=4, num_kv_heads=2,
        head_dim=64, intermediate_size=768, max_seq_len=256, vocab_size=32000,
    ),
    "MICRO": ModelConfig(
        hidden_size=384, num_layers=8, num_attention_heads=6, num_kv_heads=2,
        head_dim=64, intermediate_size=1024, max_seq_len=512, vocab_size=32000,
    ),
    "SMALL": ModelConfig(  # Raspberry Pi target — design wiki MVP
        hidden_size=512, num_layers=10, num_attention_heads=8, num_kv_heads=2,
        head_dim=64, intermediate_size=1536, max_seq_len=512, vocab_size=32000,
    ),
    "MEDIUM": ModelConfig(
        hidden_size=768, num_layers=12, num_attention_heads=12, num_kv_heads=4,
        head_dim=64, intermediate_size=2048, max_seq_len=1024, vocab_size=32000,
    ),
    "LARGE": ModelConfig(
        hidden_size=1024, num_layers=16, num_attention_heads=16, num_kv_heads=4,
        head_dim=64, intermediate_size=2752, max_seq_len=2048, vocab_size=32000,
    ),
}


# ---------- Calculators ----------

class ParameterCounter:
    @staticmethod
    def count(cfg: ModelConfig) -> Dict[str, int]:
        H = cfg.hidden_size
        V = cfg.vocab_size
        L = cfg.num_layers
        Kv = cfg.num_kv_heads
        D = cfg.head_dim
        I = cfg.intermediate_size

        embedding = V * H

        # Per layer:
        # Q: H*H, K: H*(Kv*D), V: H*(Kv*D), O: H*H
        attn_per_layer = H * H + H * Kv * D + H * Kv * D + H * H
        # SwiGLU: gate(H→I) + up(H→I) + down(I→H)
        mlp_per_layer = H * I + H * I + I * H
        # Two RMSNorm per layer (attn + mlp pre-norm)
        norm_per_layer = 2 * H

        total_layers = L * (attn_per_layer + mlp_per_layer + norm_per_layer)

        # Final RMSNorm
        final_norm = H
        # LM head — tied to embedding so 0 extra params
        lm_head = 0 if cfg.tie_embeddings else V * H

        total = embedding + total_layers + final_norm + lm_head

        return {
            "embedding": embedding,
            "attention_per_layer": attn_per_layer,
            "mlp_per_layer": mlp_per_layer,
            "norm_per_layer": norm_per_layer,
            "all_layers": total_layers,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "total": total,
        }


class MemoryEstimator:
    @staticmethod
    def estimate_training_vram_gb(
        cfg: ModelConfig,
        batch_size: int = 2,
        dtype_bytes: int = 2,  # bfloat16
    ) -> Dict[str, float]:
        counts = ParameterCounter.count(cfg)
        params = counts["total"]

        weights_gb = params * dtype_bytes / 1e9
        grads_gb = params * dtype_bytes / 1e9
        # Adam: first + second moment in fp32
        adam_gb = params * 2 * 4 / 1e9

        # Activation memory: rough estimate per layer per token
        # Each layer stores: hidden states + attention weights + intermediate
        tokens = batch_size * cfg.max_seq_len
        act_per_token_bytes = (
            cfg.hidden_size * 4            # hidden state fp32
            + cfg.num_attention_heads * cfg.max_seq_len * 4  # attn weights
            + cfg.intermediate_size * 4    # mlp intermediate
        )
        activations_gb = cfg.num_layers * tokens * act_per_token_bytes / 1e9

        total_gb = weights_gb + grads_gb + adam_gb + activations_gb

        return {
            "weights_gb": round(weights_gb, 3),
            "gradients_gb": round(grads_gb, 3),
            "adam_states_gb": round(adam_gb, 3),
            "activations_gb": round(activations_gb, 3),
            "total_estimated_gb": round(total_gb, 3),
        }


class KVCacheEstimator:
    @staticmethod
    def estimate_inference_gb(cfg: ModelConfig, dtype_bytes: int = 2) -> float:
        # 2 (K and V) × layers × seq_len × kv_heads × head_dim × bytes
        cache_bytes = (
            2 * cfg.num_layers * cfg.max_seq_len
            * cfg.num_kv_heads * cfg.head_dim * dtype_bytes
        )
        return round(cache_bytes / 1e9, 4)


class NASCalculator:
    @staticmethod
    def select(tier: str) -> ModelConfig:
        if tier not in TIER_CONFIGS:
            raise ValueError(f"Unknown tier '{tier}'. Choose from: {list(TIER_CONFIGS)}")
        return TIER_CONFIGS[tier]

    @staticmethod
    def print_summary(cfg: ModelConfig) -> None:
        counts = ParameterCounter.count(cfg)
        mem = MemoryEstimator.estimate_training_vram_gb(cfg)
        kv = KVCacheEstimator.estimate_inference_gb(cfg)

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            t1 = Table(title="Model Architecture", show_header=True, header_style="bold magenta")
            t1.add_column("Parameter")
            t1.add_column("Value")
            for k, v in asdict(cfg).items():
                t1.add_row(k, str(v))
            console.print(t1)

            t2 = Table(title="Parameter Budget", show_header=True, header_style="bold blue")
            t2.add_column("Component")
            t2.add_column("Count")
            t2.add_column("M params")
            for k, v in counts.items():
                t2.add_row(k, f"{v:,}", f"{v/1e6:.2f}")
            console.print(t2)

            t3 = Table(title="VRAM Estimate (batch=2, bfloat16)", header_style="bold yellow")
            t3.add_column("Component")
            t3.add_column("GB")
            for k, v in mem.items():
                t3.add_row(k, str(v))
            t3.add_row("KV cache (inference)", str(kv))
            console.print(t3)

        except ImportError:
            print(f"Total params: {counts['total']/1e6:.2f}M")
            print(f"Estimated training VRAM: {mem['total_estimated_gb']:.2f} GB")
            print(f"KV cache (inference): {kv:.4f} GB")


def save_model_schema(cfg: ModelConfig, param_counts: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": asdict(cfg),
        "parameter_counts": param_counts,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, allow_unicode=True)
