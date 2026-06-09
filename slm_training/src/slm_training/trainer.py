"""Stage 7: Training loop with OOM recovery, gradient checkpointing, and checkpointing."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from .architecture import ModelConfig
from .model import HindiSLM


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 50_000
    warmup_steps: int = 1_000
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    num_workers: int = 0
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1_000
    dtype: str = "bfloat16"  # bfloat16 or float16
    max_oom_retries: int = 5


def save_training_config(cfg: TrainingConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    return torch.bfloat16 if dtype_str == "bfloat16" else torch.float16


def _build_optimizer(model: HindiSLM, cfg: TrainingConfig) -> torch.optim.Optimizer:
    decay_params = [p for name, p in model.named_parameters() if p.ndim >= 2]
    no_decay_params = [p for name, p in model.named_parameters() if p.ndim < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)


def _cosine_lr(step: int, warmup: int, total: int, lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return checkpoints[-1] if checkpoints else None


def _save_checkpoint(
    model: HindiSLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    ckpt_dir: Path,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
) -> None:
    save_path = ckpt_dir / f"step_{step:07d}"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pt")
    torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
    meta = {
        "step": step,
        "loss": loss,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }
    with open(save_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _load_checkpoint(ckpt_path: Path, model: HindiSLM, optimizer: torch.optim.Optimizer) -> int:
    model.load_state_dict(torch.load(ckpt_path / "model.pt", map_location="cpu"))
    optimizer.load_state_dict(torch.load(ckpt_path / "optimizer.pt", map_location="cpu"))
    with open(ckpt_path / "meta.json") as f:
        meta = json.load(f)
    return meta["step"]


class OOMRecovery:
    """Progressively relaxes training settings on CUDA OOM."""

    def __init__(self, train_cfg: TrainingConfig, model_cfg: ModelConfig):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.retries = 0

    def on_oom(self) -> bool:
        """Apply next recovery strategy. Returns False when exhausted."""
        torch.cuda.empty_cache()
        self.retries += 1

        if self.retries > self.train_cfg.max_oom_retries:
            return False

        if self.retries == 1:
            self.train_cfg.per_device_batch_size = max(1, self.train_cfg.per_device_batch_size // 2)
            print(f"[OOM] Retry {self.retries}: batch_size → {self.train_cfg.per_device_batch_size}")
        elif self.retries == 2:
            self.train_cfg.gradient_accumulation_steps = max(1, self.train_cfg.gradient_accumulation_steps // 2)
            print(f"[OOM] Retry {self.retries}: grad_accum → {self.train_cfg.gradient_accumulation_steps}")
        elif self.retries == 3:
            self.model_cfg.max_seq_len = self.model_cfg.max_seq_len // 2
            print(f"[OOM] Retry {self.retries}: seq_len → {self.model_cfg.max_seq_len}")
        elif self.retries == 4:
            self.model_cfg.hidden_size = 384
            self.model_cfg.num_layers = 8
            print(f"[OOM] Retry {self.retries}: tier down → hidden=384, layers=8")

        return True


def train(
    model: HindiSLM,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    train_loader,
    val_loader,
    ckpt_dir: Path,
    device: torch.device,
    tb_writer=None,
) -> None:
    """Main training loop with OOM recovery and checkpoint resumption."""

    dtype = _get_torch_dtype(train_cfg.dtype)
    model = model.to(device)
    optimizer = _build_optimizer(model, train_cfg)

    # Resume from checkpoint if available
    start_step = 0
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        start_step = _load_checkpoint(latest_ckpt, model, optimizer)
        model = model.to(device)
        print(f"[trainer] Resumed from checkpoint: {latest_ckpt.name} (step {start_step})")
    else:
        print("[trainer] Starting training from scratch")

    oom_recovery = OOMRecovery(train_cfg, model_cfg)

    try:
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}  lr={task.fields[lr]:.2e}"),
            TimeRemainingColumn(),
        )
        task = progress.add_task(
            "Training", total=train_cfg.max_steps - start_step, loss=0.0, lr=train_cfg.learning_rate
        )
        use_rich = True
    except ImportError:
        progress = None
        use_rich = False

    step = start_step
    train_iter = iter(train_loader)
    log_loss = 0.0   # accumulates over log_every steps for TensorBoard average
    step_loss = 0.0  # per-optimizer-step loss for progress bar display
    t0 = time.perf_counter()

    if use_rich:
        progress.start()

    try:
        while step < train_cfg.max_steps:
            model.train()
            optimizer.zero_grad()
            step_loss = 0.0  # reset each optimizer step
            oom_occurred = False

            for micro_step in range(train_cfg.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(device)
                labels = input_ids.clone()

                try:
                    with torch.amp.autocast(device_type=device.type, dtype=dtype):
                        _, loss = model(input_ids, labels=labels)
                    loss = loss / train_cfg.gradient_accumulation_steps
                    loss.backward()
                    step_loss += loss.item()

                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    can_continue = oom_recovery.on_oom()
                    if not can_continue:
                        print("[trainer] OOM: all recovery strategies exhausted. Stopping.")
                        return
                    oom_occurred = True
                    break

            # Skip optimizer step if OOM cleared the gradients mid-accumulation
            if oom_occurred:
                continue

            # Update LR
            lr = _cosine_lr(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm).item()
            optimizer.step()
            step += 1
            log_loss += step_loss

            if use_rich:
                progress.update(task, advance=1, loss=step_loss, lr=lr)

            if step % train_cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                tokens_per_sec = (
                    train_cfg.log_every
                    * train_cfg.per_device_batch_size
                    * train_cfg.gradient_accumulation_steps
                    * model_cfg.max_seq_len
                    / elapsed
                )
                avg_loss = log_loss / train_cfg.log_every
                vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                if not use_rich:
                    print(
                        f"step={step:6d}  loss={avg_loss:.4f}  lr={lr:.2e}"
                        f"  grad_norm={grad_norm:.3f}  tok/s={tokens_per_sec:,.0f}  vram={vram_gb:.2f}GB"
                    )
                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, step)
                    tb_writer.add_scalar("train/lr", lr, step)
                    tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                    tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)
                    tb_writer.add_scalar("train/vram_gb", vram_gb, step)
                log_loss = 0.0
                t0 = time.perf_counter()

            if step % train_cfg.eval_every == 0 and val_loader is not None:
                val_loss = evaluate_loss(model, val_loader, device, dtype)
                if not use_rich:
                    print(f"  [eval] step={step}  val_loss={val_loss:.4f}  perplexity={math.exp(val_loss):.2f}")
                if tb_writer:
                    tb_writer.add_scalar("val/loss", val_loss, step)
                    tb_writer.add_scalar("val/perplexity", math.exp(val_loss), step)

            if step % train_cfg.save_every == 0:
                _save_checkpoint(model, optimizer, step, step_loss, ckpt_dir, model_cfg, train_cfg)
                if not use_rich:
                    print(f"  [ckpt] Saved checkpoint at step {step}")

    except KeyboardInterrupt:
        print(f"\n[trainer] Interrupted at step {step} — saving checkpoint ...")
        _save_checkpoint(model, optimizer, step, step_loss, ckpt_dir, model_cfg, train_cfg)
        print(f"[trainer] Checkpoint saved. Re-run train.py to resume.")
        raise  # propagate so train.py logs the clean stop message

    finally:
        if use_rich:
            progress.stop()

    # Final checkpoint (only reached when training completes normally)
    _save_checkpoint(model, optimizer, step, step_loss, ckpt_dir, model_cfg, train_cfg)
    print(f"[trainer] Training complete at step {step}. Final checkpoint saved.")


@torch.no_grad()
def evaluate_loss(model: HindiSLM, loader, device: torch.device, dtype: torch.dtype) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            _, loss = model(input_ids, labels=labels)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)
