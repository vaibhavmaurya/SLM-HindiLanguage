"""Stage 1: System profiling — hardware detection and training tier classification."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SystemProfile:
    os_name: str
    python_version: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_threads: int
    ram_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_vram_gb: float
    gpu_architecture: str
    cuda_version: str
    bfloat16_supported: bool
    disk_free_gb: float
    training_tier: str  # MICRO / SMALL / MEDIUM / LARGE


def detect_profile(disk_path: str = ".") -> SystemProfile:
    import sys

    os_name = platform.system() + " " + platform.release()
    python_version = sys.version.split()[0]

    cpu_model = _cpu_model()
    cpu_cores, cpu_threads = _cpu_counts()
    ram_gb = _ram_gb()
    disk_free_gb = shutil.disk_usage(disk_path).free / 1e9

    gpu_available = False
    gpu_name = "None"
    gpu_vram_gb = 0.0
    gpu_architecture = "None"
    cuda_version = "None"
    bfloat16_supported = False

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = props.total_memory / 1e9
            gpu_architecture = _gpu_architecture(props)
            cuda_version = torch.version.cuda or "unknown"
            bfloat16_supported = torch.cuda.is_bf16_supported()
    except ImportError:
        pass

    tier = _classify_tier(gpu_vram_gb)

    return SystemProfile(
        os_name=os_name,
        python_version=python_version,
        cpu_model=cpu_model,
        cpu_cores_physical=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=round(ram_gb, 1),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram_gb, 2),
        gpu_architecture=gpu_architecture,
        cuda_version=cuda_version,
        bfloat16_supported=bfloat16_supported,
        disk_free_gb=round(disk_free_gb, 1),
        training_tier=tier,
    )


def _classify_tier(vram_gb: float) -> str:
    if vram_gb <= 0:
        return "CPU_ONLY"
    if vram_gb < 4:
        return "MICRO"
    if vram_gb < 10:
        return "SMALL"
    if vram_gb < 20:
        return "MEDIUM"
    return "LARGE"


def _cpu_model() -> str:
    try:
        import subprocess
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name", "/value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip()
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _cpu_counts() -> tuple[int, int]:
    physical = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or physical
        logical = psutil.cpu_count(logical=True) or physical
        return physical, logical
    except ImportError:
        return physical, physical


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0


def _gpu_architecture(props) -> str:
    cc = (props.major, props.minor)
    arch_map = {
        (8, 9): "Ada Lovelace",
        (8, 6): "Ampere",
        (8, 0): "Ampere",
        (7, 5): "Turing",
        (7, 0): "Volta",
        (6, 1): "Pascal",
        (6, 0): "Pascal",
    }
    return arch_map.get(cc, f"SM {props.major}.{props.minor}")


def save_profile(profile: SystemProfile, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(profile), f, default_flow_style=False, allow_unicode=True)


def print_profile_table(profile: SystemProfile) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="System Profile", show_header=True, header_style="bold cyan")
        table.add_column("Component", style="bold")
        table.add_column("Value")

        rows = [
            ("OS", profile.os_name),
            ("Python", profile.python_version),
            ("CPU", profile.cpu_model),
            ("CPU Cores (physical)", str(profile.cpu_cores_physical)),
            ("CPU Threads", str(profile.cpu_threads)),
            ("RAM", f"{profile.ram_gb:.1f} GB"),
            ("GPU Available", "Yes" if profile.gpu_available else "No"),
            ("GPU Name", profile.gpu_name),
            ("GPU VRAM", f"{profile.gpu_vram_gb:.2f} GB"),
            ("GPU Architecture", profile.gpu_architecture),
            ("CUDA Version", profile.cuda_version),
            ("bfloat16 Supported", "Yes" if profile.bfloat16_supported else "No"),
            ("Free Disk", f"{profile.disk_free_gb:.1f} GB"),
            ("Training Tier", f"[bold green]{profile.training_tier}[/bold green]"),
        ]
        for k, v in rows:
            table.add_row(k, v)
        console.print(table)
    except ImportError:
        for field, val in asdict(profile).items():
            print(f"  {field:30s}: {val}")
