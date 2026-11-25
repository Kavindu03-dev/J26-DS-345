"""Print GPU/OS details and write results/env.json. Exit code 1 if there is no NVIDIA GPU."""

import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MIN_VRAM_MB = 8 * 1024
PACKAGES = ("vllm", "xgrammar", "transformers", "torch", "unsloth", "trl", "fastapi", "pydantic")


def parse_query(text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`."""
    gpus = []
    for line in text.strip().splitlines():
        name, memory, driver = (part.strip() for part in line.rsplit(",", 2))
        gpus.append({"name": name, "vram_mb": int(memory.split()[0]), "driver": driver})
    return gpus


def parse_cuda(text: str) -> str | None:
    match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", text)
    return match.group(1) if match else None


def build_warnings(gpus: list[dict[str, Any]], os_name: str) -> list[str]:
    warnings = []
    if not gpus:
        warnings.append("No NVIDIA GPU found.")
    for gpu in gpus:
        if gpu["vram_mb"] < MIN_VRAM_MB:
            warnings.append(f"{gpu['name']} has {gpu['vram_mb']} MB VRAM, below 8 GB.")
    if os_name != "Linux":
        warnings.append(f"Running on {os_name}. vLLM needs Linux or WSL2.")
    return warnings


def _smi(*args: str) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def collect() -> dict[str, Any]:
    gpus = parse_query(
        _smi("--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    )
    os_name = platform.system()
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "gpus": gpus,
        "cuda": parse_cuda(_smi()),
        "os": os_name,
        "os_release": platform.release(),
        "wsl2": os_name == "Linux" and "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in PACKAGES},
        "warnings": build_warnings(gpus, os_name),
    }


def main() -> int:
    env = collect()
    path = ROOT / "results" / "env.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(env, indent=2))
    for warning in env["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote {path}")
    return 0 if env["gpus"] else 1


if __name__ == "__main__":
    sys.exit(main())
