"""Capture the software and hardware context of an experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a file, or ``None`` when it is absent."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_info(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def collect_environment(project_root: Path) -> dict[str, Any]:
    """Collect reproducibility metadata without making a GPU mandatory."""
    metadata: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: _package_version(name)
            for name in ("ultralytics", "torch", "torchvision", "numpy", "PyYAML")
        },
        "source": _git_info(project_root),
        "cuda": {"available": False, "version": None, "devices": []},
    }

    try:
        import torch

        available = torch.cuda.is_available()
        metadata["cuda"] = {
            "available": available,
            "version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            ),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            if available
            else [],
        }
    except ImportError:
        pass
    return metadata


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
