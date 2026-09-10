"""YAML configuration loading with deterministic inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return value


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML file and recursively resolve its optional ``extends`` key."""
    config_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ConfigError(f"Circular configuration inheritance: {config_path}")
    seen.add(config_path)

    current = _read_yaml(config_path)
    parent_name = current.pop("extends", None)
    if parent_name is None:
        result = current
    else:
        parent_path = (config_path.parent / str(parent_name)).resolve()
        result = deep_merge(load_config(parent_path, seen), current)

    seen.remove(config_path)
    return result


def require_sections(config: dict[str, Any], *sections: str) -> None:
    """Require non-empty mapping sections in a resolved config."""
    missing = [name for name in sections if not isinstance(config.get(name), dict)]
    if missing:
        raise ConfigError(f"Missing configuration section(s): {', '.join(missing)}")
