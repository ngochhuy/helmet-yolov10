"""Configuration-driven low-light enhancement dispatch for E3."""

from __future__ import annotations

from typing import Any

import numpy as np

from helmet_yolov10.enhancement.clahe import apply_clahe
from helmet_yolov10.enhancement.gamma import apply_gamma


def apply_enhancement(image: np.ndarray, method: str, settings: dict[str, Any]) -> np.ndarray:
    """Apply one configured E3 method to a BGR image."""
    if method == "clahe":
        return apply_clahe(
            image,
            clip_limit=settings.get("clip_limit", 2.0),
            tile_grid_size=settings.get("tile_grid_size", (8, 8)),
        )
    if method == "gamma":
        return apply_gamma(image, gamma=settings.get("gamma", 0.7))
    raise ValueError(f"Unsupported enhancement method: {method}")
