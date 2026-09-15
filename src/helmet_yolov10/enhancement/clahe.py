"""CLAHE enhancement applied to the L channel of a BGR image."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray, *, clip_limit: float = 2.0, tile_grid_size: Sequence[int] = (8, 8)
) -> np.ndarray:
    """Return a BGR image with CLAHE applied in LAB colour space."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("CLAHE expects a uint8 NumPy image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("CLAHE expects a three-channel BGR image")
    if not isinstance(clip_limit, (int, float)) or float(clip_limit) <= 0:
        raise ValueError("clip_limit must be positive")
    if len(tile_grid_size) != 2 or any(not isinstance(value, int) or value <= 0 for value in tile_grid_size):
        raise ValueError("tile_grid_size must contain two positive integers")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    enhanced = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
