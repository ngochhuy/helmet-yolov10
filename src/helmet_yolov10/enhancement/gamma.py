"""Gamma correction for uint8 BGR images."""

from __future__ import annotations

import cv2
import numpy as np


def apply_gamma(image: np.ndarray, *, gamma: float = 0.7) -> np.ndarray:
    """Apply ``output = 255 * (input / 255) ** gamma`` to a BGR image.

    A gamma below one brightens a conventional uint8 image; a gamma above one
    darkens it.  The implementation uses a lookup table to preserve uint8
    output and avoid per-pixel Python work.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Gamma correction expects a uint8 NumPy image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Gamma correction expects a three-channel BGR image")
    if not isinstance(gamma, (int, float)) or float(gamma) <= 0:
        raise ValueError("gamma must be positive")

    values = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip(np.rint(255.0 * values ** float(gamma)), 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)
