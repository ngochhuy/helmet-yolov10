"""Unit tests for image-enhancement components."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from helmet_yolov10.enhancement.clahe import apply_clahe
from helmet_yolov10.enhancement.gamma import apply_gamma
from helmet_yolov10.enhancement.pipeline import apply_enhancement


def test_clahe_preserves_bgr_image_shape_and_type() -> None:
    image = np.full((16, 16, 3), 30, dtype=np.uint8)

    enhanced = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


def test_gamma_below_one_brightens_an_image() -> None:
    image = np.full((4, 4, 3), 64, dtype=np.uint8)

    enhanced = apply_gamma(image, gamma=0.7)

    assert int(enhanced.mean()) > int(image.mean())


def test_pipeline_dispatches_configured_method() -> None:
    image = np.full((4, 4, 3), 64, dtype=np.uint8)

    enhanced = apply_enhancement(image, "gamma", {"gamma": 0.7})

    assert enhanced.shape == image.shape
    with pytest.raises(ValueError, match="Unsupported"):
        apply_enhancement(image, "unknown", {})
