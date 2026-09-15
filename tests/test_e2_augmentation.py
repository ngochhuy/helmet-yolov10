"""Tests for the E2 train-only augmentation protocol configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from helmet_yolov10.augmentation.enhanced import (
    AugmentationConfigError,
    parse_e2_augmentation,
)
from helmet_yolov10.utils.config import load_config


def test_e2_config_resolves_custom_and_native_augmentation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "E2_augmentation.yaml")
    settings = parse_e2_augmentation(config["augmentation"])

    assert config["experiment"]["id"] == "E2"
    assert settings.brightness_contrast_probability == 0.40
    assert settings.noise_sigma_min == 5.0
    assert settings.blur_kernel_max == 5
    assert config["training"]["scale"] == 0.7
    assert config["training"]["translate"] == 0.15
    assert config["training"]["close_mosaic"] == 15


def test_e2_augmentation_rejects_even_blur_kernel() -> None:
    config = {
        "brightness_contrast": {"probability": 0.4, "brightness_limit": 0.2, "contrast_limit": 0.15},
        "gaussian_noise": {"probability": 0.15, "sigma_min": 5, "sigma_max": 15},
        "gaussian_blur": {
            "probability": 0.15,
            "kernel_min": 2,
            "kernel_max": 5,
            "sigma_min": 0.1,
            "sigma_max": 1.5,
        },
    }

    with pytest.raises(AugmentationConfigError, match="odd integers"):
        parse_e2_augmentation(config)
