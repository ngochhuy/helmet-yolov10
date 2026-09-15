"""Train-only E2 photometric augmentation for the Ultralytics YOLO backend.

The public YOLO train arguments cover common HSV and geometric transforms, but
do not expose the project protocol's Gaussian noise and blur parameters.  This
module inserts those transforms into Ultralytics' Albumentations stage while a
training run is active.  It never affects validation or test transforms.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from typing import Any, Iterator


class AugmentationConfigError(ValueError):
    """Raised when the E2 augmentation section is incomplete or invalid."""


@dataclass(frozen=True)
class E2Augmentation:
    """Validated E2 photometric augmentation settings."""

    brightness_contrast_probability: float
    brightness_limit: float
    contrast_limit: float
    noise_probability: float
    noise_sigma_min: float
    noise_sigma_max: float
    blur_probability: float
    blur_kernel_min: int
    blur_kernel_max: int
    blur_sigma_min: float
    blur_sigma_max: float


def _probability(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise AugmentationConfigError(f"{name} must be a number in [0, 1]")
    return float(value)


def _positive(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise AugmentationConfigError(f"{name} must be positive")
    return float(value)


def parse_e2_augmentation(value: Any) -> E2Augmentation:
    """Parse the ``augmentation`` mapping from ``E2_augmentation.yaml``."""
    if not isinstance(value, dict):
        raise AugmentationConfigError("E2 requires an 'augmentation' mapping")

    def section(name: str) -> dict[str, Any]:
        candidate = value.get(name)
        if not isinstance(candidate, dict):
            raise AugmentationConfigError(f"augmentation.{name} must be a mapping")
        return candidate

    brightness = section("brightness_contrast")
    noise = section("gaussian_noise")
    blur = section("gaussian_blur")
    kernel_min = blur.get("kernel_min")
    kernel_max = blur.get("kernel_max")
    if (
        not isinstance(kernel_min, int)
        or not isinstance(kernel_max, int)
        or kernel_min <= 0
        or kernel_max < kernel_min
        or kernel_min % 2 == 0
        or kernel_max % 2 == 0
    ):
        raise AugmentationConfigError(
            "gaussian_blur kernel bounds must be positive odd integers with min <= max"
        )

    result = E2Augmentation(
        brightness_contrast_probability=_probability(
            brightness.get("probability"), "brightness_contrast.probability"
        ),
        brightness_limit=_probability(brightness.get("brightness_limit"), "brightness_limit"),
        contrast_limit=_probability(brightness.get("contrast_limit"), "contrast_limit"),
        noise_probability=_probability(noise.get("probability"), "gaussian_noise.probability"),
        noise_sigma_min=_positive(noise.get("sigma_min"), "gaussian_noise.sigma_min"),
        noise_sigma_max=_positive(noise.get("sigma_max"), "gaussian_noise.sigma_max"),
        blur_probability=_probability(blur.get("probability"), "gaussian_blur.probability"),
        blur_kernel_min=kernel_min,
        blur_kernel_max=kernel_max,
        blur_sigma_min=_positive(blur.get("sigma_min"), "gaussian_blur.sigma_min"),
        blur_sigma_max=_positive(blur.get("sigma_max"), "gaussian_blur.sigma_max"),
    )
    if result.noise_sigma_min > result.noise_sigma_max:
        raise AugmentationConfigError("gaussian_noise.sigma_min must be <= sigma_max")
    if result.blur_sigma_min > result.blur_sigma_max:
        raise AugmentationConfigError("gaussian_blur.sigma_min must be <= sigma_max")
    return result


def _gauss_noise(albumentations: Any, settings: E2Augmentation) -> Any:
    """Create GaussNoise for both Albumentations 1.x and 2.x APIs."""
    parameters = inspect.signature(albumentations.GaussNoise).parameters
    if "std_range" in parameters:
        # Albumentations 2.x expresses noise standard deviation relative to 255.
        return albumentations.GaussNoise(
            std_range=(settings.noise_sigma_min / 255.0, settings.noise_sigma_max / 255.0),
            p=settings.noise_probability,
        )
    return albumentations.GaussNoise(
        var_limit=(settings.noise_sigma_min**2, settings.noise_sigma_max**2),
        p=settings.noise_probability,
    )


def _e2_transform(albumentations: Any, original: Any, settings: E2Augmentation) -> Any:
    transforms = [
        albumentations.RandomBrightnessContrast(
            brightness_limit=settings.brightness_limit,
            contrast_limit=settings.contrast_limit,
            p=settings.brightness_contrast_probability,
        ),
        _gauss_noise(albumentations, settings),
        albumentations.GaussianBlur(
            blur_limit=(settings.blur_kernel_min, settings.blur_kernel_max),
            sigma_limit=(settings.blur_sigma_min, settings.blur_sigma_max),
            p=settings.blur_probability,
        ),
        *original.transforms,
    ]
    bbox_processor = getattr(original, "processors", {}).get("bboxes")
    bbox_params = getattr(bbox_processor, "params", None)
    if bbox_params is None:
        bbox_params = albumentations.BboxParams(format="yolo", label_fields=["class_labels"])
    return albumentations.Compose(
        transforms,
        bbox_params=bbox_params,
        additional_targets=getattr(original, "additional_targets", None),
    )


@contextmanager
def e2_augmentation_context(settings: E2Augmentation) -> Iterator[None]:
    """Temporarily add E2 transforms to Ultralytics' training augmentation stage.

    Ultralytics builds validation transforms separately, without its
    ``Albumentations`` train stage, so this patch is scoped to the call to
    ``model.train`` and does not change validation/test inputs.
    """
    try:
        import albumentations as albumentations
        import ultralytics.data.augment as yolo_augment
    except ImportError as exc:
        raise RuntimeError(
            "E2 requires albumentations and the official YOLOv10/Ultralytics backend. "
            "Install the project's train extra before running E2."
        ) from exc

    original_factory = yolo_augment.Albumentations

    def e2_factory(p: float = 1.0) -> Any:
        augmentation = original_factory(p=p)
        original_transform = getattr(augmentation, "transform", None)
        if original_transform is None:
            raise RuntimeError("Ultralytics disabled its Albumentations train transform")
        augmentation.transform = _e2_transform(albumentations, original_transform, settings)
        return augmentation

    yolo_augment.Albumentations = e2_factory
    try:
        yield
    finally:
        yolo_augment.Albumentations = original_factory
