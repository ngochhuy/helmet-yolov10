"""Strict validation for a prepared YOLO detection dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DatasetValidationError(ValueError):
    """Raised when a prepared dataset cannot safely be used for training."""


@dataclass(frozen=True)
class SplitReport:
    images: int
    labelled_images: int
    background_images: int
    objects: int


def _normalise_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        names = {index: str(name) for index, name in enumerate(value)}
    elif isinstance(value, dict):
        try:
            names = {int(index): str(name) for index, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError("Class IDs in 'names' must be integers") from exc
    else:
        raise DatasetValidationError("Dataset YAML must define 'names' as a list or mapping")
    if sorted(names) != list(range(len(names))):
        raise DatasetValidationError("Class IDs must be contiguous and start at 0")
    if not names:
        raise DatasetValidationError("At least one class must be configured")
    return names


def _resolve_dataset_root(data_yaml: Path, value: str, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    project_candidate = (project_root / path).resolve()
    yaml_candidate = (data_yaml.parent / path).resolve()
    if project_candidate.exists() or not yaml_candidate.exists():
        return project_candidate
    return yaml_candidate


def _validate_label_file(label_path: Path, class_count: int) -> int:
    objects = 0
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"Label is not UTF-8: {label_path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise DatasetValidationError(
                f"Expected 5 values at {label_path}:{line_number}, got {len(fields)}"
            )
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise DatasetValidationError(
                f"Non-numeric YOLO label at {label_path}:{line_number}"
            ) from exc
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < class_count:
            raise DatasetValidationError(
                f"Invalid class ID {fields[0]} at {label_path}:{line_number}"
            )
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise DatasetValidationError(
                f"Coordinates must be normalised to [0, 1] at {label_path}:{line_number}"
            )
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise DatasetValidationError(
                f"Box width and height must be positive at {label_path}:{line_number}"
            )
        objects += 1
    return objects


def validate_dataset(data_yaml: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Validate all configured splits and return dataset statistics."""
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.is_file():
        raise DatasetValidationError(f"Dataset YAML does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise DatasetValidationError("Dataset YAML must contain a mapping")

    names = _normalise_names(config.get("names"))
    raw_root = config.get("path")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise DatasetValidationError("Dataset YAML must define a non-empty 'path'")
    root = _resolve_dataset_root(yaml_path, raw_root, Path(project_root).resolve())
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset root does not exist: {root}")

    reports: dict[str, dict[str, int]] = {}
    split_files: dict[str, set[Path]] = {}
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not isinstance(split_value, str) or not split_value.strip():
            raise DatasetValidationError(f"Dataset YAML must define '{split}'")
        image_dir = (root / split_value).resolve()
        if not image_dir.is_dir():
            raise DatasetValidationError(f"Missing {split} image directory: {image_dir}")
        images = sorted(
            path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise DatasetValidationError(f"No images found in {split}: {image_dir}")

        objects = 0
        labelled_images = 0
        identities: set[Path] = set()
        for image_path in images:
            relative = image_path.relative_to(image_dir)
            identities.add(relative.with_suffix(""))
            try:
                image_relative_to_root = image_path.relative_to(root)
            except ValueError as exc:
                raise DatasetValidationError(f"Image is outside dataset root: {image_path}") from exc
            parts = list(image_relative_to_root.parts)
            try:
                images_index = parts.index("images")
            except ValueError as exc:
                raise DatasetValidationError(
                    f"Image path must contain an 'images' directory: {image_path}"
                ) from exc
            parts[images_index] = "labels"
            label_path = (root.joinpath(*parts)).with_suffix(".txt")
            if not label_path.is_file():
                raise DatasetValidationError(f"Missing label file for {image_path}: {label_path}")
            count = _validate_label_file(label_path, len(names))
            objects += count
            labelled_images += int(count > 0)

        split_files[split] = identities
        report = SplitReport(
            images=len(images),
            labelled_images=labelled_images,
            background_images=len(images) - labelled_images,
            objects=objects,
        )
        reports[split] = asdict(report)

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_files[first] & split_files[second]
        if overlap:
            example = sorted(str(path) for path in overlap)[:3]
            raise DatasetValidationError(
                f"Potential split leakage ({first}/{second}) for relative IDs: {example}"
            )

    return {
        "data_yaml": str(yaml_path),
        "dataset_root": str(root),
        "names": names,
        "splits": reports,
        "total_images": sum(report["images"] for report in reports.values()),
        "total_objects": sum(report["objects"] for report in reports.values()),
    }


def write_resolved_data_yaml(
    source: str | Path, destination: str | Path, project_root: str | Path
) -> Path:
    """Write a copy whose dataset root is absolute for backend portability."""
    source_path = Path(source).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["path"] = str(
        _resolve_dataset_root(source_path, str(config["path"]), Path(project_root).resolve())
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return destination_path.resolve()
