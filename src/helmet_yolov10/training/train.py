"""Reproducible YOLOv10n baseline training entry point."""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from helmet_yolov10.data.validation import (
    DatasetValidationError,
    validate_dataset,
    write_resolved_data_yaml,
)
from helmet_yolov10.utils.config import ConfigError, load_config, require_sections
from helmet_yolov10.utils.environment import collect_environment, sha256_file, write_json
from helmet_yolov10.utils.logger import get_logger
from helmet_yolov10.utils.seed import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGGER = get_logger(__name__)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _validate_baseline_config(config: dict[str, Any]) -> None:
    require_sections(config, "experiment", "model", "training", "output")
    experiment = config["experiment"]
    model = config["model"]
    training = config["training"]

    if experiment.get("id") != "E1":
        raise ConfigError("Baseline trainer requires experiment.id = 'E1'")
    if str(model.get("architecture", "")).lower() != "yolov10n":
        raise ConfigError("Baseline architecture must be YOLOv10n")
    if training.get("imgsz") != 640:
        raise ConfigError("Baseline input size must be 640")

    positive = ("epochs", "batch", "lr0", "momentum", "weight_decay", "patience")
    invalid = [key for key in positive if not isinstance(training.get(key), (int, float)) or training[key] <= 0]
    if invalid:
        raise ConfigError(f"Training values must be positive: {', '.join(invalid)}")
    if not isinstance(training.get("seed"), int):
        raise ConfigError("training.seed must be an integer")


def _load_backend() -> Callable[..., Any]:
    try:
        from ultralytics import YOLOv10

        return YOLOv10
    except (ImportError, AttributeError):
        try:
            from ultralytics import YOLO

            return YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO backend is unavailable. Install the official THU-MIG/yolov10 "
                "package in the training environment."
            ) from exc


def _new_run_directory(output_root: Path, run_name: str) -> Path:
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose another --run-name."
        )
    run_dir.mkdir(parents=True)
    return run_dir


def train_baseline(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    device: str | int | None = None,
    weights: str | Path | None = None,
    validate_only: bool = False,
    backend_factory: Callable[..., Any] | None = None,
) -> Path | None:
    """Validate inputs and train E1, returning its run directory."""
    config_file = _project_path(config_path)
    config = load_config(config_file)
    _validate_baseline_config(config)

    training = dict(config["training"])
    data_yaml = _project_path(training.pop("data"))
    dataset_report = validate_dataset(data_yaml, PROJECT_ROOT)
    LOGGER.info(
        "Dataset valid: %s images, %s objects",
        dataset_report["total_images"],
        dataset_report["total_objects"],
    )
    if validate_only:
        return None

    weights_path = _project_path(weights or config["model"]["weights"])
    if not weights_path.is_file():
        source = config["model"].get("pretrained_source", "the official release")
        raise FileNotFoundError(
            f"Pretrained weights not found: {weights_path}. Download them from {source}."
        )

    if device is not None:
        training["device"] = device
    seed_everything(training["seed"], bool(training.get("deterministic", True)))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_name = run_name or f"baseline_seed{training['seed']}_{timestamp}"
    output_root = _project_path(config["output"]["root"])
    run_dir = _new_run_directory(output_root, selected_name)

    resolved_config_path = run_dir / "config.resolved.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    resolved_data_path = write_resolved_data_yaml(
        data_yaml, run_dir / "data.resolved.yaml", PROJECT_ROOT
    )
    write_json(run_dir / "dataset_report.json", dataset_report)

    metadata = collect_environment(PROJECT_ROOT)
    metadata.update(
        {
            "status": "running",
            "experiment_id": config["experiment"]["id"],
            "run_name": selected_name,
            "seed": training["seed"],
            "weights": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "config": str(resolved_config_path),
            "config_sha256": sha256_file(resolved_config_path),
        }
    )
    write_json(run_dir / "metadata.json", metadata)

    training.update(
        {
            "data": str(resolved_data_path),
            "project": str(output_root),
            "name": selected_name,
            "exist_ok": True,
        }
    )

    try:
        backend = (backend_factory or _load_backend())(str(weights_path))
        backend.train(**training)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
        write_json(run_dir / "metadata.json", metadata)
        raise

    metadata["status"] = "completed"
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "metadata.json", metadata)
    LOGGER.info("Baseline completed: %s", run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the reproducible E1 YOLOv10n baseline")
    parser.add_argument("--config", default="configs/E1_baseline.yaml")
    parser.add_argument("--run-name", help="Unique output directory name")
    parser.add_argument("--device", help="Override the configured device, e.g. 0 or cpu")
    parser.add_argument("--weights", help="Override the pretrained .pt path")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and dataset without loading the model",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        train_baseline(
            args.config,
            run_name=args.run_name,
            device=args.device,
            weights=args.weights,
            validate_only=args.validate_only,
        )
    except (ConfigError, DatasetValidationError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
