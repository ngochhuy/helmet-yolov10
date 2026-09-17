"""Evaluate a YOLOv10 baseline checkpoint on the locked test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from helmet_yolov10.data.validation import (
    DatasetValidationError,
    validate_dataset,
    write_resolved_data_yaml,
)
from helmet_yolov10.training.train import PROJECT_ROOT, _load_backend, _project_path
from helmet_yolov10.utils.config import load_config, require_sections
from helmet_yolov10.utils.environment import collect_environment, sha256_file, write_json
from helmet_yolov10.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _as_serialisable_metrics(result: Any) -> dict[str, float]:
    raw = getattr(result, "results_dict", None)
    if not isinstance(raw, dict):
        raise RuntimeError("YOLO validation result does not expose results_dict")
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    return metrics


def evaluate_checkpoint(
    checkpoint: str | Path,
    config_path: str | Path = "configs/E1_baseline.yaml",
    *,
    data_path: str | Path | None = None,
    run_name: str | None = None,
    device: str | int | None = None,
    backend_factory: Callable[..., Any] | None = None,
) -> Path:
    """Evaluate one checkpoint and persist metrics plus environment metadata."""
    checkpoint_path = _project_path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    config = load_config(_project_path(config_path))
    require_sections(config, "training", "evaluation", "output")
    if data_path is not None:
        data_yaml = _project_path(data_path)
    else:
        data_yaml = _project_path(config["training"]["data"])
    dataset_report = validate_dataset(data_yaml, PROJECT_ROOT)

    evaluation = dict(config["evaluation"])
    if device is not None:
        evaluation["device"] = device
    elif "device" in config["training"]:
        evaluation["device"] = config["training"]["device"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_name = run_name or f"test_{timestamp}"
    output_root = _project_path(config["output"]["root"]) / "evaluation"
    output_dir = output_root / selected_name
    if output_dir.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    resolved_data = write_resolved_data_yaml(
        data_yaml, output_dir / "data.resolved.yaml", PROJECT_ROOT
    )
    evaluation.update(
        {
            "data": str(resolved_data),
            "project": str(output_root),
            "name": selected_name,
            "exist_ok": True,
        }
    )
    backend = (backend_factory or _load_backend())(str(checkpoint_path))
    result = backend.val(**evaluation)
    metrics = _as_serialisable_metrics(result)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "dataset_report.json", dataset_report)

    metadata = collect_environment(PROJECT_ROOT)
    metadata.update(
        {
            "experiment_id": config.get("experiment", {}).get("id"),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "evaluation_arguments": evaluation,
        }
    )
    write_json(output_dir / "metadata.json", metadata)
    LOGGER.info("Evaluation completed: %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an E1 checkpoint on test data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/E1_baseline.yaml")
    parser.add_argument("--data", help="Custom data.yaml path for subtest evaluation")
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        evaluate_checkpoint(
            args.checkpoint,
            args.config,
            data_path=args.data,
            run_name=args.run_name,
            device=args.device,
        )

    except (DatasetValidationError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
