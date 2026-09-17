"""Tests for baseline configuration, data validation and orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from helmet_yolov10.data.validation import DatasetValidationError, validate_dataset
from helmet_yolov10.training.train import train_baseline
from helmet_yolov10.utils.config import load_config


def _make_dataset(root: Path, *, invalid_coordinate: bool = False) -> Path:
    for index, split in enumerate(("train", "val", "test")):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f"{split}_{index}.jpg").write_bytes(b"test-image-placeholder")
        coordinate = "1.2" if invalid_coordinate and split == "train" else "0.5"
        (label_dir / f"{split}_{index}.txt").write_text(
            f"0 {coordinate} 0.5 0.2 0.2\n", encoding="utf-8"
        )

    data_yaml = root.parent / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "helmet", 1: "head", 2: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_experiment_config(
    tmp_path: Path, data_yaml: Path, weights: Path, output_root: Path
) -> Path:
    config = {
        "experiment": {"id": "E1", "name": "baseline"},
        "model": {"architecture": "YOLOv10n", "weights": str(weights)},
        "training": {
            "data": str(data_yaml),
            "imgsz": 640,
            "epochs": 1,
            "batch": 1,
            "optimizer": "SGD",
            "lr0": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "patience": 1,
            "seed": 42,
            "deterministic": True,
            "device": "cpu",
        },
        "output": {"root": str(output_root)},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_e1_config_resolves_base_values() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "E1_baseline.yaml")

    assert config["experiment"]["id"] == "E1"
    assert config["model"]["architecture"] == "YOLOv10n"
    assert config["training"]["imgsz"] == 640
    assert config["training"]["seed"] == 42
    assert config["output"]["root"] == "experiments/E1_baseline"


def test_dataset_validator_reports_all_splits(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")

    report = validate_dataset(data_yaml, tmp_path)

    assert report["total_images"] == 3
    assert report["total_objects"] == 3
    assert report["splits"]["test"]["images"] == 1


def test_dataset_validator_rejects_non_normalised_boxes(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset", invalid_coordinate=True)

    with pytest.raises(DatasetValidationError, match="normalised"):
        validate_dataset(data_yaml, tmp_path)


def test_training_orchestration_passes_locked_baseline_arguments(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    weights = tmp_path / "yolov10n.pt"
    weights.write_bytes(b"fake-weights")
    config = _make_experiment_config(tmp_path, data_yaml, weights, tmp_path / "runs")
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self, model_path: str) -> None:
            calls["model_path"] = model_path

        def train(self, **kwargs: object) -> None:
            calls["train"] = kwargs

    run_dir = train_baseline(config, run_name="unit-test", backend_factory=FakeModel)

    assert run_dir == tmp_path / "runs" / "unit-test"
    assert calls["model_path"] == str(weights)
    train_args = calls["train"]
    assert isinstance(train_args, dict)
    assert train_args["imgsz"] == 640
    assert train_args["seed"] == 42
    assert train_args["exist_ok"] is True
    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "data.resolved.yaml").is_file()
    assert (run_dir / "dataset_report.json").is_file()
    assert (run_dir / "metadata.json").is_file()
