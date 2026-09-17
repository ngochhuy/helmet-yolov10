"""Unit tests for E5 full method inference pipeline."""

from typing import Any
import numpy as np

from helmet_yolov10.inference.full_method import predict_full_method


class DummyBox:
    def __init__(self) -> None:
        import torch

        self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        self.conf = torch.tensor([0.95])
        self.cls = torch.tensor([0.0])

    def __len__(self) -> int:
        return 1


class DummyPred:
    def __init__(self) -> None:
        self.boxes = DummyBox()


class DummyModel:
    def predict(self, image: np.ndarray, **kwargs: Any) -> list[DummyPred]:
        return [DummyPred()]


def test_predict_full_method_with_enhancement_and_tiled() -> None:
    model = DummyModel()
    dummy_img = np.zeros((800, 800, 3), dtype=np.uint8)

    enhancement_config = {
        "method": "clahe",
        "color_space": "LAB",
        "channel": "L",
        "clip_limit": 2.0,
        "tile_grid_size": [8, 8],
    }

    tiled_config = {
        "tile_size": [640, 640],
        "overlap": 0.25,
        "confidence_threshold": 0.001,
        "merge_iou_threshold": 0.7,
        "merge_method": "class_aware_nms",
    }

    res = predict_full_method(
        model,
        dummy_img,
        enhancement_config=enhancement_config,
        tiled_config=tiled_config,
    )

    assert "boxes" in res
    assert "scores" in res
    assert "classes" in res
    assert res["tiles"] > 1
    assert res["total_seconds"] >= 0
    assert res["total_fps"] > 0
