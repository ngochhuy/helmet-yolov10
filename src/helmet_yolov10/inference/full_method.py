"""Full methodology inference pipeline combining low-light enhancement and tiled inference (E5)."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from helmet_yolov10.enhancement.pipeline import apply_enhancement
from helmet_yolov10.inference.tiled import predict_tiled


def predict_full_method(
    model: Any,
    image: np.ndarray,
    *,
    enhancement_config: dict[str, Any] | None = None,
    tiled_config: dict[str, Any] | None = None,
    device: str | int | None = None,
) -> dict[str, Any]:
    """Execute the full E5 proposed pipeline on a single image.

    Pipeline steps:
    1. Apply low-light image enhancement (CLAHE/Gamma) if enhancement_config is provided.
    2. Perform overlapping tiled inference for small objects if tiled_config is provided.
    3. Aggregate predictions and measure total execution latency / FPS.
    """
    start_time = perf_counter()

    # Step 1: Preprocessing / Enhancement
    if enhancement_config and enhancement_config.get("method"):
        method = enhancement_config["method"]
        enhanced_image = apply_enhancement(image, method, enhancement_config)
    else:
        enhanced_image = image

    # Step 2: Inference (Tiled or Standard)
    if tiled_config:
        tile_size = tuple(tiled_config.get("tile_size", (640, 640)))
        overlap = float(tiled_config.get("overlap", 0.25))
        conf = float(tiled_config.get("confidence_threshold", 0.001))
        iou = float(tiled_config.get("merge_iou_threshold", 0.7))

        result = predict_tiled(
            model=model,
            image=enhanced_image,
            tile_size=tile_size,
            overlap=overlap,
            conf=conf,
            iou=iou,
            device=device,
        )
    else:
        # Standard inference
        conf = 0.001 if not tiled_config else tiled_config.get("confidence_threshold", 0.001)
        pred = model.predict(enhanced_image, conf=conf, device=device, verbose=False)[0]
        boxes_obj = pred.boxes
        if len(boxes_obj):
            boxes = boxes_obj.xyxy.cpu().numpy()
            scores = boxes_obj.conf.cpu().numpy()
            classes = boxes_obj.cls.cpu().numpy().astype(int)
        else:
            boxes = np.empty((0, 4))
            scores = np.empty(0)
            classes = np.empty(0, dtype=int)

        result = {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "tiles": 1,
        }

    total_seconds = perf_counter() - start_time
    result["total_seconds"] = total_seconds
    result["total_fps"] = 1.0 / total_seconds if total_seconds > 0 else float("inf")

    return result
