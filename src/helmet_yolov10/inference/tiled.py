"""Overlapping tiled inference and class-aware NMS merging for E4."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import numpy as np

@dataclass(frozen=True)
class Tile:
    x1: int; y1: int; x2: int; y2: int

def make_tiles(height: int, width: int, tile_size: tuple[int, int], overlap: float) -> list[Tile]:
    """Cover an image with overlapping tiles, retaining border pixels."""
    tile_h, tile_w = tile_size
    if min(height, width, tile_h, tile_w) <= 0 or not 0 <= overlap < 1: raise ValueError("invalid tile dimensions or overlap")
    sy, sx = max(1, round(tile_h * (1 - overlap))), max(1, round(tile_w * (1 - overlap)))
    ys, xs = list(range(0, max(height-tile_h, 0)+1, sy)) or [0], list(range(0, max(width-tile_w, 0)+1, sx)) or [0]
    if ys[-1] != max(height-tile_h, 0): ys.append(max(height-tile_h, 0))
    if xs[-1] != max(width-tile_w, 0): xs.append(max(width-tile_w, 0))
    return [Tile(x, y, min(x+tile_w, width), min(y+tile_h, height)) for y in ys for x in xs]

def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter = np.prod(np.maximum(np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]), 0), axis=1)
    return inter / np.maximum(np.prod(box[2:]-box[:2]) + np.prod(boxes[:, 2:]-boxes[:, :2], axis=1) - inter, 1e-9)

def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> np.ndarray:
    kept: list[int] = []
    for class_id in np.unique(classes):
        order = np.where(classes == class_id)[0][np.argsort(scores[classes == class_id])[::-1]]
        while len(order):
            current = int(order[0]); kept.append(current); order = order[1:][_iou(boxes[current], boxes[order[1:]]) <= iou_threshold]
    return np.asarray(sorted(kept, key=lambda index: scores[index], reverse=True), dtype=int)

def predict_tiled(model: Any, image: np.ndarray, *, tile_size: tuple[int, int]=(640,640), overlap: float=.25, conf: float=.001, iou: float=.7, device: str|int|None=None) -> dict[str, Any]:
    tiles = make_tiles(*image.shape[:2], tile_size, overlap); all_boxes=[]; all_scores=[]; all_classes=[]; started=perf_counter()
    for tile in tiles:
        result = model.predict(image[tile.y1:tile.y2, tile.x1:tile.x2], imgsz=tile_size[0], conf=conf, iou=iou, device=device, verbose=False)[0]; boxes=result.boxes
        if len(boxes):
            xyxy=boxes.xyxy.cpu().numpy(); xyxy[:,[0,2]] += tile.x1; xyxy[:,[1,3]] += tile.y1; all_boxes.append(xyxy); all_scores.append(boxes.conf.cpu().numpy()); all_classes.append(boxes.cls.cpu().numpy().astype(int))
    boxes=np.concatenate(all_boxes) if all_boxes else np.empty((0,4)); scores=np.concatenate(all_scores) if all_scores else np.empty(0); classes=np.concatenate(all_classes) if all_classes else np.empty(0,dtype=int); keep=class_aware_nms(boxes,scores,classes,iou) if len(boxes) else np.empty(0,dtype=int); seconds=perf_counter()-started
    return {"boxes":boxes[keep],"scores":scores[keep],"classes":classes[keep],"tiles":len(tiles),"seconds":seconds,"fps":1/seconds if seconds else float("inf")}
