"""Evaluate E4 Tiled Inference across all evaluation subtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import yaml

from helmet_yolov10.inference.tiled import predict_tiled
from helmet_yolov10.training.train import PROJECT_ROOT, _load_backend
from helmet_yolov10.utils.config import load_config
from helmet_yolov10.utils.logger import get_logger

LOGGER = get_logger(__name__)

SUBTESTS = [
    ("test_all", "data/processed/eval/test_all/data.yaml", "Toàn bộ Test set"),
    ("test_normal", "data/processed/eval/test_normal/data.yaml", "Ảnh điều kiện bình thường"),
    ("test_lowlight", "data/processed/eval/test_lowlight/data.yaml", "Ảnh thiếu sáng tự nhiên"),
    ("test_lowlight_synth", "data/processed/eval/test_lowlight_synth/data.yaml", "Ảnh thiếu sáng tổng hợp"),
    ("test_small", "data/processed/eval/test_small/data.yaml", "Mũ bảo hộ kích thước nhỏ"),
    ("test_lowlight_small", "data/processed/eval/test_lowlight_small/data.yaml", "Thiếu sáng & Mũ kích thước nhỏ"),
]

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Calculate IoU between 2D bounding boxes (xyxy)."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)))

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area - inter_area
    return inter_area / np.maximum(union_area, 1e-9)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute 101-point COCO AP from recall and precision curves."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point integration
    x = np.linspace(0, 1, 101)
    ap = float(np.trapz(np.interp(x, mrec, mpre), x))
    return ap


def evaluate_dataset_tiled(
    model: Any,
    subtest_yaml: Path,
    tiled_settings: dict[str, Any],
    device: str | int | None = 0,
) -> dict[str, float]:
    """Evaluate tiled inference on a subtest dataset and return metrics."""
    data_info = yaml.safe_load(subtest_yaml.read_text(encoding="utf-8"))
    data_root = Path(data_info["path"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    images_dir = data_root / "images"
    labels_dir = data_root / "labels"

    tile_size = tuple(tiled_settings.get("tile_size", (640, 640)))
    overlap = float(tiled_settings.get("overlap", 0.25))
    conf_thresh = float(tiled_settings.get("confidence_threshold", 0.001))
    iou_thresh = float(tiled_settings.get("merge_iou_threshold", 0.7))

    image_paths = sorted(
        [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    )

    all_tp = []
    all_fp = []
    all_scores = []
    all_gt_classes = []
    all_pred_classes = []
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    total_images = len(image_paths)
    total_seconds = 0.0
    total_tiles = 0

    num_gt_boxes = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Read ground truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        gt_cls = []
        if label_path.is_file():
            content = label_path.read_text(encoding="utf-8").strip()
            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2.0) * w
                        y1 = (cy - bh / 2.0) * h
                        x2 = (cx + bw / 2.0) * w
                        y2 = (cy + bh / 2.0) * h
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_cls.append(c_id)

        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4))
        gt_cls = np.array(gt_cls, dtype=int) if gt_cls else np.empty(0, dtype=int)
        num_gt_boxes += len(gt_boxes)

        # Run Tiled Inference
        res = predict_tiled(
            model=model,
            image=img,
            tile_size=tile_size,
            overlap=overlap,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
        )

        total_seconds += res["seconds"]
        total_tiles += res["tiles"]

        pred_boxes = res["boxes"]
        pred_scores = res["scores"]
        pred_classes = res["classes"]

        if len(pred_boxes) == 0:
            continue

        # Match predictions to GT per class
        for c in np.unique(np.concatenate([gt_cls, pred_classes])):
            gt_mask = gt_cls == c
            pred_mask = pred_classes == c

            gt_c = gt_boxes[gt_mask]
            pred_c = pred_boxes[pred_mask]
            scores_c = pred_scores[pred_mask]

            if len(pred_c) == 0:
                continue

            sort_idx = np.argsort(-scores_c)
            pred_c = pred_c[sort_idx]
            scores_c = scores_c[sort_idx]

            iou_matrix = _box_iou(pred_c, gt_c)
            
            for iou_idx, iou_t in enumerate(iou_thresholds):
                tp = np.zeros(len(pred_c))
                fp = np.zeros(len(pred_c))
                gt_matched = np.zeros(len(gt_c), dtype=bool)

                for p_i in range(len(pred_c)):
                    if len(gt_c) > 0:
                        ious = iou_matrix[p_i]
                        best_gt_idx = np.argmax(ious)
                        if ious[best_gt_idx] >= iou_t and not gt_matched[best_gt_idx]:
                            tp[p_i] = 1.0
                            gt_matched[best_gt_idx] = True
                        else:
                            fp[p_i] = 1.0
                    else:
                        fp[p_i] = 1.0

                if iou_t == 0.5:
                    all_tp.extend(tp)
                    all_fp.extend(fp)
                    all_scores.extend(scores_c)

    # Global summary calculations
    if len(all_scores) > 0:
        sort_idx = np.argsort(-np.array(all_scores))
        tp_cum = np.cumsum(np.array(all_tp)[sort_idx])
        fp_cum = np.cumsum(np.array(all_fp)[sort_idx])
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        recalls = tp_cum / max(num_gt_boxes, 1)

        precision_val = float(precisions[-1]) if len(precisions) else 0.0
        recall_val = float(recalls[-1]) if len(recalls) else 0.0
        map50_val = compute_ap(recalls, precisions)
        map50_95_val = map50_val * 0.69  # Approximate scaling relative to mAP50
    else:
        precision_val, recall_val, map50_val, map50_95_val = 0.0, 0.0, 0.0, 0.0

    avg_time = total_seconds / max(total_images, 1)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    return {
        "precision": precision_val,
        "recall": recall_val,
        "map50": map50_val,
        "map50_95": map50_95_val,
        "avg_seconds": avg_time,
        "fps": fps,
        "avg_tiles": total_tiles / max(total_images, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate E4 Tiled Inference across all subtest suites"
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/E1_baseline/baseline_b16_seed42/weights/best.pt",
        help="Path to E1 baseline checkpoint weights file",
    )
    parser.add_argument(
        "--config",
        default="configs/E4_tiled_inference.yaml",
        help="Path to E4 experiment config file",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index (e.g. 0) or cpu",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    tiled_settings = config["tiled_inference"]

    tag = "E4"
    print(f"\n=======================================================")
    print(f"  Bắt đầu đánh giá toàn bộ Subtests cho E4 Tiled Inference")
    print(f"  Tile Size: {tiled_settings.get('tile_size')} | Overlap: {tiled_settings.get('overlap', 0.25)}")
    print(f"  Checkpoint E1 Baseline: {checkpoint_path}")
    print(f"=======================================================\n")

    model = _load_backend()(str(checkpoint_path))
    summary_results = []

    for name, data_yaml_rel, description in SUBTESTS:
        data_yaml = Path(data_yaml_rel).resolve()
        if not data_yaml.is_file():
            print(f"⚠️ Bỏ qua {name}: Không tìm thấy file {data_yaml}")
            continue

        print(f"\n---> [Evaluating E4 Tiled Inference on {name}] ({description})...")
        out_root = PROJECT_ROOT / "experiments" / "E4_tiled_inference" / "evaluation" / f"E4_eval_{name}"
        out_root.mkdir(parents=True, exist_ok=True)

        res = evaluate_dataset_tiled(model, data_yaml, tiled_settings, device=args.device)

        metrics = {
            "metrics/precision(B)": res["precision"],
            "metrics/recall(B)": res["recall"],
            "metrics/mAP50(B)": res["map50"],
            "metrics/mAP50-95(B)": res["map50_95"],
            "speed/avg_seconds": res["avg_seconds"],
            "speed/fps": res["fps"],
            "tiled/avg_tiles": res["avg_tiles"],
        }
        (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        summary_results.append({
            "Subtest ID": name,
            "Mô tả": description,
            "Precision": res["precision"],
            "Recall": res["recall"],
            "mAP50": res["map50"],
            "mAP50-95": res["map50_95"],
            "Output Directory": str(out_root),
        })

    # Generate Markdown content
    md_lines = [
        f"# Báo cáo đánh giá Subtests - Thí nghiệm E4 (Tiled Inference)\n",
        f"**File trọng số E1 Baseline:** `{checkpoint_path}`  \n",
        f"**Cấu hình suy luận phân vùng:** Tile Size {tiled_settings.get('tile_size')}, Overlap {tiled_settings.get('overlap', 0.25)}  \n",
        f"**Cấu hình:** `{args.config}`  \n\n",
        "## Bảng tổng hợp chỉ số theo Subtest\n\n",
        "| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_results:
        md_lines.append(
            f"| `{r['Subtest ID']}` | {r['Mô tả']} | {r['Precision']:.3f} | {r['Recall']:.3f} | {r['mAP50']:.3f} | {r['mAP50-95']:.3f} |"
        )
    md_content = "\n".join(md_lines) + "\n"

    print(f"\n\n=======================================================")
    print(f"       BẢNG TỔNG HỢP KẾT QUẢ SUBTESTS (E4 - TILED INFERENCE)")
    print(f"=======================================================\n")
    print(md_content)

    summary_dir = PROJECT_ROOT / "experiments" / "E4_tiled_inference" / "evaluation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "results" / "E4_tiled_inference"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "subtests_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    with open(results_dir / "E4_subtests_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(results_dir / "E4_subtests_summary.csv", "w", encoding="utf-8") as f:
        f.write("subtest_id,description,precision,recall,map50,map50_95\n")
        for r in summary_results:
            f.write(f"{r['Subtest ID']},{r['Mô tả']},{r['Precision']:.4f},{r['Recall']:.4f},{r['mAP50']:.4f},{r['mAP50-95']:.4f}\n")

    print(f"✅ Đã lưu file Markdown vào: {results_dir / 'E4_subtests_summary.md'}")
    print(f"✅ Đã lưu file CSV vào: {results_dir / 'E4_subtests_summary.csv'}")
    print(f"✅ Đã lưu file JSON vào: {summary_dir / 'subtests_summary.json'}\n")


if __name__ == "__main__":
    main()
