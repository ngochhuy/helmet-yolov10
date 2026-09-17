"""Evaluate E3 Low-Light Enhancement across all evaluation subtests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import yaml

from helmet_yolov10.enhancement.pipeline import apply_enhancement
from helmet_yolov10.evaluation.evaluate import _as_serialisable_metrics
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


def stage_enhanced_subtest(
    data_yaml_path: Path,
    method: str,
    method_settings: dict[str, Any],
    stage_root: Path,
) -> Path:
    """Create a staged copy of a subtest dataset with low-light enhancement applied."""
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    source_root = Path(data["path"])
    if not source_root.is_absolute():
        source_root = PROJECT_ROOT / source_root

    source_images = source_root / "images"
    destination_images = stage_root / "images"
    destination_labels = stage_root / "labels"

    if stage_root.exists():
        shutil.rmtree(stage_root)

    destination_images.mkdir(parents=True, exist_ok=True)

    # Symlink labels
    if (source_root / "labels").exists():
        os.symlink(source_root / "labels", destination_labels, target_is_directory=True)

    # Enhance images
    for source_image in source_images.rglob("*"):
        if source_image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = source_image.relative_to(source_images)
        destination_image = destination_images / relative
        destination_image.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {source_image}")

        enhanced = apply_enhancement(image, method, method_settings)
        if not cv2.imwrite(str(destination_image), enhanced):
            raise IOError(f"Cannot write image: {destination_image}")

    staged_data = dict(data)
    staged_data["path"] = str(stage_root.resolve())
    staged_data["train"] = "images"
    staged_data["val"] = "images"
    staged_data["test"] = "images"

    staged_yaml = stage_root / "data.yaml"
    staged_yaml.write_text(yaml.safe_dump(staged_data, sort_keys=False), encoding="utf-8")
    return staged_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate E3 Low-Light Enhancement across all subtest suites"
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/E1_baseline/baseline_b16_seed42/weights/best.pt",
        help="Path to E1 baseline checkpoint weights file",
    )
    parser.add_argument(
        "--config",
        default="configs/E3_enhancement.yaml",
        help="Path to E3 experiment config file",
    )
    parser.add_argument(
        "--method",
        default="clahe",
        choices=["clahe", "gamma"],
        help="Enhancement method to evaluate (clahe or gamma)",
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
    method = args.method
    method_settings = config["enhancement"]["candidates"][method]

    tag = "E3"
    print(f"\n=======================================================")
    print(f"  Bắt đầu đánh giá toàn bộ Subtests cho E3 Low-Light Enhancement")
    print(f"  Phương pháp nâng sáng: {method.upper()} ({method_settings})")
    print(f"  Checkpoint E1 Baseline: {checkpoint_path}")
    print(f"=======================================================\n")

    model = _load_backend()(str(checkpoint_path))
    summary_results = []

    staged_base = PROJECT_ROOT / "experiments" / "E3_enhancement" / "staged_eval"

    for name, data_yaml_rel, description in SUBTESTS:
        data_yaml = Path(data_yaml_rel).resolve()
        if not data_yaml.is_file():
            print(f"⚠️ Bỏ qua {name}: Không tìm thấy file {data_yaml}")
            continue

        print(f"\n---> [Evaluating E3 ({method}) on {name}] ({description})...")
        stage_dir = staged_base / f"{name}_{method}"
        staged_yaml = stage_enhanced_subtest(data_yaml, method, method_settings, stage_dir)

        out_root = PROJECT_ROOT / "experiments" / "E3_enhancement" / "evaluation"
        run_name = f"E3_eval_{name}"

        result = model.val(
            data=str(staged_yaml),
            project=str(out_root),
            name=run_name,
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.7,
            device=args.device,
            exist_ok=True,
            verbose=False,
        )

        metrics = _as_serialisable_metrics(result)
        p = metrics.get("metrics/precision(B)", 0.0)
        r = metrics.get("metrics/recall(B)", 0.0)
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)

        summary_results.append(
            {
                "Subtest ID": name,
                "Mô tả": description,
                "Precision": p,
                "Recall": r,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "Output Directory": str(out_root / run_name),
            }
        )

    # Generate Markdown content
    md_lines = [
        f"# Báo cáo đánh giá Subtests - Thí nghiệm E3 (Low-light Enhancement: {method.upper()})\n",
        f"**File trọng số E1 Baseline:** `{checkpoint_path}`  \n",
        f"**Phương pháp nâng sáng:** `{method.upper()}` ({method_settings})  \n",
        f"**Cấu hình:** `{args.config}`  \n\n",
        "## Bảng tổng hợp chỉ số theo Subtest\n\n",
        "| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for res in summary_results:
        md_lines.append(
            f"| `{res['Subtest ID']}` | {res['Mô tả']} | {res['Precision']:.3f} | {res['Recall']:.3f} | {res['mAP50']:.3f} | {res['mAP50-95']:.3f} |"
        )
    md_content = "\n".join(md_lines) + "\n"

    print(f"\n\n=======================================================")
    print(f"       BẢNG TỔNG HỢP KẾT QUẢ SUBTESTS (E3 - {method.upper()})")
    print(f"=======================================================\n")
    print(md_content)

    # Save summary files with method name in filename
    summary_dir = PROJECT_ROOT / "experiments" / "E3_enhancement" / "evaluation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "results" / "E3_enhancement"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / f"subtests_summary_{method}.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    # Save method-specific and default files in results/E3_enhancement/
    report_md_filename = f"E3_{method}_subtests_summary.md"
    report_csv_filename = f"E3_{method}_subtests_summary.csv"

    with open(results_dir / report_md_filename, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(results_dir / "E3_subtests_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(results_dir / report_csv_filename, "w", encoding="utf-8") as f:
        f.write("subtest_id,description,precision,recall,map50,map50_95\n")
        for res in summary_results:
            f.write(
                f"{res['Subtest ID']},{res['Mô tả']},{res['Precision']:.4f},{res['Recall']:.4f},{res['mAP50']:.4f},{res['mAP50-95']:.4f}\n"
            )
    with open(results_dir / "E3_subtests_summary.csv", "w", encoding="utf-8") as f:
        f.write("subtest_id,description,precision,recall,map50,map50_95\n")
        for res in summary_results:
            f.write(
                f"{res['Subtest ID']},{res['Mô tả']},{res['Precision']:.4f},{res['Recall']:.4f},{res['mAP50']:.4f},{res['mAP50-95']:.4f}\n"
            )

    print(f"✅ Đã lưu file Markdown vào: {results_dir / report_md_filename}")
    print(f"✅ Đã lưu file CSV vào: {results_dir / report_csv_filename}")
    print(f"✅ Đã lưu file JSON vào: {summary_dir / f'subtests_summary_{method}.json'}\n")


if __name__ == "__main__":
    main()
