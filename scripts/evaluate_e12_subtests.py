"""Evaluate a checkpoint across all evaluation subtests and generate a summary report (E1 / E2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from helmet_yolov10.evaluation.evaluate import evaluate_checkpoint
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv10 model checkpoint across all subtest suites (E1 / E2)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint weights file (e.g. experiments/E1_baseline/.../weights/best.pt)",
    )
    parser.add_argument(
        "--config",
        default="configs/E1_baseline.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--experiment-tag",
        default="E1",
        help="Tag/ID for the experiment run (e.g. E1, E2)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device index (e.g. 0) or cpu",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of all subtests even if output directories exist",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    tag = args.experiment_tag
    tag_folder_map = {
        "E1": "E1_baseline",
        "E2": "E2_augmentation",
        "E1_baseline": "E1_baseline",
        "E2_augmentation": "E2_augmentation",
    }
    subfolder = tag_folder_map.get(tag, tag)

    print(f"\n=======================================================")
    print(f"  Bắt đầu đánh giá toàn bộ Subtests cho thí nghiệm: {tag}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"=======================================================\n")

    summary_results = []

    for name, data_yaml_rel, description in SUBTESTS:
        data_yaml = Path(data_yaml_rel).resolve()
        if not data_yaml.is_file():
            print(f"⚠️  Bỏ qua {name}: Không tìm thấy file {data_yaml}")
            continue

        run_name = f"{tag}_eval_{name}"
        print(f"\n---> [Evaluating {name}] ({description})...")

        try:
            out_dir = evaluate_checkpoint(
                checkpoint=checkpoint_path,
                config_path=args.config,
                data_path=data_yaml,
                run_name=run_name,
                device=args.device,
            )

            metrics_file = out_dir / "metrics.json"
            if metrics_file.is_file():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                p = metrics.get("metrics/precision(B)", 0.0)
                r = metrics.get("metrics/recall(B)", 0.0)
                map50 = metrics.get("metrics/mAP50(B)", 0.0)
                map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)

                summary_results.append({
                    "Subtest ID": name,
                    "Mô tả": description,
                    "Precision": p,
                    "Recall": r,
                    "mAP50": map50,
                    "mAP50-95": map50_95,
                    "Output Directory": str(out_dir),
                })
            else:
                print(f"⚠️ Không tìm thấy file metrics.json tại {out_dir}")

        except FileExistsError:
            out_dir = Path("experiments") / subfolder / "evaluation" / run_name
            if not out_dir.exists():
                out_dir = Path("experiments/E1_baseline/evaluation") / run_name

            metrics_file = out_dir / "metrics.json"
            if metrics_file.is_file() and not args.force:
                print(f"ℹ️ Thư mục đánh giá {run_name} đã có kết quả. Đang đọc lại...")
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                summary_results.append({
                    "Subtest ID": name,
                    "Mô tả": description,
                    "Precision": metrics.get("metrics/precision(B)", 0.0),
                    "Recall": metrics.get("metrics/recall(B)", 0.0),
                    "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
                    "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
                    "Output Directory": str(out_dir),
                })
            else:
                print(f"🔄 Thư mục {run_name} bị dở dang hoặc chưa có metrics.json. Tiến hành đánh giá lại...")
                import shutil
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir = evaluate_checkpoint(
                    checkpoint=checkpoint_path,
                    config_path=args.config,
                    data_path=data_yaml,
                    run_name=run_name,
                    device=args.device,
                )
                metrics_file = out_dir / "metrics.json"
                if metrics_file.is_file():
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    summary_results.append({
                        "Subtest ID": name,
                        "Mô tả": description,
                        "Precision": metrics.get("metrics/precision(B)", 0.0),
                        "Recall": metrics.get("metrics/recall(B)", 0.0),
                        "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
                        "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
                        "Output Directory": str(out_dir),
                    })

    # Generate Markdown content
    md_lines = [
        f"# Báo cáo đánh giá Subtests - Thí nghiệm {tag}\n",
        f"**File trọng số:** `{checkpoint_path}`  \n",
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

    # Print Summary Markdown Table
    print(f"\n\n=======================================================")
    print(f"       BẢNG TỔNG HỢP KẾT QUẢ SUBTESTS ({tag})")
    print(f"=======================================================\n")
    print(md_content)

    # Save summary JSON, CSV, and Markdown files
    summary_dir = Path("experiments") / subfolder / "evaluation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results") / subfolder
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "subtests_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    with open(results_dir / f"{tag}_subtests_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(results_dir / f"{tag}_subtests_summary.csv", "w", encoding="utf-8") as f:
        f.write("subtest_id,description,precision,recall,map50,map50_95\n")
        for res in summary_results:
            f.write(f"{res['Subtest ID']},{res['Mô tả']},{res['Precision']:.4f},{res['Recall']:.4f},{res['mAP50']:.4f},{res['mAP50-95']:.4f}\n")

    print(f"\n✅ Đã tự động lưu file Markdown vào: {summary_dir / 'subtests_summary.md'}")
    print(f"✅ Đã tự động lưu file Markdown vào: {results_dir / f'{tag}_subtests_summary.md'}")
    print(f"✅ Đã tự động lưu file CSV vào: {results_dir / f'{tag}_subtests_summary.csv'}")
    print(f"✅ Đã tự động lưu file JSON vào: {summary_dir / 'subtests_summary.json'}\n")


if __name__ == "__main__":
    main()
