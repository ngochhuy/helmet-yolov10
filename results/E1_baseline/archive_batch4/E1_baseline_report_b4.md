# Báo cáo kết quả huấn luyện Thí nghiệm Baseline (E1 - Batch size 4)

**Mô hình:** YOLOv10n (640x640)  
**Mục đích:** Thiết lập mốc so sánh cơ sở (Baseline) cho bài toán phát hiện mũ bảo hộ (Lần train với batch=4)  
**Thời gian thực hiện:** 16/09/2026 (10:08:41 - 12:42:12 UTC, tổng ~2 giờ 33 phút)  
**Run ID:** `baseline_seed42_20260916T100841Z`  

---

## 1. Môi trường & Cấu hình huấn luyện

### Phần cứng & Thư viện
- **GPU:** NVIDIA Tesla V100-SXM2-16GB
- **Hệ điều hành:** Linux 6.8.0-138-generic x86_64
- **Môi trường Python:** Python 3.12.11
- **Thư viện chính:** PyTorch `2.5.1+cu121`, Ultralytics `8.1.34`, NumPy `2.2.6`

### Tham số cấu hình chính (`E1_baseline.yaml` / `base.yaml`)
- **Kiến trúc mô hình:** YOLOv10n (2,695,196 tham số, 8.2 GFLOPs)
- **Kích thước đầu vào (`imgsz`):** 640 × 640
- **Số Epoch (`epochs`):** 100
- **Kích thước Batch (`batch`):** 4
- **Optimizer:** SGD (`lr0`: 0.01, `lrf`: 0.01, `momentum`: 0.937, `weight_decay`: 0.0005)
- **Tự động ép kiểu (`amp`):** True (FP16)
- **Random Seed:** 42 (`deterministic`: true)
- **Pretrained weights:** `yolov10n.pt`

---

## 2. Kết quả đánh giá hiệu năng (Validation Results - `best.pt`)

| Lớp (Class) | Mẫu (Instances) | Precision (Box P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tổng thể (all)** | **5,545** | **0.931** (93.1%) | **0.890** (89.0%) | **0.941** (94.1%) | **0.655** (65.5%) |
| 🪖 **`helmet`** | 4,337 | **0.944** (94.4%) | **0.927** (92.7%) | **0.967** (96.7%) | **0.678** (67.8%) |
| 🧑 **`head`** | 1,208 | **0.918** (91.8%) | **0.854** (85.4%) | **0.914** (91.4%) | **0.631** (63.1%) |

---

## 3. Danh mục Lưu trữ File Artifacts

Tất cả kết quả của đợt train này được lưu trữ tại: `experiments/E1/baseline_seed42_20260916T100841Z/`
