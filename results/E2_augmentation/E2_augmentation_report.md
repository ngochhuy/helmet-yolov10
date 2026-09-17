# Báo cáo quá trình và kết quả huấn luyện Thí nghiệm E2 (Data Augmentation)

**Mô hình:** YOLOv10n (640x640)  
**Mục đích:** Đánh giá ảnh hưởng của tăng cường dữ liệu thực tế (Train-only Photometric Augmentation) lên khả năng phát hiện mũ bảo hộ  
**Thời gian thực hiện:** 16/09/2026 (21:46:12 - 23:05:45 GMT+7, tổng ~1 giờ 19 phút)  
**Run ID:** `augmentation_b16_seed42`  

---

## 1. Môi trường & Cấu hình huấn luyện

### Phần cứng & Thư viện
- **GPU:** NVIDIA Tesla V100-SXM2-16GB
- **Hệ điều hành:** Linux 6.8.0-138-generic x86_64
- **Môi trường Python:** Python 3.12.11
- **Thư viện chính:** PyTorch `2.5.1+cu121`, Ultralytics `8.1.34`, NumPy `2.2.6`

### Tham số cấu hình chính (`E2_augmentation.yaml`)
- **Kiến trúc mô hình:** YOLOv10n (2,695,196 tham số, 8.2 GFLOPs)
- **Kích thước đầu vào (`imgsz`):** 640 × 640
- **Số Epoch (`epochs`):** 100
- **Kích thước Batch (`batch`):** 16
- **Optimizer:** SGD (`lr0`: 0.01, `lrf`: 0.01, `momentum`: 0.937, `weight_decay`: 0.0005)
- **Tự động ép kiểu (`amp`):** True (FP16)
- **Random Seed:** 42 (`deterministic`: true)
- **Pretrained weights:** `yolov10n.pt`

### Các tham số Tăng cường dữ liệu (Augmentation Pipeline - Chỉ áp dụng cho tập Train)
- **Chỉnh độ sáng & độ tương phản (Brightness/Contrast):**
  - Xác suất xuất hiện (`probability`): `0.40` (40% số ảnh train)
  - Giới hạn độ sáng (`brightness_limit`): `±0.20`
  - Giới hạn độ tương phản (`contrast_limit`): `±0.15`
- **Nhiễu Gaussian (Gaussian Noise):**
  - Xác suất xuất hiện (`probability`): `0.15`
  - Biên độ sigma (`sigma_min` - `sigma_max`): `5.0` đến `15.0`
- **Làm mờ Gaussian (Gaussian Blur):**
  - Xác suất xuất hiện (`probability`): `0.15`
  - Kích thước kernel (`kernel_min` - `kernel_max`): `3` đến `5`
  - Độ lệch chuẩn (`sigma_min` - `sigma_max`): `0.1` đến `1.5`
- **Tăng cường hình học (Geometry & Mosaic):**
  - Dịch chuyển (`translate`): `0.15`
  - Tỷ lệ thu phóng (`scale`): `0.7`
  - Lật ngang (`fliplr`): `0.5`
  - Ghép ảnh Mosaic (`mosaic`): `1.0` (tắt ở `15` epoch cuối)

---

## 2. Phân bố Dữ liệu Huấn luyện & Kiểm định

| Tập dữ liệu (Split) | Số lượng ảnh | Số lượng đối tượng (Objects) | Tác động Augmentation |
| :--- | :---: | :---: | :--- |
| **Train** | 7,180 | 25,751 | Áp dụng đầy đủ Photometric & Geometric Augmentation |
| **Validation** | 1,541 | 5,545 | **Không áp dụng Augmentation** (Giữ nguyên ảnh gốc để đánh giá) |
| **Tổng cộng** | **8,721** | **31,296** | 2 nhãn: `0: helmet`, `1: head` |

---

## 3. Kết quả đánh giá trong quá trình huấn luyện (Validation Results - `best.pt`)

Trong 100 epoch, mô hình đạt điểm Fitness tối ưu tại epoch **96** và được lưu tự động vào `best.pt`. Kết quả đánh giá trên tập Validation:

### Báo cáo chi tiết theo từng nhãn

| Lớp (Class) | Mẫu (Instances) | Precision (Box P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tổng thể (all)** | **5,545** | **0.942** (94.2%) | **0.888** (88.8%) | **0.945** (94.5%) | **0.658** (65.8%) |
| 🪖 **`helmet`** | 4,337 | **0.952** (95.2%) | **0.925** (92.5%) | **0.969** (96.9%) | **0.680** (68.0%) |
| 🧑 **`head`** | 1,208 | **0.932** (93.2%) | **0.851** (85.1%) | **0.921** (92.1%) | **0.635** (63.5%) |

### Tốc độ suy luận (Inference Speed trên V100 GPU)
- **Tiền xử lý (Preprocess):** 0.1 ms / ảnh
- **Suy luận (Inference):** 1.2 ms / ảnh
- **Hậu xử lý (Postprocess):** 0.1 ms / ảnh
- **Tốc độ:** **~714 FPS**

---

## 4. Phân tích & Nhận xét quá trình huấn luyện

1. **Tác động của Data Augmentation lên mô hình:**
   - Việc bổ sung Photometric Augmentation (nhiễu, mờ, biến đổi độ sáng/tương phản) giúp mô hình **tăng Precision** từ `0.931` (ở E1) lên **`0.942`** (ở E2).
   - Chỉ số **mAP50** trên tập validation tăng từ `0.941` lên **`0.945`** và **mAP50-95** tăng từ `0.655` lên **`0.658`**.
   - Cả 2 nhãn `helmet` và `head` đều ghi nhận sự tăng trưởng ổn định về độ chính xác Precision.

2. **Quá trình hội tụ (Convergence):**
   - Nhờ cờ `close_mosaic: 15`, ở 15 epoch cuối cùng (từ epoch 85 đến 100), khi tắt ghép ảnh Mosaic, loss trên tập Validation giảm sâu và các chỉ số mAP đạt đỉnh cao nhất ở epoch 96.

3. **Thời gian huấn luyện:**
   - Với `batch: 16` trên GPU Tesla V100, 100 epoch hoàn thành chỉ trong **1 giờ 19 phút** (nhanh hơn đáng kể so với `batch: 4` trước đó).

---

## 5. Danh mục Lưu trữ File Artifacts

Toàn bộ artifacts của lượt train E2 này được lưu trữ tại: [`experiments/E2/augmentation_b16_seed42/`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42)

- **Trọng số:** [`weights/best.pt`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/weights/best.pt), [`weights/last.pt`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/weights/last.pt)
- **Bảng chỉ số:** [`results.csv`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/results.csv), [`metadata.json`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/metadata.json), [`dataset_report.json`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/dataset_report.json)
- **Biểu đồ thị giác:** [`results.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/results.png), [`confusion_matrix.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/confusion_matrix.png), [`PR_curve.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/PR_curve.png), [`F1_curve.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/F1_curve.png)
