# Báo cáo quá trình và kết quả huấn luyện Thí nghiệm Baseline E1 (Batch Size 16)

**Mô hình:** YOLOv10n (640x640)  
**Mục đích:** Thiết lập mốc so sánh cơ sở (Baseline) chuẩn theo `batch: 16` phục vụ đối chiếu công bằng với các thí nghiệm E2–E5  
**Thời gian thực hiện:** 16/09/2026 (17:00:52 - 18:16:44 UTC, tổng ~1 giờ 15 phút)  
**Run ID:** `baseline_b16_seed42`  

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
- **Kích thước Batch (`batch`):** 16 *(Đã cập nhật để tăng tốc độ huấn luyện)*
- **Optimizer:** SGD (`lr0`: 0.01, `lrf`: 0.01, `momentum`: 0.937, `weight_decay`: 0.0005)
- **Tự động ép kiểu (`amp`):** True (FP16)
- **Random Seed:** 42 (`deterministic`: true)
- **Pretrained weights:** `yolov10n.pt`

---

## 2. Phân bố Dữ liệu (Dataset Distribution)

Thống kê tập dữ liệu đã qua chuẩn hóa (`data/processed`):

| Tập dữ liệu (Split) | Số lượng ảnh | Số lượng đối tượng (Objects) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **Train** | 7,180 | 25,751 | 7,178 ảnh có nhãn, 2 ảnh nền |
| **Validation** | 1,541 | 5,545 | 100% ảnh có nhãn |
| **Test** | 1,533 | 5,625 | Khóa dùng cho đánh giá cuối |
| **Tổng cộng** | **10,254** | **36,921** | 2 nhãn: `0: helmet`, `1: head` |

---

## 3. Kết quả đánh giá trong quá trình huấn luyện (Validation Results - `best.pt`)

Đánh giá tự động với trọng số tốt nhất (`best.pt`) thu được từ epoch tối ưu trên tập Validation:

### Báo cáo chi tiết theo từng nhãn

| Lớp (Class) | Mẫu (Instances) | Precision (Box P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tổng thể (all)** | **5,545** | **0.933** (93.3%) | **0.887** (88.7%) | **0.943** (94.3%) | **0.650** (65.0%) |
| 🪖 **`helmet`** | 4,337 | **0.945** (94.5%) | **0.926** (92.6%) | **0.968** (96.8%) | **0.676** (67.6%) |
| 🧑 **`head`** | 1,208 | **0.921** (92.1%) | **0.848** (84.8%) | **0.918** (91.8%) | **0.624** (62.4%) |

### Tốc độ suy luận (Inference Speed trên V100 GPU)
- **Tiền xử lý (Preprocess):** 0.1 ms / ảnh
- **Suy luận (Inference):** 1.2 ms / ảnh
- **Hậu xử lý (Postprocess):** 0.1 ms / ảnh
- **Tốc độ ước tính:** **~714 FPS**

---

## 4. Phân tích & Nhận xét

1. **Hiệu năng khi tăng Batch size lên 16:**
   - Khi tăng kích thước batch từ `batch: 4` lên `batch: 16`, chỉ số mAP50 trên tập Validation duy trì ở mức cao và đạt **94.3%** (so với 94.1% ở batch 4).
   - Precision tổng thể đạt **93.3%** và Recall đạt **88.7%**.

2. **Cải thiện thời gian huấn luyện:**
   - Việc tăng `batch: 16` đã giúp thời gian train 100 epoch giảm từ **2 giờ 33 phút** xuống chỉ còn **1 giờ 15 phút** (tiết kiệm hơn 50% thời gian).

3. **Tính chuẩn hóa cho toàn bộ thí nghiệm:**
   - Kết quả lượt train E1 (`batch: 16`) này trở thành **mốc Baseline chính thức** để đối chiếu với E2 (`batch: 16`) và các thí nghiệm E3, E4, E5 tiếp theo dưới cùng một protocol công bằng.

---

## 5. Danh mục Lưu trữ File Artifacts

Tất cả kết quả của đợt train E1 (`batch: 16`) này được lưu trữ tại: [`experiments/E1/baseline_b16_seed42/`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42)

- **Trọng số:** [`weights/best.pt`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/weights/best.pt), [`weights/last.pt`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/weights/last.pt)
- **Bảng chỉ số:** [`results.csv`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/results.csv), [`metadata.json`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/metadata.json), [`dataset_report.json`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/dataset_report.json)
- **Biểu đồ thị giác:** [`results.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/results.png), [`confusion_matrix.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/confusion_matrix.png), [`PR_curve.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/PR_curve.png), [`F1_curve.png`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/F1_curve.png)
