# Script Evaluation Guide & Overview

Bản hướng dẫn và so sánh quy trình hoạt động của 4 kịch bản đánh giá (`scripts/evaluate_*.py`) trong dự án nghiên cứu phát hiện mũ bảo hộ với YOLOv10n.

---

## 1. Bảng So sánh Tổng quan 4 Script Evaluation

| File Script | Phục vụ Thí nghiệm | Trọng số Mô hình được nạp (`best.pt`) | Tiền xử lý ảnh (Preprocessing) | Quy trình Suy luận (Inference Pipeline) | Câu hỏi Nghiên cứu (Research Questions) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **[`evaluate_e12_subtests.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/scripts/evaluate_e12_subtests.py)** | **E1 (Baseline)** & **E2 (Augmentation)** | • `E1_baseline/.../best.pt`<br>• `E2_augmentation/.../best.pt` | ❌ **Ảnh gốc RAW** (Không can thiệp độ sáng) | **Single-pass Baseline**<br>(Đưa trực tiếp ảnh $640\times 640$ vào mô hình) | • **E1:** Đặt mốc so sánh gốc.<br>• **E2:** Đo hiệu quả Data Augmentation (**RQ1**). |
| **[`evaluate_e3_subtests.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/scripts/evaluate_e3_subtests.py)** | **E3 (Low-light Enhancement)** | `E1_baseline/.../best.pt` | 🟢 **Nâng sáng ảnh**: CLAHE (kênh LAB L) hoặc Gamma Correction ($\gamma=0.7$) | **Single-pass Baseline**<br>(Đưa ảnh đã nâng sáng vào mô hình) | Đo hiệu quả nâng sáng ảnh vùng tối trước khi suy luận (**RQ2**). |
| **[`evaluate_e4_subtests.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/scripts/evaluate_e4_subtests.py)** | **E4 (Tiled Inference)** | `E1_baseline/.../best.pt` | ❌ **Ảnh gốc RAW** (Không nâng sáng) | 🟢 **Phân vùng suy luận**: Chia tile $640\times 640$ (overlap 25%) + Class-aware NMS | Đo hiệu quả soi phát hiện mũ bảo hộ kích thước nhỏ ở cự ly xa (**RQ3**). |
| **[`evaluate_e5_subtests.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/scripts/evaluate_e5_subtests.py)** | **E5 (Full Proposed Method)** | `E2_augmentation/.../best.pt` *(Weights dẻo dai nhất)* | 🟢 **Nâng sáng ảnh**: Gamma Correction ($\gamma=0.7$) | 🟢 **Phân vùng suy luận**: Chia tile $640\times 640$ (overlap 25%) + Class-aware NMS | Đánh giá **Phương pháp Đề xuất Hoàn chỉnh (Full Pipeline)** & Trade-off FPS (**RQ4**). |

---

## 2. Chi tiết Luồng Xử lý Dữ liệu (Data Pipeline Flow)

### 1️⃣ `evaluate_e12_subtests.py` (Thử nghiệm E1 & E2)
$$\text{Ảnh gốc (RAW)} \xrightarrow{} \text{Model YOLOv10} \xrightarrow{} \text{Bounding Box & Scores}$$
- **Mô tả:** Đánh giá độ chính xác tiêu chuẩn không có bất kỳ bước tiền xử lý hay chia tile nào. Dùng để đo hiệu năng gốc của mô hình E1 baseline và mô hình E2 (sau khi train với Augmentation).

### 2️⃣ `evaluate_e3_subtests.py` (Thử nghiệm E3)
$$\text{Ảnh gốc (RAW)} \xrightarrow{\text{Nâng sáng Gamma / CLAHE}} \text{Ảnh sáng} \xrightarrow{} \text{Model E1 Baseline} \xrightarrow{} \text{Bounding Box & Scores}$$
- **Mô tả:** Thêm bước biến đổi mượt dải sáng trước khi suy luận nhằm hỗ trợ mô hình nhận diện vật thể bị che khuất trong bóng tối.

### 3️⃣ `evaluate_e4_subtests.py` (Thử nghiệm E4)
$$\text{Ảnh gốc (RAW)} \xrightarrow{\text{Chia ô Tile 640x640}} \text{Các Tile} \xrightarrow{} \text{Model E1 Baseline} \xrightarrow{\text{Class-aware NMS}} \text{Bounding Box gộp}$$
- **Mô tả:** Cắt ảnh lớn thành các vùng nhỏ $640\times 640$ đè lấp $25\%$, suy luận từng vùng để không bỏ sót các mũ bảo hộ nhỏ ở xa, sau đó gộp các ô phát hiện bằng Class-aware NMS.

### 4️⃣ `evaluate_e5_subtests.py` (Thử nghiệm E5 - Full Proposed Method)
$$\text{Ảnh gốc (RAW)} \xrightarrow{\text{Nâng sáng Gamma}} \text{Ảnh sáng} \xrightarrow{\text{Chia ô Tile 640x640}} \text{Các Tile} \xrightarrow{} \text{Model E2 Weights} \xrightarrow{\text{Class-aware NMS}} \text{Bounding Box gộp}$$
- **Mô tả:** Tích hợp **cả 3 cải tiến mạnh nhất** (Trọng số mô hình E2 + Tiền xử lý nâng sáng Gamma E3 + Suy luận phân vùng Tiled Split E4) để đạt độ phủ Recall và mAP50-95 tối đa trên các kịch bản khó nhất.

---

## 3. Cú pháp Lệnh Thực thi Chuẩn (CLI Execution Commands)

```bash
# 1. Đánh giá E1 Baseline
uv run python scripts/evaluate_e12_subtests.py \
  --checkpoint experiments/E1_baseline/baseline_b16_seed42/weights/best.pt \
  --config configs/E1_baseline.yaml \
  --experiment-tag E1 --device 0

# 2. Đánh giá E2 Data Augmentation
uv run python scripts/evaluate_e12_subtests.py \
  --checkpoint experiments/E2_augmentation/augmentation_b16_seed42/weights/best.pt \
  --config configs/E2_augmentation.yaml \
  --experiment-tag E2 --device 0

# 3. Đánh giá E3 Low-light Enhancement (Gamma / CLAHE)
uv run python scripts/evaluate_e3_subtests.py --method gamma --device 0

# 4. Đánh giá E4 Tiled Inference
uv run python scripts/evaluate_e4_subtests.py --device 0

# 5. Đánh giá E5 Proposed Full Pipeline
uv run python scripts/evaluate_e5_subtests.py --device 0
```
