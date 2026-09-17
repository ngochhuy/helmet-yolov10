# Báo cáo Phương pháp Đề xuất Thí nghiệm E5 (Full Method Pipeline)

**Mô hình:** YOLOv10n (640x640) tích hợp Pipeline 3 thành phần  
**Mục đích:** Báo cáo thiết lập, nguyên lý hoạt động, cấu hình chi tiết và quy trình suy luận kết hợp (Data Augmentation + Low-light Enhancement + Tiled Inference) phục vụ giải quyết **RQ4**  
**Cấu hình chính:** [`configs/E5_full_method.yaml`](file:///home/nguyenhuynh/Documents/helmet-yolov10/configs/E5_full_method.yaml)  
**Mã nguồn suy luận:** [`src/helmet_yolov10/inference/full_method.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/src/helmet_yolov10/inference/full_method.py)  
**Notebook thực thi:** [`notebooks/E5_full_method_evaluation.ipynb`](file:///home/nguyenhuynh/Documents/helmet-yolov10/notebooks/E5_full_method_evaluation.ipynb)  

---

## 1. Tổng quan Nghiên cứu & Nguyên lý Pipeline E5

Thí nghiệm E5 đại diện cho **phương pháp đề xuất hoàn chỉnh (Full Proposed Method)** trong đề tài nghiên cứu. Pipeline kết hợp 3 thành phần cải tiến đã được kiểm chứng độc lập ở các thí nghiệm trước (E2, E3 và E4):

```text
[Ảnh Đầu Vào (Input Image)]
        │
        ▼
┌────────────────────────────────────────┐
│  Thành phần 1: Low-Light Enhancement  │  (E3 - Gamma Correction gamma=0.7)
│  Tăng cường độ sáng & độ tương phản   │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│  Thành phần 2: Tiled Inference Engine  │  (E4 - 640x640 Tile, overlap=25%)
│  Chia nhỏ ảnh & Suy luận phân vùng     │
└───────────────────┬────────────────────┘
                    │ (Sử dụng E2 Model Weights: augmentation_b16_seed42/best.pt)
                    ▼
┌────────────────────────────────────────┐
│  Thành phần 3: Class-Aware NMS Merge   │  (Gộp bBox trùng lặp giữa các Tile)
│  Tổng hợp Bounding Box & Conf Score    │
└───────────────────┬────────────────────┘
                    │
                    ▼
      [Kết quả Bounding Box & FPS/Latency]
```

### Chi tiết 3 Thành phần Tích hợp:
1. **Trọng số Mô hình (từ Thí nghiệm E2 - Data Augmentation):**
   - Sử dụng trọng số `best.pt` của mô hình E2 trained với Data Augmentation (nhiễu Gaussian, mờ Gaussian, biến đổi độ sáng/tương phản thực tế).
   - Đường dẫn trọng số: [`experiments/E2_augmentation/augmentation_b16_seed42/weights/best.pt`](file:///home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2_augmentation/augmentation_b16_seed42/weights/best.pt).

2. **Tiền xử lý nâng sáng (Low-light Enhancement - từ Thí nghiệm E3):**
   - Thuật toán được chọn từ kết quả thực nghiệm E3: **Gamma Correction ($\gamma = 0.7$)**.
   - Tham số: `gamma: 0.7`.
   - Lý do chọn: Kết quả thực nghiệm E3 chứng minh Gamma Correction vượt trội hơn CLAHE trên tất cả các tập ảnh thiếu sáng (`test_lowlight`, `test_lowlight_synth`, `test_lowlight_small`).
   - Tác dụng: Nâng cao độ tương phản chi tiết trên các vùng ảnh bị tối mà không làm cháy sáng hay gây méo màu.

3. **Suy luận theo vùng (Tiled Inference - từ Thí nghiệm E4):**
   - Kích thước vùng (Tile Size): $640 \times 640$.
   - Độ phủ lặp (Overlap): $25\%$ ($0.25$).
   - Thuật toán gộp Box: **Class-Aware Non-Maximum Suppression (NMS)** với ngưỡng IoU $= 0.7$, ngưỡng tin cậy Conf $= 0.001$.
   - Tác dụng: Cải thiện vượt bậc khả năng phát hiện mũ bảo hộ kích thước nhỏ ở khoảng cách xa hoặc góc khuất.

---

## 2. Thông số Cấu hình Hệ thống (`configs/E5_full_method.yaml`)

| Nhóm thông số | Tên tham số | Giá trị | Giải thích chi tiết |
| :--- | :--- | :---: | :--- |
| **Thí nghiệm** | `id` / `name` | `E5` / `full_method` | Định danh thí nghiệm pipeline đề xuất |
| **Mô hình** | `architecture` | `YOLOv10n` | Kiến trúc mô hình baseline YOLOv10n |
| | `weights` | `experiments/E2_augmentation/.../best.pt` | Trọng số tối ưu đã được train ở E2 |
| **Tiền xử lý (E3)** | `method` | `clahe` | Thuật toán tăng cường ảnh thiếu sáng |
| | `color_space` / `channel` | `LAB` / `L` | Xử lý trên kênh độ sáng (Luminance) |
| | `clip_limit` | `2.0` | Giới hạn cắt đỉnh histogram chống nhiễu |
| | `tile_grid_size` | `[8, 8]` | Kích thước ma trận ô cân bằng histogram |
| **Suy luận Vùng (E4)** | `tile_size` | `[640, 640]` | Kích thước mỗi tile đưa vào mô hình |
| | `overlap` | `0.25` (25%) | Tỷ lệ đè lấp giữa các tile kề nhau |
| | `confidence_threshold` | `0.001` | Ngưỡng tin cậy lọc bBox |
| | `merge_iou_threshold` | `0.7` | Ngưỡng IoU loại bỏ bBox trùng lặp |
| | `merge_method` | `class_aware_nms` | NMS lọc bBox tách biệt theo từng class ID |
| **Đầu ra** | `root` | `experiments/E5_full_method` | Thư mục lưu kết quả thí nghiệm |

---

## 3. Kiến trúc Code & Module triển khai

Pipeline E5 được lập trình theo chuẩn mô-đun hóa cao:

- **Module suy luận chính:** [`src/helmet_yolov10/inference/full_method.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/src/helmet_yolov10/inference/full_method.py)
  - Hàm `predict_full_method()` nhận ảnh BGR đầu vào và config tham số.
  - Tự động thực hiện xử lý 3 bước (Enhancement $\rightarrow$ Tiled Split $\rightarrow$ Model Forward $\rightarrow$ NMS Merge).
  - Đo chính xác tổng thời gian thực thi (Latency `total_seconds`) và tốc độ xử lý (`total_fps`).

- **Đánh giá tự động Subtests (E1/E2):** [`scripts/evaluate_e12_subtests.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/scripts/evaluate_e12_subtests.py)
  - Hỗ trợ cờ `--experiment-tag E5` để tự động chạy đánh giá trên 6 kịch bản subtest:
    1. `test_all`: Toàn bộ Test set
    2. `test_normal`: Ảnh điều kiện bình thường
    3. `test_lowlight`: Ảnh thiếu sáng tự nhiên
    4. `test_lowlight_synth`: Ảnh thiếu sáng tổng hợp diện rộng
    5. `test_small`: Mũ bảo hộ kích thước nhỏ
    6. `test_lowlight_small`: Ảnh kết hợp vừa thiếu sáng vừa mũ nhỏ

- **Kiểm thử đơn vị (Unit Tests):** [`tests/test_full_method.py`](file:///home/nguyenhuynh/Documents/helmet-yolov10/tests/test_full_method.py)
  - Kiểm định toàn bộ logic tích hợp, kiểm tra tính đúng đắn của hàm gộp box và đo thời gian (Đã pass 100%).

---

## 4. Mục tiêu Đánh giá & Trade-off dự kiến (Trả lời RQ4)

Thí nghiệm E5 sẽ giải quyết **Câu hỏi nghiên cứu RQ4**:
> *Pipeline kết hợp cải thiện bao nhiêu so me với baseline (E1) và phải đánh đổi như thế nào về thời gian suy luận (FPS/Latency)?*

### Các kỳ vọng lý thuyết & Trade-off:
1. **Độ chính xác (mAP50 & mAP50-95):**
   - Kỳ vọng đạt độ chính xác **cao nhất trong toàn bộ các thí nghiệm (E1-E5)** trên các tập subtest kết hợp khó như `test_lowlight_small` và `test_lowlight_synth`.
   - Kết hợp ưu điểm chống gãy nét/tối ảnh của CLAHE (E3) cùng khả năng soi góc xa của Tiled Inference (E4) và độ dẻo dai của weights trained Augmentation (E2).

2. **Chi phí tính toán & Tốc độ suy luận (Latency vs FPS):**
   - Chi phí suy luận sẽ tăng do ảnh bị chia làm nhiều Tile $640 \times 640$ (ví dụ: ảnh kích thước $1280 \times 1280$ sẽ tạo ra khoảng 9 tiles).
   - Sự đánh đổi này là hoàn toàn phù hợp trong các ứng dụng thực tế yêu cầu độ an toàn cao (Safety Helmet Compliance Detection) hơn là thời gian thực cực nhanh.

---

## 5. Danh mục Lưu trữ File Artifacts E5

- **File cấu hình:** [`configs/E5_full_method.yaml`](file:///home/nguyenhuynh/Documents/helmet-yolov10/configs/E5_full_method.yaml)
- **Báo cáo thiết kế phương pháp:** [`results/E5_full_method/E5_full_method_design_report.md`](file:///home/nguyenhuynh/Documents/helmet-yolov10/results/E5_full_method/E5_full_method_design_report.md)
- **Tóm tắt kết quả Subtests:** `results/E5_full_method/E5_subtests_summary.md` và `E5_subtests_summary.csv`
- **Notebook demo & benchmark:** [`notebooks/E5_full_method_evaluation.ipynb`](file:///home/nguyenhuynh/Documents/helmet-yolov10/notebooks/E5_full_method_evaluation.ipynb)
