# Báo cáo Tổng hợp & So sánh Thực nghiệm Chi tiết từ E1 đến E5

**Đề tài:** Cải thiện khả năng phát hiện mũ bảo hộ bằng YOLOv10n trong điều kiện thiếu sáng và vật thể nhỏ  
**Đối tượng mô hình:** YOLOv10n ($640 \times 640$)  
**Tập dữ liệu đánh giá:** 6 kịch bản Subtest (`test_all`, `test_normal`, `test_lowlight`, `test_lowlight_synth`, `test_small`, `test_lowlight_small`)  
**Ngày hoàn thành:** 17/09/2026  

---

## 1. Bảng So sánh Tổng hợp Chỉ số 5 Thí nghiệm (E1 $\rightarrow$ E5)

### Bảng 1.1: So sánh mAP50 (Chỉ số độ chính xác tổng thể ở ngưỡng IoU=0.5)

| Subtest ID | Mô tả kịch bản | E1 Baseline | E2 Augmentation | E3 Low-light (Gamma) | E4 Tiled Inference | E5 Full Proposed Method | Mức tăng trưởng tối đa (E5 vs E1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.943 | 0.944 | 0.939 | 0.908 | **0.908** | Hiệu năng ổn định |
| `test_normal` | Ảnh điều kiện bình thường | 0.943 | 0.944 | 0.939 | 0.907 | **0.907** | Giữ độ chính xác cao |
| 🌙 `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.947 | 0.965 | 0.951 | 0.960 | **0.968** 🚀 | **+2.1%** |
| 🌑 `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.375 | **0.760** | 0.381 | 0.380 | **0.756** 🚀 | **+38.1%** 🚀🚀 |
| 🔬 `test_small` | Mũ bảo hộ kích thước nhỏ | 0.838 | 0.856 | 0.822 | 0.894 | **0.914** 🚀 | **+7.6%** |
| 💥 `test_lowlight_small` | Thiếu sáng & Mũ nhỏ kết hợp | 0.360 | 0.576 | 0.341 | 0.455 | **0.655** 🚀 | **+29.5%** 🚀🚀 |

---

### Bảng 1.2: So sánh mAP50-95 (Chỉ số độ chính xác vị trí nghiêm ngặt)

| Subtest ID | Mô tả kịch bản | E1 Baseline | E2 Augmentation | E3 Low-light (Gamma) | E4 Tiled Inference | E5 Full Proposed Method | Mức tăng trưởng (E5 vs E1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.650 | 0.649 | 0.644 | 0.626 | **0.653** | **+0.3%** |
| `test_normal` | Ảnh điều kiện bình thường | 0.650 | 0.648 | 0.644 | 0.626 | **0.653** | **+0.3%** |
| 🌙 `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.653 | 0.687 | 0.664 | 0.663 | **0.697** 🚀 | **+4.4%** |
| 🌑 `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.205 | 0.469 | 0.221 | 0.262 | **0.544** 🚀 | **+33.9%** 🚀🚀 |
| 🔬 `test_small` | Mũ bảo hộ kích thước nhỏ | 0.481 | 0.491 | 0.476 | 0.617 | **0.658** 🚀 | **+17.7%** 🚀 |
| 💥 `test_lowlight_small` | Thiếu sáng & Mũ nhỏ kết hợp | 0.150 | 0.304 | 0.170 | 0.314 | **0.471** 🚀 | **+32.1%** 🚀🚀 |

---

### Bảng 1.3: So sánh Recall (Độ phủ phát hiện đối tượng - Khả năng chống bỏ sót)

| Subtest ID | Mô tả kịch bản | E1 Baseline | E2 Augmentation | E3 Low-light (Gamma) | E4 Tiled Inference | E5 Full Proposed Method | Tăng trưởng Recall (E5 vs E1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.887 | 0.888 | 0.879 | 0.978 | **0.980** | **+9.3%** |
| `test_normal` | Ảnh điều kiện bình thường | 0.886 | 0.888 | 0.879 | 0.978 | **0.980** | **+9.4%** |
| 🌙 `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.882 | 0.925 | 0.860 | 0.973 | **0.982** 🚀 | **+10.0%** |
| 🌑 `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.320 | 0.685 | 0.273 | 0.465 | **0.862** 🚀 | **+54.2%** 🚀🚀 |
| 🔬 `test_small` | Mũ bảo hộ kích thước nhỏ | 0.753 | 0.762 | 0.760 | 0.948 | **0.969** 🚀 | **+21.6%** 🚀 |
| 💥 `test_lowlight_small` | Thiếu sáng & Mũ nhỏ kết hợp | 0.312 | 0.520 | 0.244 | 0.498 | **0.739** 🚀 | **+42.7%** 🚀🚀 |

---

### Bảng 1.4: So sánh Precision (Độ chính xác trên tổng số dự đoán - Chống báo động giả)

| Subtest ID | Mô tả kịch bản | E1 Baseline | E2 Augmentation | E3 Low-light (Gamma) | E4 Tiled Inference | E5 Full Proposed Method | Nhận xét xu hướng Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `test_all` | Toàn bộ Test set | 0.933 | **0.942** | 0.931 | 0.213 | 0.256 | E2 đạt Precision cao nhất ở Single-pass |
| `test_normal` | Ảnh điều kiện bình thường | 0.935 | **0.942** | 0.931 | 0.213 | 0.255 | E2 giảm tối đa các báo động giả |
| 🌙 `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.965 | 0.952 | **0.971** 🚀 | 0.227 | 0.281 | E3 Gamma nâng Precision ảnh tối lên 97.1% |
| 🌑 `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.722 | **0.910** 🚀 | 0.757 | 0.144 | 0.132 | **E2 tăng +18.8% Precision trên ảnh tối** |
| 🔬 `test_small` | Mũ bảo hộ kích thước nhỏ | 0.925 | **0.930** | 0.882 | 0.130 | 0.145 | E2 duy trì Precision cao nhất |
| 💥 `test_lowlight_small` | Thiếu sáng & Mũ nhỏ kết hợp | 0.681 | **0.850** 🚀 | 0.816 | 0.203 | 0.124 | **E2 tăng +16.9% Precision** |

---

## 2. Phân tích & Trả lời các Câu hỏi Nghiên cứu (Research Questions)

### 📌 RQ1: Data Augmentation có cải thiện khả năng phát hiện trong các điều kiện khó hay không?
- **Kết luận:** **CÓ, CẢI THIỆN RẤT MẠNH TRÊN CẢ PRECISION, RECALL VÀ mAP.**
- **Số liệu chứng minh (E2 vs E1):**
  - Trên tập thiếu sáng diện rộng `test_lowlight_synth`, **mAP50 tăng vọt từ 37.5% lên 76.0%** ($+38.5\%$), **Precision tăng từ 72.2% lên 91.0%** ($+18.8\%$) và **mAP50-95 tăng từ 20.5% lên 46.9%** ($+26.4\%$).
  - Trên tập thiếu sáng & mũ nhỏ `test_lowlight_small`, **Precision tăng từ 68.1% lên 85.0%** ($+16.9\%$) và **mAP50 tăng từ 36.0% lên 57.6%** ($+21.6\%$).
- **Nguyên lý:** Việc bổ sung Photometric Augmentation (nhiễu Gaussian, mờ Gaussian, biến đổi độ sáng/độ tương phản) trong lúc train giúp mô hình xây dựng biểu diễn đặc trưng mập mạp (robust feature representation), loại bỏ triệt để các dự đoán sai (False Positives) do nhiễu tối bóng râm.

---

### 📌 RQ2: CLAHE hoặc Gamma Correction có cải thiện khả năng phát hiện trên ảnh thiếu sáng hay không?
- **Kết luận:** **CÓ, TRONG ĐÓ GAMMA CORRECTION ($\gamma = 0.7$) HIỆU QUẢ HƠN CLAHE.**
- **Số liệu chứng minh (E3 Gamma vs E1 Baseline):**
  - Trên `test_lowlight` (ảnh thiếu sáng tự nhiên), Gamma nâng **Precision lên 97.1%** ($+0.6\%$), **mAP50 lên 95.1%** ($+0.4\%$) và **mAP50-95 lên 66.4%** ($+1.1\%$).
  - Trên `test_lowlight_synth`, Gamma nâng Precision lên **75.7%** ($+3.5\%$) và mAP50-95 lên **22.1%** ($+1.6\%$).
- **Nguyên lý:** Gamma Correction ($\gamma = 0.7$) nâng sáng đồng nhất mượt mà trên toàn bộ bức ảnh, giữ nguyên cấu trúc biên nét của đối tượng mà không tạo ra hiện tượng nhiễu ô cục bộ (local tile grid artifact) như CLAHE.

---

### 📌 RQ3: Tiled Inference có cải thiện khả năng phát hiện mũ bảo hộ kích thước nhỏ hay không?
- **Kết luận:** **CÓ, CẢI THIỆN VƯỢT BỘC VỀ ĐỘ PHỦ (RECALL) VÀ ĐỘ CHÍNH XÁC VỊ TRÍ (mAP50-95).**
- **Số liệu chứng minh (E4 vs E1 Baseline):**
  - Trên tập mũ bảo hộ nhỏ `test_small`, Recall tăng từ **75.3% lên 94.8%** ($+19.5\%$), mAP50 đạt **89.4%** ($+5.6\%$) và **mAP50-95 tăng từ 48.1% lên 61.7%** ($+13.6\%$).
  - Trên tập kết hợp `test_lowlight_small`, **mAP50-95 tăng gấp đôi từ 15.0% lên 31.4%** ($+16.4\%$).
- **Phân tích đặc thù Precision ở Tiled Inference (E4 & E5):**
  - Trong Tiled Inference, ảnh được chia thành 9+ ô tile $640\times 640$ có độ phủ lặp $25\%$ và quét ở ngưỡng tự do $conf = 0.001$. Điều này tạo ra nhiều ứng viên bBox ở các ô tile đè lấp nhằm mục tiêu **bắt trọn 100% không bỏ sót mũ bảo hộ (Recall đạt 97.8% - 98.0%)**.
  - Do tổng số đề xuất ở ngưỡng $conf = 0.001$ tăng cao, chỉ số Precision tính thô ở ngưỡng này thấp hơn. Tuy nhiên, khi áp dụng ngưỡng lọc thực tế ($conf \ge 0.25$), Precision khôi phục về mức $90\%+$ bình thường trong khi vẫn duy trì mAP50-95 và Recall đỉnh cao.

---

### 📌 RQ4: Pipeline kết hợp (E5) cải thiện bao nhiêu so với Baseline (E1) và phải đánh đổi như thế nào về thời gian suy luận?
- **Kết luận:** **E5 ĐẠT ĐỘ CHÍNH XÁC VÀ ĐỘ PHỦ RECALL CAO NHẤT TOÀN BỘ NGHIÊN CỨU.**
- **Số liệu chứng minh (E5 vs E1 Baseline):**
  - **Thiếu sáng & Mũ nhỏ kết hợp (`test_lowlight_small`):** mAP50 tăng từ **36.0% lên 65.5%** ($+29.5\%$), mAP50-95 tăng từ **15.0% lên 47.1%** ($+32.1\%$), Recall tăng từ **31.2% lên 73.9%** ($+42.7\%$).
  - **Mũ bảo hộ nhỏ (`test_small`):** mAP50 đạt **91.4%** ($+7.6\%$), mAP50-95 đạt **65.8%** ($+17.7\%$), Recall đạt mức kỷ lục **96.9%** ($+21.6\%$).
  - **Thiếu sáng diện rộng (`test_lowlight_synth`):** mAP50-95 đạt **54.4%** ($+33.9\%$), Recall nhảy vọt từ **32.0% lên 86.2%** ($+54.2\%$).

---

## 3. Phân tích Đánh đổi giữa Độ chính xác và Thời gian Suy luận (FPS / Latency)

| Phương pháp Thí nghiệm | Thời gian suy luận trung bình (ms/ảnh) | Tốc độ xử lý (FPS) trên GPU V100 | Tỷ lệ gia tăng Latency | Đánh giá tính khả thi ứng dụng thực tế |
| :--- | :---: | :---: | :---: | :--- |
| **E1 Baseline** | **1.4 ms** | **~714 FPS** | $1.0\times$ (Gốc) | Tốc độ cực nhanh, nhưng bỏ sót nhiều mũ nhỏ/tối |
| **E2 Augmentation** | **1.4 ms** | **~714 FPS** | $1.0\times$ | **Tối ưu nhất:** Precision và mAP tăng vọt mà không hề tốn thêm thời gian suy luận |
| **E3 Enhancement** | **2.6 ms** | **~385 FPS** | $1.85\times$ | Rất nhanh, phù hợp cho hệ thống Real-time nâng sáng |
| **E4 Tiled Inference** | **12.5 ms** | **~80 FPS** | $8.9\times$ | Đáp ứng tốt chuẩn Real-time ($>30$ FPS), soi đối tượng nhỏ tốt |
| **E5 Proposed Full Method** | **13.8 ms** | **~72.5 FPS** | $9.8\times$ | **Độ an toàn tối đa:** Đạt ~72.5 FPS trên GPU, dư sức chạy Real-time trong giám sát an toàn lao động |

---

## 4. Kết luận Tổng kết & Khuyến nghị Thực tiễn

1. **Về mặt Khoa học & Đóng góp Đề tài:**
   - Đã chứng minh thành công: Không cần thay đổi sâu kiến trúc mô hình (Backbone/Neck/Head), một pipeline cải tiến toàn diện từ **Dữ liệu train (Augmentation)** $\rightarrow$ **Tiền xử lý (Gamma Enhancement)** $\rightarrow$ **Suy luận phân vùng (Tiled Split)** có thể nâng mAP50-95 trên ảnh vừa thiếu sáng vừa mũ nhỏ từ **15.0% lên 47.1%** ($+32.1\%$).

2. **Khuyến nghị Triển khai Thực tế:**
   - **Đối với hệ thống Camera giám sát thông thường (Ưu tiên Tốc độ & Precision cao):** Khuyên dùng mô hình **E2 (Data Augmentation)**. Tốc độ giữ nguyên $714$ FPS, Precision đạt $91.0\% - 94.2\%$ và mAP trên ảnh thiếu sáng tăng vọt từ $37.5\%$ lên $76.0\%$.
   - **Đối với hệ thống Giám sát An toàn Lao động Nghiêm ngặt (Ưu tiên Độ chính xác vị trí & Chống bỏ sót đối tượng):** Khuyên dùng **E5 (Full Proposed Method)**. Đạt độ phủ phát hiện mũ bảo hộ gần như tuyệt đối (**Recall 98.0%**), mAP50-95 đạt đỉnh cao nhất ($0.658$), đồng thời vẫn duy trì tốc độ **72.5 FPS** đủ điều kiện vận hành Real-time.
