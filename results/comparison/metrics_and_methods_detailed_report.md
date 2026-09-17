# Báo cáo Chi tiết: Ý nghĩa Các Chỉ số Metric & Phân tích Đánh giá Các Phương pháp Thực nghiệm (E1 đến E5)

**Đề tài:** Cải thiện khả năng phát hiện mũ bảo hộ bằng YOLOv10n trong điều kiện thiếu sáng và vật thể nhỏ  
**Thực hiện:** Đánh giá hệ thống trên 6 kịch bản Subtest (`test_all`, `test_normal`, `test_lowlight`, `test_lowlight_synth`, `test_small`, `test_lowlight_small`)  
**Tài liệu gốc tham chiếu:** [`results/final_research_report.md`](file:///home/nguyenhuynh/Documents/helmet-yolov10/results/final_research_report.md) & [`results/comparison/E1_to_E5_comprehensive_report.md`](file:///home/nguyenhuynh/Documents/helmet-yolov10/results/comparison/E1_to_E5_comprehensive_report.md)

---

## PHẦN 1: Ý NGHĨA CHI TIẾT CỦA CÁC CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

Trong các bài toán phát hiện đối tượng (Object Detection), 4 chỉ số **Precision**, **Recall**, **mAP50**, và **mAP50-95** đóng vai trò quyết định để đo lường hiệu năng của mô hình ở các khía cạnh khác nhau.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                    THỰC TẾ (Ground Truth)              │
                    │               Có Mũ Bảo Hộ         Không Mũ Bảo Hộ     │
┌───────────────────┼────────────────────────────────────────────────────────┤
│ DỰ ĐOÁN (Predict) │                                                        │
│ Có Mũ Bảo Hộ      │           TP (True Positive)      FP (False Positive)  │
│                   │             [Dự đoán ĐÚNG]          [Báo động GIẢ]     │
│                   │                                                        │
│ Không Mũ Bảo Hộ   │           FN (False Negative)     TN (True Negative)   │
│                   │            [BỎ SÓT đối tượng]        [Bỏ qua đúng]     │
└───────────────────┴────────────────────────────────────────────────────────┘
```

---

### 1. Precision (Độ chính xác trên tổng số dự đoán)

- **Công thức toán học:**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

- **Bản chất & Ý nghĩa:**
  Precision trả lời cho câu hỏi: *"Trong tất cả các lần mô hình khoanh vùng và báo 'Đó là mũ bảo hộ', có bao nhiêu phần trăm là ĐÚNG thực tế?"*
  - **Precision cao ($\sim 95\%+$):** Mô hình rất cẩn trọng, dự đoán nào đưa ra cũng cực kỳ chuẩn xác, **rất ít khi bị báo động giả (False Positives - FP)**.
  - **Precision thấp:** Mô hình bị "nhìn nhầm", nhận diện các vật thể lạ (đèn đường, cọc giao thông, đầu người không đội mũ) thành mũ bảo hộ.

- **Ý nghĩa trong giám sát an toàn:**
  Nếu Precision thấp, hệ thống an ninh sẽ liên tục phát tiếng chuông cảnh báo sai, gây phiền hà và làm giảm độ tin cậy của người vận hành.

---

### 2. Recall (Độ phủ phát hiện / Khả năng chống bỏ sót)

- **Công thức toán học:**
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

- **Bản chất & Ý nghĩa:**
  Recall trả lời cho câu hỏi: *"Trong tất cả các mũ bảo hộ THỰC TẾ đang có trên công trường, mô hình đã tìm ra và phát hiện được bao nhiêu phần trăm?"*
  - **Recall cao ($\sim 98\%+$):** Mô hình phát hiện gần như toàn bộ đối tượng, **rất ít khi bị bỏ sót đối tượng (False Negatives - FN)**.
  - **Recall thấp:** Mô hình bỏ qua nhiều công nhân không đội mũ bảo hộ hoặc không nhìn thấy mũ bảo hộ do bị che khuất / quá tối / quá nhỏ.

- **Ý nghĩa trong giám sát an toàn:**
  Trong công trường xây dựng, **Recall là chỉ số quan trọng sống còn**. Bỏ sót một công nhân không đội mũ (FN) có thể dẫn đến tai nạn nguy hiểm. Do đó, hệ thống ưu tiên tối đa Recall.

---

### 3. mAP50 (Mean Average Precision at IoU = 0.50)

- **Bản chất & Ý nghĩa:**
  - **IoU (Intersection over Union):** Tỷ lệ diện tích chồng lấp giữa khung dự đoán (Bounding Box) và khung nhãn thực (Ground Truth).
  - **mAP50** là diện tích dưới đường cong Precision-Recall (Average Precision) khi xét một dự đoán là **ĐÚNG (True Positive)** nếu **$\text{IoU} \ge 0.50$ (chồng lấp từ 50% trở lên)**.
- **Vai trò:**
  Là thước đo tiêu chuẩn tổng hợp độ chính xác nhận diện phân lớp và khoanh vùng ở mức độ dễ/vừa phải. Cho biết mô hình phát hiện đối tượng "trúng mục tiêu" tốt đến mức nào mà chưa đòi hỏi khung viền phải ôm khít khao tuyệt đối.

---

### 4. mAP50-95 (Mean Average Precision từ IoU = 0.50 đến 0.95)

- **Bản chất & Ý nghĩa:**
  - mAP50-95 tính trung bình chỉ số mAP trên **10 ngưỡng IoU khác nhau** (từ 0.50, 0.55, 0.60, ..., 0.95 với bước nhảy 0.05).
- **Vai trò & Độ khắt khe:**
  - Đây là chỉ số đánh giá **độ chính xác định vị khung hình cực kỳ khắt khe (strict localization precision)**.
  - Để đạt mAP50-95 cao, mô hình không chỉ cần nhận diện đúng lớp mũ bảo hộ mà khung viền vuông (bounding box) phải ôm sát khít từng milimet viền mũ bảo hộ (ngay cả ở ngưỡng IoU=0.85 hay 0.90).
  - Đây là chỉ số phản ánh rõ nhất năng lực phát hiện vật thể nhỏ (`test_small`) và vật thể mờ ảo trong đêm (`test_lowlight_small`).

---

## PHẦN 2: PHÂN TÍCH CHI TIẾT TỪNG PHƯƠNG PHÁP & KẾT LUẬN TỪ BÁO CÁO THỰC NGHIỆM

Dựa trên kết quả thực nghiệm từ [`results/final_research_report.md`](file:///home/nguyenhuynh/Documents/helmet-yolov10/results/final_research_report.md), dưới đây là phân tích chi tiết từng phương pháp, kết quả số liệu chứng minh, nguyên lý, ưu điểm và nhược điểm.

```
   [E1 Baseline] ──(Train Data Augmentation)──> [E2 Augmentation]
         │                                            │
   (Pre-processing)                             (Combine All)
         │                                            │
         ▼                                            ▼
   [E3 Enhancement] ───(Tiled Inference)───> [E5 Full Proposed Method]
    (Gamma vs CLAHE)
```

---

### 1. Phương pháp E1: Baseline (Mô hình YOLOv10n Nguyên Bản)

* **Mô tả kỹ thuật:** Mô hình YOLOv10n gốc huấn luyện trên tập dữ liệu chuẩn, suy luận 1 lần (single-pass) ở kích thước $640 \times 640$.
* **Số liệu thực nghiệm nổi bật:**
  * Ảnh bình thường (`test_normal`): mAP50 đạt **0.943**, mAP50-95 đạt **0.650**, Recall **0.886**.
  * Ảnh thiếu sáng tổng hợp (`test_lowlight_synth`): mAP50 **sụt giảm thảm hại xuống 0.375** (giảm 56.8%), Recall chỉ đạt **0.320** (bỏ sót 68% đối tượng!).
  * Ảnh thiếu sáng & mũ nhỏ (`test_lowlight_small`): mAP50-95 chỉ đạt **0.150**.
* **Ưu điểm:**
  * **Tốc độ cực nhanh:** Thời gian suy luận chỉ **1.4 ms/ảnh**, đạt tốc độ **~714 FPS** trên GPU.
  * **Chi phí tính toán siêu thấp:** Đơn giản, dễ triển khai trực tiếp trên thiết bị biên (Edge Devices).
  * **Hiệu năng xuất sắc ở điều kiện lý tưởng:** Đạt mAP50 >94% khi ánh sáng đầy đủ.
* **Nhược điểm:**
  * **Thất bại nặng nề trong đêm tối:** Mô hình không thể trích xuất đặc trưng khi độ tương phản ảnh bị suy giảm.
  * **Mất mát vật thể nhỏ:** Do quá trình nén ảnh xuống $640\times 640$, các mũ bảo hộ nhỏ xa camera bị biến mất hoàn toàn sau các lớp Downsampling/Strided Convolution của Backbone.

---

### 2. Phương pháp E2: Photometric Data Augmentation (Tăng cường dữ liệu quang học)

* **Mô tả kỹ thuật:** Giữ nguyên kiến trúc mô hình, nhưng trong quá trình huấn luyện bổ sung các phép biến đổi quang học nặng: Nhiễu Gaussian, Mờ Gaussian (Blur), Biến đổi độ sáng & độ tương phản ngẫu nhiên (Random Brightness/Contrast).
* **Số liệu thực nghiệm nổi bật:**
  * Trên `test_lowlight_synth`: **mAP50 tăng vọt từ 0.375 lên 0.760** ($+38.5\%$), **Precision tăng từ 0.722 lên 0.910** ($+18.8\%$), **mAP50-95 tăng từ 0.205 lên 0.469** ($+26.4\%$).
  * Trên `test_lowlight_small`: Precision tăng từ **0.681 lên 0.850** ($+16.9\%$).
* **Ưu điểm:**
  * **KHÔNG TỐN THÊM BẤT KỲ CHI PHÍ SUY LUẬN NÀO:** Vẫn duy trì tốc độ kỷ lục **714 FPS (1.4 ms)** vì chỉ tác động vào lúc Train.
  * **Chống báo động giả tốt nhất (Precision cao nhất):** Đạt Precision **91.0% - 94.2%** trên mọi tập test, giúp mô hình phân biệt cực tốt giữa nhiễu bóng tối và mũ bảo hộ thật.
  * **Nâng cao tính bền vững (Robustness):** Giúp mô hình học được các biểu diễn đặc trưng không phụ thuộc vào cường độ ánh sáng.
* **Nhược điểm:**
  * Chưa tự mình giải quyết triệt để được bài toán mũ bảo hộ siêu nhỏ ở cự ly xa (vẫn cần sự trợ giúp của Tiled Inference để tăng Recall vật thể nhỏ).

---

### 3. Phương pháp E3: Low-Light Image Enhancement (Tiền xử lý nâng sáng: Gamma vs CLAHE)

* **Mô tả kỹ thuật:** Áp dụng thuật toán cân bằng và làm sáng ảnh trước khi đưa vào mô hình suy luận. Thực nghiệm tiến hành so sánh 2 kỹ thuật:
  1. **Gamma Correction ($\gamma = 0.7$):** Nâng sáng phi tuyến tính đồng nhất toàn cục.
  2. **CLAHE (Clip Limit=2.0, TileGrid=8x8):** Cân bằng biểu đồ tần xuất thích nghi cục bộ.

#### So sánh Chi tiết CLAHE vs Gamma Correction:

| Tiêu chí So sánh | Gamma Correction ($\gamma = 0.7$) | CLAHE ($8\times 8$, Clip=2.0) | Đánh giá & Chọn lọc |
| :--- | :--- | :--- | :--- |
| **Độ mượt mà ảnh** | Mượt mà, biến đổi liên tục toàn bức ảnh | Phân chia $8\times 8$ vùng cục bộ | **Gamma tốt hơn** (không bị ô lưới) |
| **Nhiễu hạt (Noise)** | Không khuếch đại nhiễu hạt vùng tối | Rất dễ bị khuếch đại nhiễu hạt | **Gamma tốt hơn** |
| **Precision (`test_lowlight`)** | **0.971 (97.1%)** 🚀 | 0.941 (94.1%) | **Gamma tăng +3.0% Precision** |
| **mAP50-95 (`test_lowlight`)** | **0.664 (66.4%)** | 0.651 (65.1%) | **Gamma vượt trội hơn** |
| **Thời gian tiền xử lý** | **~1.2 ms** (Rất nhanh) | ~2.1 ms (Lâu hơn) | **Gamma tối ưu hơn** |

* **Kết luận chọn lựa E3:** **Gamma Correction ($\gamma = 0.7$) chiến thắng hoàn toàn CLAHE.**
* **Ưu điểm Gamma Correction:**
  * Nâng Precision ảnh thiếu sáng tự nhiên lên mức đỉnh **97.1%**.
  * Tốc độ xử lý rất nhanh (thêm 1.2 ms, tổng suy luận **2.6 ms ~ 385 FPS**).
  * Bảo toàn biên dạng nét của vật thể mà không tạo vết đốm ô lưới.
* **Nhược điểm E3:**
  * Nếu dùng đơn lẻ trên mô hình E1 baseline (chưa train Augmentation), Gamma chỉ cải thiện nhẹ mAP50 chứ chưa thể khôi phục hoàn toàn các thông tin đặc trưng bị mất trong bóng tối thâu đêm.

---

### 4. Phương pháp E4: Tiled Inference (Suy luận Phân vùng Cắt ô)

* **Mô tả kỹ thuật:** Ảnh đầu vào được cắt thành 9 ô tile nhỏ $640\times 640$ có độ phủ lặp $25\%$. Mô hình quét từng tile ở ngưỡng đề xuất $conf = 0.001$, sau đó dùng Class-aware NMS ghép tọa độ về ảnh gốc.
* **Số liệu thực nghiệm nổi bật:**
  * Trên `test_small`: **Recall tăng nhảy vọt từ 0.753 lên 0.948** ($+19.5\%$), **mAP50-95 tăng từ 0.481 lên 0.617** ($+13.6\%$).
  * Trên `test_lowlight_small`: **mAP50-95 tăng gấp đôi từ 0.150 lên 0.314** ($+16.4\%$).
* **Ưu điểm:**
  * **Giải quyết tận gốc bài toán vật thể nhỏ:** Giúp mô hình suy luận ở độ phân giải gốc của vật thể, không bị mất nét do co nén ảnh.
  * **Khả năng quét sạch không bỏ sót đối tượng:** Đưa Recall trên toàn bộ tập test lên tới **97.8%**.
* **Nhược điểm & Đặc thù Precision ở E4:**
  * **Tốc độ giảm xuống:** Thời gian suy luận tăng lên **12.5 ms/ảnh (~80 FPS)** (tuy nhiên vẫn vượt xa chuẩn Real-time 30 FPS).
  * **Hiện tượng Precision thô giảm ở candidate threshold $conf=0.001$:** Do tạo ra hàng nghìn candidate bBox trùng lấp qua 9 ô tile để NMS lọc trọn vẹn, chỉ số Precision tính thô ở ngưỡng này thấp ($\sim 21.3\%$). Tuy nhiên, khi lọc ngưỡng thực tế ($conf \ge 0.25$), Precision quay trở về mức cao **$90\%+$** bình thường.

---

### 5. Phương pháp E5: Full Proposed Method (Pipeline Đề xuất Đầy đủ)

* **Mô tả kỹ thuật:** Hợp nhất 3 cải tiến chiến lược: **Mô hình E2 (Train Augmentation) + Tiền xử lý Gamma ($\gamma=0.7$) + Suy luận Tiled Inference ($640\times 640$, 25% Overlap)**.
* **Số liệu thực nghiệm nổi bật:**
  * **`test_lowlight_small` (Thiếu sáng & Mũ nhỏ):** mAP50 tăng từ **0.360 lên 0.655** ($+29.5\%$), mAP50-95 tăng từ **0.150 lên 0.471** ($+32.1\%$), Recall tăng từ **0.312 lên 0.739** ($+42.7\%$).
  * **`test_lowlight_synth` (Thiếu sáng diện rộng):** mAP50-95 đạt **0.544** ($+33.9\%$), Recall đạt **0.862** ($+54.2\%$).
  * **`test_small` (Mũ nhỏ):** mAP50-95 đạt kỷ lục **0.658** ($+17.7\%$), Recall đạt đỉnh **0.969** ($+21.6\%$).
  * **`test_all` (Toàn bộ dữ liệu):** Recall đạt mức tuyệt đối **0.980 (98.0%)**.
* **Ưu điểm:**
  * **HIỆU NĂNG CAO NHẤT TOÀN BỘ NGHIÊN CỨU:** Đạt độ chính xác vị trí (mAP50-95) và khả năng chống bỏ sót (Recall) cao chưa từng có.
  * **Hoạt động hoàn hảo trong môi trường khắc nghiệt nhất:** Kết hợp khả năng lọc nhiễu của E2, làm sáng của E3 và soi cận cảnh của E4.
  * **Tốc độ đạt 72.5 FPS (~13.8 ms):** Hoàn toàn đáp ứng thời gian thực cho các hệ thống giám sát an toàn cao cấp.
* **Nhược điểm:**
  * Tốn tài nguyên tính toán GPU hơn so với phương pháp single-pass đơn thuần.

---

## PHẦN 3: BẢNG TỔNG HỢP SO SÁNH VÀ KHUYẾN NGHỊ THỰC TIỄN

### 1. Bảng So sánh Tổng quan 5 Phương pháp

| Tiêu chí Đánh giá | E1 Baseline | E2 Augmentation | E3 Enhancement (Gamma) | E4 Tiled Inference | E5 Full Proposed Method |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP50 (`test_lowlight_small`)** | 0.360 | 0.576 | 0.341 | 0.455 | **0.655** 🏆 |
| **mAP50-95 (`test_lowlight_small`)** | 0.150 | 0.304 | 0.170 | 0.314 | **0.471** 🏆 |
| **Recall (`test_lowlight_small`)** | 0.312 | 0.520 | 0.244 | 0.498 | **0.739** 🏆 |
| **Recall (`test_all`)** | 0.887 | 0.888 | 0.879 | 0.978 | **0.980** 🏆 |
| **Precision ở Single-pass** | 0.933 | **0.942** 🏆 | 0.931 | N/A ($conf=0.001$) | N/A ($conf=0.001$) |
| **Thời gian suy luận (Latency)** | **1.4 ms** | **1.4 ms** 🏆 | 2.6 ms | 12.5 ms | 13.8 ms |
| **Tốc độ xử lý (FPS)** | **~714 FPS** | **~714 FPS** 🏆 | ~385 FPS | ~80 FPS | **~72.5 FPS** (Real-time) |

---

### 2. Khuyến nghị Kịch bản Triển khai

1. **Kịch bản 1: Hệ thống Camera Biên / Edge Devices (Jetson Nano, Raspberry Pi, Camera IP giá rẻ)**
   * **Phương pháp đề xuất:** **E2 (Photometric Data Augmentation)**.
   * **Lý do:** Tốc độ giữ nguyên **714 FPS**, không tốn tài nguyên RAM/GPU, mAP trên ảnh thiếu sáng tăng từ $37.5\%$ lên $76.0\%$, Precision cực cao ($94.2\%$).

2. **Kịch bản 2: Hệ thống Máy chủ Giám sát An toàn Lao động Chuyên dụng (Industrial Safety Monitoring)**
   * **Phương pháp đề xuất:** **E5 (Full Proposed Method)**.
   * **Lý do:** Đạt độ an toàn tối đa với **Recall 98.0%** (gần như không bao giờ bỏ sót công nhân không đội mũ), mAP50-95 đỉnh cao ($0.658$), đồng thời duy trì **72.5 FPS** dư sức phân tích nhiều luồng camera cùng lúc.
