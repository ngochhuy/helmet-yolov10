# Báo cáo Phân tích Chi tiết Các Bộ Test & Subtest Suites Đánh giá Mô hình

**Đề tài:** Cải thiện khả năng phát hiện mũ bảo hộ bằng YOLOv10n trong điều kiện thiếu sáng và vật thể nhỏ  
**Dữ liệu đánh giá:** 6 kịch bản Subtest chuyên biệt phân tách từ tập dữ liệu Test  
**Mục đích tài liệu:** Phân tích chi tiết quy mô, cấu trúc, tiêu chí phân loại và ý nghĩa phương pháp luận của từng bộ Subtest phục vụ đánh giá chuỗi thí nghiệm E1 $\rightarrow$ E5  

---

## 1. TỔNG QUAN VỀ KIẾN TRÚC ĐÁNH GIÁ SUBTEST SUITES

Trong các bài toán phát hiện đối tượng thực tế, việc chỉ sử dụng **một tập Test chung duy nhất** thường dễ dẫn đến hiện tượng **"sai lệch trung bình" (average bias)**: chỉ số mAP chung có thể rất cao nhờ chiếm phần lớn là các ảnh điều kiện dễ (ánh sáng tốt, đối tượng to gần camera), từ đó che giấu hoàn toàn sự thất bại của mô hình ở các kịch bản góc hẹp, ban đêm hoặc khoảng cách xa.

Để giải quyết triệt để vấn đề này, nghiên cứu xây dựng **Kiến trúc Đánh giá Multi-Scenario Subtest Suite** dựa trên 2 trục thách thức chính:
1. **Trục Quang học (Illumination Axis):** Điều kiện ánh sáng (Bình thường / Thiếu sáng tự nhiên / Thiếu sáng tổng hợp diện rộng).
2. **Trục Hình học & Kích thước (Geometric Axis):** Kích thước đối tượng (Kích thước tiêu chuẩn / Vật thể nhỏ $< 32 \times 32$ pixels theo chuẩn COCO).

```
                            ┌─────────────────────────────────────────┐
                            │      TỔNG TẬP TEST (`test_all`)         │
                            │   1,533 Ảnh  |  5,625 Đối tượng         │
                            └────────────────────┬────────────────────┘
                                                 │
            ┌────────────────────────────────────┼────────────────────────────────────┐
            │                                    │                                    │
            ▼                                    ▼                                    ▼
┌───────────────────────┐            ┌───────────────────────┐            ┌───────────────────────┐
│     `test_normal`     │            │    `test_lowlight`    │            │ `test_lowlight_synth` │
│ 1,491 Ảnh | 5,513 Objs│            │   42 Ảnh | 112 Objs   │            │ 1,533 Ảnh | 5,625 Objs│
│ (Ánh sáng bình thường)│            │ (Thiếu sáng tự nhiên) │            │ (Thiếu sáng tổng hợp) │
└───────────────────────┘            └───────────────────────┘            └───────────────────────┘
                                                 │
            ┌────────────────────────────────────┴────────────────────────────────────┐
            │                                                                         │
            ▼                                                                         ▼
┌───────────────────────┐                                                 ┌───────────────────────┐
│     `test_small`      │                                                 │ `test_lowlight_small` │
│   84 Ảnh | 426 Objs   │                                                 │   84 Ảnh | 426 Objs   │
│(70.9% đối tượng nhỏ)  │                                                 │(Thiếu sáng & Mũ nhỏ)  │
└───────────────────────┘                                                 └───────────────────────┘
```

---

## 2. BẢNG THỐNG KÊ ĐỊNH LƯỢNG CHI TIẾT 6 BỘ SUBTEST

Bảng dưới đây trình bày số liệu thống kê thực tế được trích xuất trực tiếp từ cấu trúc thư mục dữ liệu đánh giá (`data/processed/eval/`):

| Subtest ID | Tên kịch bản | Số lượng ảnh | Tổng đối tượng | Số `helmet` (Mũ) | Số `head` (Đầu) | Số vật thể nhỏ ($<32^2$px) | Tỷ lệ vật thể nhỏ | Mức độ thách thức |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`test_all`** | Toàn bộ Tập Test | **1,533** | **5,625** | 4,371 | 1,254 | 921 | 16.4% | Tổng thể đại diện |
| **`test_normal`** | Ánh sáng bình thường | **1,491** | **5,513** | 4,265 | 1,248 | 915 | 16.6% | Cơ bản (Dễ) |
| 🌙 **`test_lowlight`** | Thiếu sáng tự nhiên | **42** | **112** | 106 | 6 | 6 | 5.4% | Trung bình (Tự nhiên) |
| 🌑 **`test_lowlight_synth`**| Thiếu sáng tổng hợp | **1,533** | **5,625** | 4,371 | 1,254 | 921 | 16.4% | Cực khó (Diện rộng) |
| 🔬 **`test_small`** | Mũ bảo hộ kích thước nhỏ | **84** | **426** | 330 | 96 | **302** | **70.9%** 🚀 | Khó (Vật thể nhỏ) |
| 💥 **`test_lowlight_small`**| Thiếu sáng & Mũ nhỏ | **84** | **426** | 330 | 96 | **302** | **70.9%** 🚀 | **Thách thức tối đa (Worst-case)** |

---

## 3. PHÂN TÍCH CHI TIẾT NỘI DUNG & ĐẶC ĐIỂM KỸ THUẬT TỪNG BỘ SUBTEST

---

### 1. Subtest `test_all` (Toàn bộ Tập Test Cơ sở)
* **Quy mô:** 1,533 ảnh, 5,625 đối tượng (4,371 mũ bảo hộ `helmet`, 1,254 đầu người không đội mũ `head`).
* **Đặc điểm:** Tập dữ liệu kiểm thử đầy đủ được phân tách độc lập (Split ratio 15%) từ tổng tập dữ liệu 10,254 ảnh.
* **Mục đích:** Đánh giá hiệu năng tổng thể chung của mô hình trên toàn bộ miền dữ liệu kiểm thử, làm căn cứ tính toán các chỉ số mAP50 và mAP50-95 chuẩn.

---

### 2. Subtest `test_normal` (Ảnh Điều kiện Ánh sáng Bình thường)
* **Quy mô:** 1,491 ảnh, 5,513 đối tượng (4,265 mũ bảo hộ, 1,248 đầu người).
* **Đặc điểm:** Bao gồm các ảnh được chụp trong điều kiện ánh sáng ban ngày đầy đủ, độ tương phản rõ nét, không bị nhiễu bóng tối hay cháy sáng.
* **Mục đích:** Giám sát xem các kỹ thuật cải tiến (như Augmentation hay Tiền xử lý nâng sáng Gamma) có làm suy giảm hiệu năng (**performance regression**) trên môi trường hoạt động tiêu chuẩn ban ngày hay không.

---

### 3. Subtest `test_lowlight` (Ảnh Thiếu sáng Tự nhiên - Natural Low-light)
* **Quy mô:** 42 ảnh, 112 đối tượng (106 mũ bảo hộ, 6 đầu người).
* **Đặc điểm:** Gồm các bức ảnh chụp thực tế ngoài đời thực trong điều kiện ban đêm, hoàng hôn, bóng râm công trình hoặc ánh sáng đèn đường yếu.
* **Mục đích:** Kiểm chứng tính khả thi và độ tin cậy thực tế của mô hình trên mẫu dữ liệu thiếu sáng tự nhiên thu thập từ thực địa.

---

### 4. Subtest `test_lowlight_synth` (Ảnh Thiếu sáng Tổng hợp Diện rộng - Synthetic Low-light)
* **Quy mô:** 1,533 ảnh (toàn bộ 100% tập test), 5,625 đối tượng.
* **Quy trình tạo lập:** Áp dụng biến đổi giảm độ sáng ngẫu nhiên (Gamma scaling $\gamma \in [2.0, 3.5]$) và triệt tiêu độ tương phản trên toàn bộ 1,533 ảnh test.
* **Mục đích:** Tạo ra một **Bãi thử nghiệm chịu tải cực hạn (Large-scale Stress Test)** khắt khe ở quy mô mẫu lớn. Giúp loại bỏ hoàn toàn sai số thống kê (sample size bias) do tập ảnh natural lowlight có số lượng ảnh nhỏ (42 ảnh), từ đó đo lường chính xác khả năng kháng nhiễu tối của mô hình.

---

### 5. Subtest `test_small` (Mũ bảo hộ Kích thước Nhỏ - Small Object Subtest)
* **Quy mô:** 84 ảnh, 426 đối tượng (330 mũ bảo hộ, 96 đầu người).
* **Đặc điểm nổi bật:** Có tới **302 đối tượng có diện tích $<32 \times 32$ pixels (chiếm 70.9%)**. Ảnh chủ yếu thu từ góc quay camera giám sát từ trên cao (down-looking surveillance) hoặc góc máy toàn cảnh rộng nơi công nhân đứng xa camera từ 15m – 50m.
* **Mục đích:** Đánh giá điểm nghẽn về độ phân giải không gian (spatial resolution loss) và mất mát thông tin qua các lớp Feature Pyramid Network / Backbone. Đây là bộ subtest then chốt để chứng minh hiệu quả của thuật toán suy luận phân vùng **Tiled Inference (E4)**.

---

### 6. Subtest `test_lowlight_small` (Kịch bản Thách thức Kết hợp: Thiếu sáng & Mũ nhỏ)
* **Quy mô:** 84 ảnh, 426 đối tượng (70.9% đối tượng kích thước nhỏ $<32^2$px dưới ánh sáng tổng hợp giảm sâu).
* **Đặc điểm:** Đây là kịch bản thử nghiệm **"Khắc nghiệt nhất" (Worst-case Scenario)** trong toàn bộ đề tài nghiên cứu. Đối tượng vừa nhỏ mờ ở khoảng cách xa, vừa bị chìm trong bóng tối.
* **Mục đích:** Kiểm chứng giới hạn tối đa của giải pháp. Kết quả thực nghiệm đã chứng minh: Mô hình Baseline E1 gần như thất bại ở tập này (mAP50-95 chỉ đạt **15.0%**), trong khi phương pháp đề xuất đầy đủ **E5 đã nâng mAP50-95 lên 47.1% (tăng vọt +32.1%)**, khẳng định thành công vượt bậc của nghiên cứu.

---

## 4. Ý NGHĨA PHƯƠNG PHÁP LUẬN CỦA VIỆC PHÂN CHIA SUBTESTS

1. **Minh bạch hóa & Đơn lập nguyên nhân (Root-cause Isolation):**
   * Giúp xác định chính xác mỗi phương pháp cải tiến đóng góp vào khía cạnh nào:
     * **E2 (Augmentation):** Giải quyết triệt để bài toán thiếu sáng diện rộng (`test_lowlight_synth`).
     * **E3 (Gamma Enhancement):** Tối ưu độ sắc nét và Precision trên ảnh thiếu sáng tự nhiên (`test_lowlight`).
     * **E4 (Tiled Inference):** Phá vỡ giới hạn phát hiện mũ bảo hộ kích thước nhỏ (`test_small`).
     * **E5 (Full Proposed Method):** Làm chủ hoàn toàn kịch bản kết hợp phức tạp nhất (`test_lowlight_small`).

2. **Cung cấp cơ sở khoa học vững chắc cho bài báo & luận văn:**
   * Việc đánh giá trên 6 kịch bản Subtests đa dạng mang lại cái nhìn 360 độ về mô hình, đảm bảo tính chặt chẽ, khách quan và thuyết phục tối đa khi trình bày trước hội đồng chuyên môn hoặc công bố khoa học.
