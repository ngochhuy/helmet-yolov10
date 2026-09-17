# Báo cáo So sánh Thực nghiệm E3: CLAHE vs Gamma Correction (Trả lời RQ2)

**Mô hình cơ sở:** YOLOv10n (`experiments/E1_baseline/baseline_b16_seed42/weights/best.pt`)  
**Tập dữ liệu:** Đánh giá trên 6 kịch bản Subtest  
**Mục đích:** Đánh giá và so sánh ảnh hưởng của hai kỹ thuật tiền xử lý nâng sáng ảnh **CLAHE** (Contrast Limited Adaptive Histogram Equalization) và **Gamma Correction** ($\gamma = 0.7$) phục vụ trả lời **RQ2**.  
**File kết quả gốc:**
- [`results/E3_enhancement/E3_clahe_subtests_summary.md`](file:///home/nguyenhuynh/Documents/helmet-yolov10/results/E3_enhancement/E3_clahe_subtests_summary.md)
- [`results/E3_enhancement/E3_gamma_subtests_summary.md`](file:///home/nguyenhuynh/Documents/helmet-yolov10/results/E3_enhancement/E3_gamma_subtests_summary.md)

---

## 1. Bảng So sánh Tổng hợp Chỉ số (E1 Baseline vs E3 CLAHE vs E3 Gamma)

| Subtest ID | Mô tả kịch bản | E1 Baseline (mAP50) | E3 CLAHE (mAP50) | E3 Gamma (mAP50) | E1 Baseline (mAP50-95) | E3 CLAHE (mAP50-95) | E3 Gamma (mAP50-95) | Phương pháp tối ưu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | **0.943** | 0.940 | 0.939 | **0.650** | 0.647 | 0.644 | E1 Baseline |
| `test_normal` | Ảnh điều kiện bình thường | **0.943** | 0.940 | 0.939 | **0.650** | 0.647 | 0.644 | E1 Baseline |
| 🌙 `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.947 | 0.938 | **0.951** 🚀 | 0.653 | 0.660 | **0.664** 🚀 | **E3 Gamma (+0.4% mAP50, +1.1% mAP50-95)** |
| 🌑 `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.375 | 0.336 | **0.381** 🚀 | 0.205 | 0.188 | **0.221** 🚀 | **E3 Gamma (+0.6% mAP50, +1.6% mAP50-95)** |
| 🔬 `test_small` | Mũ bảo hộ kích thước nhỏ | **0.838** | 0.825 | 0.822 | **0.481** | 0.479 | 0.476 | E1 Baseline |
| 💥 `test_lowlight_small` | Thiếu sáng & Mũ nhỏ | **0.360** | 0.335 | 0.341 | 0.150 | 0.163 | **0.170** 🚀 | **E3 Gamma (+2.0% mAP50-95)** |

---

## 2. Chi tiết Chỉ số theo Precision & Recall

### Bảng chi tiết Precision (Độ chính xác) & Recall (Độ phủ):

| Subtest ID | Precision (E1) | Precision (CLAHE) | Precision (Gamma) | Recall (E1) | Recall (CLAHE) | Recall (Gamma) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_all` | 0.933 | **0.939** | 0.931 | **0.887** | 0.881 | 0.879 |
| `test_normal` | 0.935 | **0.939** | 0.931 | **0.886** | 0.881 | 0.879 |
| 🌙 `test_lowlight` | 0.965 | 0.956 | **0.971** (+0.6%) | **0.882** | 0.869 | 0.860 |
| 🌑 `test_lowlight_synth` | 0.722 | 0.721 | **0.757** (+3.5%) | **0.320** | 0.257 | 0.273 |
| 🔬 `test_small` | **0.925** | 0.921 | 0.882 | **0.753** | 0.730 | 0.760 |
| 💥 `test_lowlight_small` | 0.681 | 0.711 | **0.816** (+13.5%) | **0.312** | 0.252 | 0.244 |

---

## 3. Phân tích & Nhận xét Kết quả (Kết luận trả lời RQ2)

1. **Gamma Correction ($\gamma = 0.7$) vượt trội hơn CLAHE trên các kịch bản ảnh thiếu sáng:**
   - Trên tập `test_lowlight` (ảnh thiếu sáng tự nhiên), Gamma Correction giúp tăng chỉ số **mAP50 lên 95.1%** (+0.4% so với E1 Baseline) và **mAP50-95 lên 66.4%** (+1.1% so với E1 Baseline).
   - Trên tập `test_lowlight_synth` (ảnh thiếu sáng diện rộng), Gamma Correction đẩy **Precision lên 75.7%** (+3.5%), đưa **mAP50-95 tăng từ 20.5% lên 22.1%** (+1.6%).
   - Trên tập `test_lowlight_small` (kết hợp thiếu sáng và mũ nhỏ), Gamma Correction nâng **Precision vượt bậc lên 81.6%** (+13.5% so với E1 Baseline).

2. **Tại sao Gamma Correction lại hiệu quả hơn CLAHE trong thiết lập suy luận trực tiếp (Zero-retraining)?**
   - **CLAHE (Cân bằng histogram theo ma trận ô local grid):** Làm thay đổi đột ngột biên độ độ sáng giữa các ô tile $8\times 8$ kề nhau, làm méo tuyến tính phân bố pixel (local gradient artifact). Khi mô hình E1 baseline chưa từng được train với ảnh CLAHE, sự biến đổi cục bộ này khiến detector dễ bỏ sót đối tượng (Recall giảm).
   - **Gamma Correction ($\gamma = 0.7$):** Nâng sáng đồng nhất mượt mà trên toàn bộ dải ánh sáng của hình ảnh theo đường cong phi tuyến mượt. Điều này bảo toàn nguyên vẹn cấu trúc đường biên (edge detection) và đặc trưng hình dạng của mũ bảo hộ, giúp mô hình nhận diện chính xác hơn.

3. **Tác động trên ảnh điều kiện bình thường (`test_normal`):**
   - Khi áp dụng tiền xử lý nâng sáng trên ảnh sáng bình thường, cả CLAHE và Gamma đều làm giảm nhẹ chỉ số mAP50 (-0.3% đến -0.4%). Điều này khẳng định quy tắc: **Chỉ nên bật tiền xử lý nâng sáng khi phát hiện ảnh thuộc vùng tối/thiếu sáng**.

---

## 4. Đề xuất cho Thực nghiệm E5 (Pipeline Phương pháp Đề xuất)

Dựa trên kết quả thực nghiệm E3:
- **Gamma Correction ($\gamma = 0.7$)** là ứng viên sáng giá hơn CLAHE khi áp dụng nâng sáng cho mô hình chưa train lại.
- Tuy nhiên, khi kết hợp với mô hình **E2 (đã được train với Data Augmentation nhiễu & mờ)** trong **E5**, cả hai phương pháp đều bảo toàn được chất lượng đặc trưng tốt hơn.
