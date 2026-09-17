# Báo cáo Bảng so sánh Đối chiếu: Baseline E1 vs Data Augmentation E2

**Protocol:** Đánh giá trên cùng phần cứng V100 GPU, cùng `batch: 16`, `epochs: 100`, `imgsz: 640` và `seed: 42`.  
**Mục đích:** Trả lời **câu hỏi nghiên cứu RQ1** ("Data Augmentation có cải thiện khả năng phát hiện mũ bảo hộ trong các điều kiện khó hay không?").

---

## 📊 Bảng so sánh chỉ số mAP50 & mAP50-95 trên 6 Subtests

| Subtest ID | Mô tả kịch bản | E1 Baseline (mAP50) | E2 Augmentation (mAP50) | Mức tăng trưởng ($\Delta$ mAP50) | E1 Baseline (mAP50-95) | E2 Augmentation (mAP50-95) | Mức tăng trưởng ($\Delta$ mAP50-95) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.943 | 0.944 | +0.1% | 0.650 | 0.649 | -0.1% |
| `test_normal` | Ảnh điều kiện bình thường | 0.943 | 0.944 | +0.1% | 0.650 | 0.648 | -0.2% |
| 🌙 `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.947 | **0.965** | **+1.8%** | 0.653 | **0.687** | **+3.4%** |
| 🌑 `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.375 | **0.760** | **+38.5%** 🚀 | 0.205 | **0.469** | **+26.4%** 🚀 |
| 🔬 `test_small` | Mũ bảo hộ kích thước nhỏ | 0.838 | **0.856** | **+1.8%** | 0.481 | **0.491** | **+1.0%** |
| 💥 `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.360 | **0.576** | **+21.6%** 🚀 | 0.150 | **0.304** | **+15.4%** 🚀 |

---

## 🎯 Phân tích & Nhận xét kết quả (Kết luận cho RQ1)

1. **Cải thiện vượt bậc ở điều kiện thiếu sáng (Low-light):**
   - Trên tập `test_lowlight_synth` (ảnh thiếu sáng diện rộng), mAP50 của E2 tăng vọt từ **37.5% lên 76.0%** (tăng **+38.5%**).
   - Trên ảnh thiếu sáng tự nhiên `test_lowlight`, mAP50 tăng từ **94.7% lên 96.5%** (+1.8%).

2. **Cải thiện ở kịch bản vật thể nhỏ (Small Object):**
   - Trên tập `test_small`, mAP50 tăng từ **83.8% lên 85.6%** (+1.8%).

3. **Cải thiện vượt trội ở kịch bản thách thức nhất (Low-light & Small):**
   - Trên tập `test_lowlight_small` (vừa tối vừa nhỏ), mAP50 tăng bứt phá từ **36.0% lên 57.6%** (tăng **+21.6%**).

4. **Kết luận khoa học:**
   Data Augmentation (gồm Photometric Augmentation như chỉnh độ sáng, tương phản, nhiễu và mờ Gaussian) **đã chứng minh hiệu quả cực kỳ mạnh mẽ** trong việc giúp mô hình YOLOv10n chống chịu các điều kiện ảnh thiếu sáng và vật thể nhỏ mà không làm suy giảm hiệu năng ở điều kiện bình thường (`test_normal` duy trì 94.4%).
