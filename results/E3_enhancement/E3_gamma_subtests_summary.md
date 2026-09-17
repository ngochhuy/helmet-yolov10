# Báo cáo đánh giá Subtests - Thí nghiệm E3 (Low-light Enhancement: GAMMA)

**File trọng số E1 Baseline:** `/home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1_baseline/baseline_b16_seed42/weights/best.pt`  

**Phương pháp nâng sáng:** `GAMMA` ({'gamma': 0.7})  

**Cấu hình:** `configs/E3_enhancement.yaml`  


## Bảng tổng hợp chỉ số theo Subtest


| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.931 | 0.879 | 0.939 | 0.644 |
| `test_normal` | Ảnh điều kiện bình thường | 0.931 | 0.879 | 0.939 | 0.644 |
| `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.971 | 0.860 | 0.951 | 0.664 |
| `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.757 | 0.273 | 0.381 | 0.221 |
| `test_small` | Mũ bảo hộ kích thước nhỏ | 0.882 | 0.760 | 0.822 | 0.476 |
| `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.816 | 0.244 | 0.341 | 0.170 |
