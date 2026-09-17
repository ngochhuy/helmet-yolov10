# Báo cáo đánh giá Subtests - Thí nghiệm E1 (Baseline batch 16)

**File trọng số:** `/home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_b16_seed42/weights/best.pt`  
**Cấu hình:** `configs/E1_baseline.yaml`  

## Bảng tổng hợp chỉ số theo Subtest

| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.933 | 0.887 | 0.943 | 0.650 |
| `test_normal` | Ảnh điều kiện bình thường | 0.935 | 0.886 | 0.943 | 0.650 |
| `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.965 | 0.882 | 0.947 | 0.653 |
| `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.722 | 0.320 | 0.375 | 0.205 |
| `test_small` | Mũ bảo hộ kích thước nhỏ | 0.925 | 0.753 | 0.838 | 0.481 |
| `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.681 | 0.312 | 0.360 | 0.150 |
