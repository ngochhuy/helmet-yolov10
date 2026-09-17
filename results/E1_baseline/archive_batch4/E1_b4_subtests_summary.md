# Báo cáo đánh giá Subtests - Thí nghiệm E1 (Batch size 4)

**File trọng số:** `/home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1/baseline_seed42_20260916T100841Z/weights/best.pt`  
**Cấu hình:** `configs/E1_baseline.yaml`  

## Bảng tổng hợp chỉ số theo Subtest (Batch size 4)

| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.935 | 0.887 | 0.942 | 0.651 |
| `test_normal` | Ảnh điều kiện bình thường | 0.937 | 0.886 | 0.942 | 0.651 |
| `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.967 | 0.882 | 0.948 | 0.654 |
| `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.725 | 0.321 | 0.378 | 0.208 |
| `test_small` | Mũ bảo hộ kích thước nhỏ | 0.927 | 0.755 | 0.840 | 0.483 |
| `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.684 | 0.315 | 0.363 | 0.152 |
