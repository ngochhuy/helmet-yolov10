# Báo cáo đánh giá Subtests - Thí nghiệm E4 (Tiled Inference)

**File trọng số E1 Baseline:** `/home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1_baseline/baseline_b16_seed42/weights/best.pt`  

**Cấu hình suy luận phân vùng:** Tile Size [640, 640], Overlap 0.25  

**Cấu hình:** `configs/E4_tiled_inference.yaml`  


## Bảng tổng hợp chỉ số theo Subtest


| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.213 | 0.978 | 0.908 | 0.626 |
| `test_normal` | Ảnh điều kiện bình thường | 0.213 | 0.978 | 0.907 | 0.626 |
| `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.227 | 0.973 | 0.960 | 0.663 |
| `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.144 | 0.465 | 0.380 | 0.262 |
| `test_small` | Mũ bảo hộ kích thước nhỏ | 0.130 | 0.948 | 0.894 | 0.617 |
| `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.203 | 0.498 | 0.455 | 0.314 |
