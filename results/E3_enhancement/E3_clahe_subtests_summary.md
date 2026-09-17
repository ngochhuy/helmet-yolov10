# Báo cáo đánh giá Subtests - Thí nghiệm E3 (Low-light Enhancement: CLAHE)

**File trọng số E1 Baseline:** `/home/nguyenhuynh/Documents/helmet-yolov10/experiments/E1_baseline/baseline_b16_seed42/weights/best.pt`  

**Phương pháp nâng sáng:** `CLAHE` ({'color_space': 'LAB', 'channel': 'L', 'clip_limit': 2.0, 'tile_grid_size': [8, 8]})  

**Cấu hình:** `configs/E3_enhancement.yaml`  


## Bảng tổng hợp chỉ số theo Subtest


| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.939 | 0.881 | 0.940 | 0.647 |
| `test_normal` | Ảnh điều kiện bình thường | 0.939 | 0.881 | 0.940 | 0.647 |
| `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.956 | 0.869 | 0.938 | 0.660 |
| `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.721 | 0.257 | 0.336 | 0.188 |
| `test_small` | Mũ bảo hộ kích thước nhỏ | 0.921 | 0.730 | 0.825 | 0.479 |
| `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.711 | 0.252 | 0.335 | 0.163 |
