# Báo cáo đánh giá Subtests - Thí nghiệm E5 (Proposed Full Pipeline)

**File trọng số E2 Augmentation:** `/home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2_augmentation/augmentation_b16_seed42/weights/best.pt`  

**Phương pháp nâng sáng:** `GAMMA` (gamma=0.7)  

**Suy luận phân vùng:** Tile Size [640, 640], Overlap 0.25  

**Cấu hình:** `configs/E5_full_method.yaml`  


## Bảng tổng hợp chỉ số theo Subtest


| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.256 | 0.980 | 0.908 | 0.653 |
| `test_normal` | Ảnh điều kiện bình thường | 0.255 | 0.980 | 0.907 | 0.653 |
| `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.281 | 0.982 | 0.968 | 0.697 |
| `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.132 | 0.862 | 0.756 | 0.544 |
| `test_small` | Mũ bảo hộ kích thước nhỏ | 0.145 | 0.969 | 0.914 | 0.658 |
| `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.124 | 0.739 | 0.655 | 0.471 |
