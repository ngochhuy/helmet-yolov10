# Báo cáo đánh giá Subtests - Thí nghiệm E2 (Data Augmentation batch 16)

**File trọng số:** `/home/nguyenhuynh/Documents/helmet-yolov10/experiments/E2/augmentation_b16_seed42/weights/best.pt`  
**Cấu hình:** `configs/E2_augmentation.yaml`  

## Bảng tổng hợp chỉ số theo Subtest

| Subtest ID | Mô tả | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `test_all` | Toàn bộ Test set | 0.943 | 0.885 | 0.944 | 0.649 |
| `test_normal` | Ảnh điều kiện bình thường | 0.944 | 0.885 | 0.944 | 0.648 |
| `test_lowlight` | Ảnh thiếu sáng tự nhiên | 0.981 | 0.872 | 0.965 | 0.687 |
| `test_lowlight_synth` | Ảnh thiếu sáng tổng hợp | 0.807 | 0.692 | 0.760 | 0.469 |
| `test_small` | Mũ bảo hộ kích thước nhỏ | 0.943 | 0.733 | 0.856 | 0.491 |
| `test_lowlight_small` | Thiếu sáng & Mũ kích thước nhỏ | 0.798 | 0.517 | 0.576 | 0.304 |
