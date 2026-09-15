# Cải thiện phát hiện mũ bảo hộ bằng YOLOv10n

Repository phục vụ nghiên cứu cải thiện khả năng phát hiện mũ bảo hộ trong hai
điều kiện khó: **ảnh thiếu sáng** và **mũ bảo hộ có kích thước nhỏ**. Nghiên cứu
sử dụng YOLOv10n làm mô hình cơ sở, tập trung cải thiện pipeline dữ liệu và suy
luận thay vì thiết kế lại backbone, neck hoặc detection head.

Các hướng được khảo sát gồm:

- tăng cường dữ liệu theo điều kiện thực tế;
- tăng cường ảnh thiếu sáng bằng CLAHE và/hoặc Gamma Correction;
- tiled inference (suy luận theo vùng) cho vật thể nhỏ;
- đánh giá sự đánh đổi giữa độ chính xác và chi phí suy luận.

> **Trạng thái:** baseline E1 đã có cấu hình, kiểm tra dataset, CLI huấn luyện và
> đánh giá. Chưa có checkpoint vì dataset chuẩn hóa và pretrained weights chưa
> được đặt trong repository. Các phương pháp E2–E5 vẫn đang được triển khai.

## Mục tiêu nghiên cứu

Mục tiêu tổng quát là đánh giá liệu một pipeline tương đối đơn giản, không thay
đổi sâu kiến trúc YOLOv10n, có thể cải thiện hiệu quả phát hiện helmet trong ảnh
thiếu sáng và đối với vật thể nhỏ hay không.

Các mục tiêu cụ thể:

1. Xây dựng YOLOv10n với kích thước đầu vào 640 × 640 làm baseline.
2. Đo mức suy giảm hiệu năng trên ảnh thiếu sáng, helmet nhỏ và trường hợp kết
   hợp cả hai điều kiện.
3. Đánh giá riêng ảnh hưởng của augmentation, CLAHE/Gamma Correction và tiled
   inference.
4. Thực hiện ablation study để xác định đóng góp của từng thành phần.
5. So sánh độ chính xác với thời gian suy luận/FPS trên cùng phần cứng.

## Câu hỏi nghiên cứu

- **RQ1:** Data augmentation có cải thiện khả năng phát hiện trong các điều kiện
  khó hay không?
- **RQ2:** CLAHE hoặc Gamma Correction có cải thiện khả năng phát hiện trên ảnh
  thiếu sáng hay không?
- **RQ3:** Tiled inference có cải thiện khả năng phát hiện helmet kích thước nhỏ
  hay không?
- **RQ4:** Pipeline kết hợp cải thiện bao nhiêu so với baseline và phải đánh đổi
  như thế nào về thời gian suy luận?

## Phạm vi

- Mô hình chính: YOLOv10n.
- Input baseline: 640 × 640.
- Không tập trung thay đổi backbone, neck hoặc detection head.
- Dataset dự kiến: Hard Hat Workers và Safety Helmet Wearing Dataset (SHWD).
- Dataset chỉ được kết hợp sau khi kiểm tra hệ nhãn, giấy phép và phân bố dữ liệu.
- Kết luận nghiên cứu chỉ được đưa ra sau khi hoàn thành thực nghiệm.

## Thiết kế thí nghiệm

| ID | Thiết lập | Mục đích |
| --- | --- | --- |
| E1 | YOLOv10n baseline | Thiết lập mốc so sánh |
| E2 | Baseline + augmentation | Đo ảnh hưởng của tăng cường dữ liệu |
| E3 | Baseline + low-light enhancement | Đánh giá CLAHE/Gamma Correction |
| E4 | Baseline + tiled inference | Đánh giá khả năng phát hiện vật thể nhỏ |
| E5 | Pipeline kết hợp | Đánh giá hiệu quả tổng thể và trade-off |

Để bảo đảm so sánh công bằng, các thí nghiệm huấn luyện phải dùng cùng dataset
split, random seed, trọng số khởi tạo, số epoch, batch size, optimizer và các
thiết lập có liên quan, ngoại trừ biến số đang được khảo sát. Việc benchmark phải
được thực hiện trên cùng phần cứng, cùng batch size và có warm-up.

Kết quả được phân tích trên toàn bộ test set và trên bốn nhóm điều kiện:

1. ảnh bình thường;
2. ảnh thiếu sáng;
3. helmet kích thước nhỏ;
4. ảnh thiếu sáng có helmet kích thước nhỏ.

## Chỉ số đánh giá

- Precision;
- Recall;
- mAP50;
- mAP50–95;
- Recall/AP cho vật thể nhỏ;
- Recall/AP trên low-light subset;
- thời gian suy luận trên mỗi ảnh và FPS;
- số tham số và chi phí tính toán, nếu cần cho phần so sánh mô hình.

Ngưỡng xác định ảnh thiếu sáng và vật thể nhỏ phải được quy định trước khi chạy
thí nghiệm, lưu trong cấu hình và áp dụng thống nhất cho mọi phương pháp.

## Cấu trúc repository

```text
helmet-yolov10/
├── configs/                  # Cấu hình dataset và thí nghiệm E1–E5
├── data/
│   └── dataset_info/         # Nguồn, giấy phép, mapping nhãn và checksum
├── experiments/              # Log và artefact của từng lần chạy
├── paper/                    # Bản thảo, hình và bảng dùng trong bài báo
├── results/                  # Kết quả tổng hợp giữa các thí nghiệm
├── scripts/                  # Entry point train/evaluate/benchmark
├── src/helmet_yolov10/
│   ├── augmentation/         # Data augmentation
│   ├── data/                 # Chuẩn hóa, phân tích và chia dataset
│   ├── enhancement/          # CLAHE và Gamma Correction
│   ├── evaluation/           # Metric và benchmark
│   ├── inference/            # Standard và tiled inference
│   ├── training/             # Quy trình huấn luyện
│   └── utils/                # Seed, logging và tiện ích chung
└── tests/                    # Kiểm thử các thành phần
```

Dataset, checkpoint, log dung lượng lớn và ảnh dự đoán không được lưu trực tiếp
trong Git. Thông tin nguồn dữ liệu, giấy phép, class mapping, split seed, số lượng
ảnh/đối tượng và checksum được ghi tại `data/dataset_info/`.

## Cài đặt môi trường phát triển

Yêu cầu Python 3.10 trở lên. Repository sử dụng `pyproject.toml` và có lockfile
cho `uv`.

```bash
uv sync
uv pip install -e .
```

Hoặc cài package ở chế độ editable bằng pip:

```bash
python -m pip install -e .
```

Danh sách thư viện runtime hiện chưa được chốt trong `pyproject.toml`; môi trường
huấn luyện hoàn chỉnh sẽ cần bổ sung phiên bản cố định của framework YOLOv10,
PyTorch, OpenCV và các thư viện đánh giá liên quan.

## Chuẩn bị dữ liệu

Dữ liệu đầu vào cần được chuyển về định dạng YOLO và tối thiểu phải có cấu trúc:

```text
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Trước khi huấn luyện cần hoàn thành các bước sau:

1. kiểm tra ảnh lỗi, annotation không hợp lệ và dữ liệu trùng lặp;
2. thống nhất class mapping giữa các nguồn dữ liệu;
3. tạo train/validation/test split xác định bằng seed;
4. thống kê số ảnh, số object và phân bố theo class;
5. định nghĩa tiêu chí small object;
6. tạo danh sách mẫu thuộc low-light subset;
7. lưu manifest và checksum để có thể tái tạo đúng dataset.

Không dùng test set để chọn tham số CLAHE, gamma, kích thước tile, overlap hoặc
confidence threshold. Các tham số này chỉ được chọn trên validation set.

## Xây dựng baseline E1

Baseline dùng implementation từ repository chính thức
[THU-MIG/YOLOv10](https://github.com/THU-MIG/yolov10). Nên tạo môi trường Python
3.10 hoặc 3.11 riêng, cài PyTorch phù hợp với CUDA, rồi cài dự án và YOLOv10:

```bash
python -m pip install -e .
python -m pip install git+https://github.com/THU-MIG/yolov10.git
```

Tải pretrained weights `yolov10n.pt` từ
[YOLOv10 v1.1](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt)
và lưu tại `weights/yolov10n.pt`.

Trước khi dùng GPU, kiểm tra cấu hình, class ID, bounding box và split:

```bash
python scripts/train.py --config configs/E1_baseline.yaml --validate-only
```

Huấn luyện baseline theo protocol đã khóa:

```bash
python scripts/train.py \
  --config configs/E1_baseline.yaml \
  --device 0 \
  --run-name baseline_seed42
```

Đánh giá checkpoint tốt nhất trên test set đã khóa:

```bash
python scripts/evaluate.py \
  --config configs/E1_baseline.yaml \
  --checkpoint experiments/E1/baseline_seed42/weights/best.pt \
  --run-name baseline_seed42_test
```

Trên PowerShell, viết lệnh trên một dòng hoặc thay `\` bằng dấu backtick.
Pipeline lưu resolved config, resolved dataset YAML, thống kê dataset, checksum
checkpoint, metric và metadata môi trường trong thư mục của từng run.

`scripts/benchmark.py` vẫn là placeholder và sẽ được hoàn thiện ở bước đánh giá
accuracy–efficiency.

## Thí nghiệm E2: data augmentation

E2 giữ nguyên model, dataset split, seed và protocol huấn luyện của E1. Chỉ
train split nhận các phép brightness/contrast, Gaussian noise và Gaussian blur;
validation/test không bị augmentation. Các tham số E2 được lưu tại
`configs/E2_augmentation.yaml`.

```bash
uv sync --extra dev --extra train
uv run python scripts/train.py --config configs/E2_augmentation.yaml --validate-only
uv run python scripts/train.py --config configs/E2_augmentation.yaml --device 0 --run-name augmentation_seed42
```

Mở `notebooks/E2_augmentation_training.ipynb` để chạy từng bước, kiểm tra input
và đánh giá `best.pt` trên test split không augmentation.

## Kiểm thử

```bash
python -m pytest
```

Các test baseline không tải model và không sử dụng GPU; backend được thay bằng
một implementation giả để kiểm tra config, dataset validation và tham số gọi
pipeline huấn luyện.
