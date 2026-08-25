# RSNA Knee MRI — Phát hiện bất thường đa nhãn

Phát hiện 12 bệnh lý khớp gối từ MRI, dùng **text-guided knowledge distillation**:
huấn luyện bằng ảnh + radiology report, nhưng lúc inference **chỉ cần ảnh**.

Bài toán cốt lõi: chỉ 58/4407 study có nhãn do người gán, nhưng cả 4407 đều có
report. Ta bóc tri thức từ 4407 cặp (ảnh, report) thay vì chỉ dựa vào 58 nhãn.

## Cài đặt

```bash
make install        # pip install -e ".[dev,viz,eval]" + pre-commit install
make check          # lint + kiểm tra kiểu + test + notebook — phải xanh
```

Không có `make`? Xem [Makefile](Makefile) để biết lệnh tương đương.
**Người mới tiếp nhận repo: đọc [CONTRIBUTING.md](CONTRIBUTING.md) trước.**

## Chạy nhanh (smoke test, không cần GPU)

```bash
python scripts/precompute_weak_labels.py --env local --limit 8
python scripts/train.py --env local --limit 8 --smoke --set train.epochs=1
python scripts/predict.py --env local
```

`--smoke` dùng đặc trưng teacher giả nên chạy được ngay mà không cần tải model nặng.

## Chạy đầy đủ (Colab A100)

```bash
python scripts/prepare_masks.py           --env colab   # S1: mask ROI (SAM)
python scripts/precompute_weak_labels.py  --env colab --use-vlm
python scripts/precompute_teacher_feats.py --env colab  # 3DINO-ViT
python scripts/precompute_guidance.py     --env colab   # guidance từ VLM
python scripts/train.py                   --env colab   # S2: KD
python scripts/evaluate.py                --env colab   # AUC trên 58 nhãn gold
python scripts/explain.py                 --env colab   # S3: CAM + báo cáo
python scripts/predict.py                 --env colab   # submission.csv
```

Hoặc mở [`notebooks/02_colab_train.ipynb`](notebooks/02_colab_train.ipynb).

Mọi lệnh cũng gọi được qua `python -m knee_mri.cli <lệnh>`, hoặc như entrypoint
đã cài (`knee-train`, `knee-predict`, ...).

## Cấu hình

Cấu hình là **dữ liệu YAML**, không phải code:

```
configs/base.yaml        # hyperparameter, kích thước model, trọng số loss
configs/env.local.yaml   # đường dẫn local, volume nhỏ
configs/env.colab.yaml   # đường dẫn Google Drive, batch lớn
configs/env.test.yaml    # volume tí hon cho unit test
```

Ghi đè bất kỳ khóa nào ngay trên dòng lệnh:

```bash
python scripts/train.py --env colab --set train.epochs=50 --set train.batch_size=16
```

Trong Python (notebook Colab), ghi đè có tác dụng thật vì `cfg` được truyền
tường minh xuống mọi hàm:

```python
from knee_mri.config import load_config
cfg = load_config("colab", overrides={"paths": {"data_root": "/content/..."}})
cfg.paths.ensure()
```

## Cấu trúc

```
configs/                 # cấu hình YAML theo môi trường
docs/                    # kiến trúc, ghi chú dữ liệu, chuẩn hóa, hướng dẫn refactor
notebooks/               # 01 phân tích dữ liệu · 02 huấn luyện Colab · 03 trực quan hóa
scripts/                 # entrypoint CLI mỏng, mỗi file một bước
tests/                   # 220 test, chạy được không cần GPU
src/knee_mri/
  config/                # schema dataclass + loader YAML
  constants.py           # LABELS, tên cột CSV — hằng số của đề bài
  data/                  # DICOM → volume, index CSV, Dataset, collate, chia tập
  labeling/              # weak label từ report (luật đa ngôn ngữ + VLM)
  models/                # student ViT-3D, teacher, SAM, sinh báo cáo
  training/              # loss, metric, checkpoint, vòng huấn luyện
  explain/               # CAM và đối chiếu với mask ROI
  inference/             # sinh submission
  pipeline/              # các bước S1/S2/S3 gọi được độc lập
  utils/                 # IO, mask, logging, seed
```

## Kiểm thử và chất lượng

```bash
make check          # đủ bộ như CI: ruff + mypy + pytest + notebook
make cov            # test kèm báo cáo độ phủ
make smoke          # chạy pipeline đầu-cuối trên tập nhỏ, không cần GPU
```

| Cổng | Trạng thái |
|---|---|
| `ruff check` + `ruff format` | sạch |
| `mypy` (49 file) | không lỗi |
| `pytest` | 220 test |
| Độ phủ | 80% |

Test khóa lại **từng** lỗi đã sửa trong đợt refactor, nên chúng không thể tái
phát một cách âm thầm. Các nhánh cần GPU và weights đã tải (`teachers/`,
`report_generator`) không chạy trong CI; phần logic thuần quanh chúng vẫn có
test — xem `tests/test_model_wrappers.py`.

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Sơ đồ pipeline S1 → S2 → S3, chi tiết từng bước |
| [docs/data_notes.md](docs/data_notes.md) | Quy mô dataset, nhãn thưa, report đa ngôn ngữ, chất lượng weak label |
| [docs/normalization.md](docs/normalization.md) | Vì sao chọn `volume_percentile_pm1` |
| [docs/refactor_guide.md](docs/refactor_guide.md) | Review đầy đủ bản cũ và kế hoạch refactor đã thực hiện |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Quy ước code, tám bất biến, cách viết test, chỗ dễ vấp |

## Bất biến cần giữ

Mỗi mục dưới đây từng là một lỗi thật và nay có test khóa lại. Bảng đối chiếu
test tương ứng nằm trong [CONTRIBUTING.md](CONTRIBUTING.md) §3.

1. `build_volume()` luôn trả `cfg.data.target_shape`, float32 — không ngoại lệ.
2. `Dataset.__getitem__` không đọc CSV, không nạp model, không gọi mạng.
3. Không có `from knee_mri.config import <HẰNG_SỐ>` ở cấp module.
4. Mọi hàm loss trả `torch.Tensor` có gradient, kể cả nhánh suy biến.
5. Mọi `nn.Parameter` tạo trong `__init__`, không bao giờ trong `forward`.
6. `torch.load` luôn có `weights_only=True`.
7. Notebook không định nghĩa hàm hay lớp.
8. Ảnh đi qua `to_unit_range()` trước khi vào model bên thứ ba (SAM, VLM).

## Giấy phép

Code: xem [LICENSE](LICENSE).
3DINO-ViT dùng CC BY-NC-ND — chỉ nghiên cứu phi thương mại, và **không sửa** code
trong `vendor/3DINO`.
