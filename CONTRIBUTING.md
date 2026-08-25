# Hướng dẫn đóng góp

Tài liệu cho người mới tiếp nhận repo. Đọc hết một lượt trước khi sửa dòng code đầu tiên.

## 1. Dựng môi trường

```bash
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev,viz,eval]" + pre-commit install
make check            # lint + test + kiểm tra notebook — phải xanh trước khi bắt đầu
```

Không có `make`? Các lệnh tương đương nằm trong [Makefile](Makefile).

Dataset **không** nằm trong repo. Test nào cần dữ liệu thật sẽ tự `skip`, nên
`make test` chạy được ngay cả khi chưa có `dataset/`.

## 2. Quy ước code

Ép tự động bởi `ruff` (cấu hình trong [pyproject.toml](pyproject.toml)):

- PEP 8, độ dài dòng tối đa **100**.
- Bắt buộc `from __future__ import annotations` ở đầu mỗi module.
- Import sắp xếp bởi `ruff` (isort), `knee_mri` là first-party.
- Type hint cho mọi tham số và giá trị trả về của hàm public.

Không ép được bằng máy, nhưng vẫn là quy ước của repo:

- **Docstring viết bằng tiếng Việt**, kiểu Google (`Args:` / `Returns:` / `Raises:`).
- Docstring trả lời **vì sao**, không phải **cái gì**. `# tăng i lên 1` là vô ích;
  `# dùng percentile vì phân bố attention lệch phải` mới đáng viết.
- Tên biến viết đủ chữ: `probabilities` chứ không `probs`, `volume` chứ không `vol`.
- Khi sửa một quyết định thiết kế cũ, ghi lại lý do ngay tại chỗ để người sau
  không "sửa ngược" lại.

## 3. Tám bất biến không được phá

Mỗi mục dưới đây từng là một lỗi thật trong repo này. Mỗi mục đều có test khóa lại.
Nếu bạn thấy mình đang định vi phạm một trong số này, gần như chắc chắn có cách khác.

| # | Bất biến | Test bảo vệ |
|---|---|---|
| 1 | `build_volume()` luôn trả `cfg.data.target_shape`, float32 | `test_data.py::TestVolumeShapeInvariant` |
| 2 | `Dataset.__getitem__` không đọc CSV, không nạp model, không gọi mạng | `test_data.py::TestCatalog::test_lookups_do_not_touch_disk` |
| 3 | Không `from knee_mri.config import <HẰNG_SỐ>` ở cấp module | `test_config.py::TestOverridesActuallyApply` |
| 4 | Hàm loss luôn trả `Tensor` có gradient, kể cả nhánh suy biến | `test_losses.py::TestAucMarginLoss` |
| 5 | `nn.Parameter` tạo trong `__init__`, không bao giờ trong `forward` | `test_student.py::TestPositionEmbedding` |
| 6 | `torch.load` luôn có `weights_only=True` | `test_pipeline_smoke.py::TestCheckpointRoundTrip` |
| 7 | Notebook không định nghĩa hàm hay lớp | rà bằng mắt khi review |
| 8 | Ảnh qua `to_unit_range()` trước khi vào SAM/VLM | `test_normalize.py`, `test_model_wrappers.py` |

Bối cảnh đầy đủ của từng lỗi nằm ở [docs/refactor_guide.md](docs/refactor_guide.md).

### Vì sao bất biến #1 quan trọng nhất

Shape cố định là thứ giữ cho mọi thứ khác đứng vững: nó cố định số patch của
ViT-3D (nên `pos_embed` cấp phát được một lần trong `__init__`), khiến collate
chỉ cần `np.stack`, và loại bỏ mọi phép pad động. Nới lỏng nó là kéo theo cả
chuỗi lỗi.

## 4. Cấu hình

Cấu hình là **dữ liệu**, không phải code. Thêm tham số mới:

1. Thêm khóa vào [configs/base.yaml](configs/base.yaml).
2. Thêm field vào dataclass tương ứng trong `src/knee_mri/config/schema.py`.
3. Đọc nó trong hàm `_build_*` tương ứng ở `src/knee_mri/config/loader.py`.
4. **Thực sự dùng nó ở đâu đó.** Field khai mà không ai đọc là bẫy: người dùng
   chỉnh YAML, không thấy gì thay đổi, và mất hàng giờ để hiểu tại sao.

Không bao giờ nhận cấu hình bằng cách import hằng số cấp module — giá trị bị sao
chép lúc import và mọi ghi đè sau đó bị nuốt im lặng. Luôn truyền `cfg` tường minh.

## 5. Viết test

- Đặt cạnh module tương ứng: `src/knee_mri/data/volume.py` → `tests/test_data.py`.
- Test cần dữ liệu thật dùng fixture `real_series_dir` (tự `skip` nếu thiếu).
- Test cần torch mở đầu bằng `torch = pytest.importorskip("torch")`.
- Fixture `cfg` là **riêng cho từng test** (artifacts trong `tmp_path`); dùng
  `session_cfg` khi chỉ cần đọc.
- Tên test mô tả **hành vi**, không phải hàm: `test_easy_negative_costs_less_than_hard_negative`
  chứ không `test_asl`.
- Sửa một lỗi thì viết test làm lỗi đó tái hiện **trước**, rồi mới sửa.

Độ phủ hiện tại ~80%. Phần thiếu chủ yếu là các nhánh cần GPU và weights đã tải
(`teachers/`, `report_generator`, `pipeline/features`) — logic thuần quanh chúng
vẫn phải có test, xem `tests/test_model_wrappers.py`.

## 6. Chạy pipeline

```bash
make smoke                                  # đầu-cuối trên tập nhỏ, không cần GPU
python -m knee_mri.cli                      # danh sách lệnh
python scripts/train.py --env colab --set train.epochs=50
```

Mọi bước đều **idempotent**: chạy lại sẽ bỏ qua phần đã cache. Dùng `--overwrite`
để buộc tính lại. Đây là điều kiện để nối tiếp một job dài bị ngắt giữa chừng.

## 7. Trước khi mở pull request

```bash
make check
```

Trong phần mô tả PR, nêu rõ:

- Vấn đề đang giải quyết và **vì sao** chọn cách đó.
- Bất biến nào bị chạm tới (nếu có) và lý do vẫn an toàn.
- Chỉ số đo được, nếu thay đổi ảnh hưởng tới chất lượng model. "Cải thiện gán
  nhãn" là vô nghĩa; "F1 trên 58 study nhãn gold: 0.549 → 0.581" thì có nghĩa.

## 8. Những chỗ dễ vấp

- **Model ID của VLM** ([configs/base.yaml](configs/base.yaml) → `vlm.model_id`)
  chưa được xác minh trên HuggingFace Hub. Kiểm tra trước khi chạy dài trên A100.
- **`model.guidance_dim`** phải khớp `hidden_size` thật của VLM. Sai thì
  `GemmaGuidanceEncoder.encode` ném lỗi ngay ở bước precompute — cố ý như vậy,
  để không lẫn file `.npy` lệch chiều rồi vỡ giữa lúc huấn luyện.
- **`vendor/3DINO`** mang license CC BY-NC-ND: chỉ nghiên cứu phi thương mại và
  **không được sửa** code trong đó.
- **Loss AUC-margin** cần mỗi batch có cả mẫu dương lẫn mẫu âm cho ít nhất một
  nhãn. Với tỉ lệ dương ~1–2% và batch nhỏ, phần lớn batch sẽ không thỏa. Tăng
  `train.batch_size` hoặc thêm sampler cân bằng nếu muốn loss này thực sự hoạt động.
