# Ghi chú về dataset

Số liệu dưới đây đo trực tiếp trên `dataset/` bằng `StudyCatalog` và
`knee_mri.labeling`, không phải trích từ tài liệu đề bài.

## Quy mô

| Chỉ số | Giá trị |
|---|---|
| Study (train) | 4407 |
| Series (train) | 24371 |
| Study có `Report` | 4407 (100%) |
| Study có nhãn cấu trúc | **58** (1.3%) |
| Series trung bình / study | ~5.5 |

## Nhãn cực thưa

Số study dương tính trên **58 study có nhãn gold**:

| Nhãn | Dương tính |
|---|---|
| Effusion | 35 |
| Synovitis | 27 |
| Medial Meniscus | 26 |
| ACL | 23 |
| Lateral Meniscus | 23 |
| PF OA | 21 |
| Contusion | 19 |
| Fracture | 18 |
| Medial OA | 15 |
| Baker's | 12 |
| Lateral OA | 11 |
| MCL | 9 |

Hệ quả thiết kế:

- Dùng **Asymmetric Loss** thay BCE — hạ trọng số mẫu âm dễ để gradient dồn vào
  mẫu dương hiếm.
- Metric là **AUC**, không phải accuracy: đoán "âm" cho mọi thứ đã cho accuracy
  rất cao mà vô dụng.
- 58 study này là **validation**, không dùng để huấn luyện (`data/splits.py`).

## Report đa ngôn ngữ

Ước lượng bằng từ khóa đặc trưng:

| Ngôn ngữ | Số study |
|---|---|
| Anh | ~1591 |
| Tây Ban Nha | ~701 |
| Hà Lan | ~303 |
| Đức | ~244 |
| Pháp | ~80 |
| Không xác định rõ | ~1488 |

Ví dụ cùng ý "không có tràn dịch" trong 5 ngôn ngữ:

```
en  No evidence of joint effusion.
es  Sin derrame articular.
nl  Geen vocht in het gewricht.
de  Kein Gelenkerguss.
fr  Pas de épanchement articulaire.
```

Bộ gán nhãn phải nhận ra **cả năm** là phủ định. Bản trước refactor chỉ nhận
`"no "`, `"without"`, `"negativ"` (tiếng Anh), nên gán dương tính sai cho hàng
trăm study.

## Chất lượng weak label

Đo trên 58 study có nhãn gold, so bản luật từ khóa cũ và mới:

| Chỉ số | Trước refactor | Sau refactor |
|---|---|---|
| Precision | 0.421 | **0.566** |
| Recall | 0.454 | **0.533** |
| F1 | 0.437 | **0.549** (+25.7%) |
| False positive | 150 | **98** |
| Study mà `Medial Meniscus == Lateral Meniscus` | **58/58 (100%)** | 47/58 |

Dòng cuối là bằng chứng trực tiếp cho lỗi P2-1: hai nhãn sụn chêm dùng chung từ
khóa trần `"menisc"` nên **luôn** nhận giá trị giống hệt nhau — 2/12 nhãn vô giá trị.

Tái lập số liệu này:

```bash
python scripts/precompute_weak_labels.py --env local
```

## Metadata series

`Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression` có ở **cả**
`train_series.csv` lẫn `test_series.csv`. Đây là đầu vào phụ trợ hợp lệ lúc
inference, khác với report (chỉ có lúc huấn luyện).

Thứ tự ưu tiên khi chọn series (`data/series_selection.py`): Sagittal (×3) >
nhạy dịch (×2) > xóa mỡ (×1). Sagittal tốt nhất cho ACL và sụn chêm; chuỗi nhạy
dịch làm nổi tràn dịch và phù tủy xương.

## Đặc điểm DICOM

- Đuôi file: `.dcm`; mỗi series có khoảng 20–40 lát.
- Độ phân giải điển hình: 512×512, đôi khi lẫn nhiều độ phân giải trong một
  series (`dicom_io.read_series` giữ nhóm shape phổ biến nhất).
- Sắp xếp lát theo `ImagePositionPatient[2]`, dự phòng bằng `InstanceNumber`.
- MRI **không** có thang đo tuyệt đối — xem `docs/normalization.md`.
