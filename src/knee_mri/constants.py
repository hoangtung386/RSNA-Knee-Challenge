"""Hằng số bất biến của bài toán — không phải cấu hình tùy chỉnh được.

Những giá trị ở đây do đề bài / định dạng dữ liệu quy định (thứ tự 12 cột nhãn,
tên cột CSV, số chiều vector metadata). Chúng không bao giờ thay đổi giữa các
môi trường, nên đặt trong code thay vì YAML.

Mọi thứ *có thể* tinh chỉnh (hyperparameter, đường dẫn, model id) nằm trong
``configs/*.yaml`` và được nạp qua :func:`knee_mri.config.load_config`.
"""

from __future__ import annotations

from typing import Final

#: 12 bệnh lý mục tiêu, đúng thứ tự cột của ``train.csv``/``sample_submission.csv``.
LABELS: Final[tuple[str, ...]] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

NUM_LABELS: Final[int] = len(LABELS)

#: Ba mặt phẳng giải phẫu xuất hiện trong ``*_series.csv``.
PLANES: Final[tuple[str, ...]] = ("Sagittal", "Axial", "Coronal")

#: one-hot mặt phẳng (3) + Fluid_Sensitive (1) + Fat_Suppression (1).
METADATA_DIM: Final[int] = len(PLANES) + 2

# ---- Tên cột trong các file CSV của cuộc thi ----
COL_STUDY: Final[str] = "StudyInstanceUID"
COL_SERIES: Final[str] = "SeriesInstanceUID"
COL_REPORT: Final[str] = "Report"
COL_PLANE: Final[str] = "Anatomical_Plane"
COL_FLUID: Final[str] = "Fluid_Sensitive"
COL_FAT: Final[str] = "Fat_Suppression"

# ---- Tên file/thư mục trong dataset ----
TRAIN_CSV: Final[str] = "train.csv"
TRAIN_SERIES_CSV: Final[str] = "train_series.csv"
TEST_CSV: Final[str] = "test.csv"
TEST_SERIES_CSV: Final[str] = "test_series.csv"
TRAIN_SERIES_DIR: Final[str] = "train_series"
TEST_SERIES_DIR: Final[str] = "test_series"

#: Đuôi file DICOM trong dataset.
DICOM_SUFFIX: Final[str] = ".dcm"

__all__ = [
    "COL_FAT",
    "COL_FLUID",
    "COL_PLANE",
    "COL_REPORT",
    "COL_SERIES",
    "COL_STUDY",
    "DICOM_SUFFIX",
    "LABELS",
    "METADATA_DIM",
    "NUM_LABELS",
    "PLANES",
    "TEST_CSV",
    "TEST_SERIES_CSV",
    "TEST_SERIES_DIR",
    "TRAIN_CSV",
    "TRAIN_SERIES_CSV",
    "TRAIN_SERIES_DIR",
]
