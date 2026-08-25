"""Đọc DICOM series thành mảng numpy thô — **không** chuẩn hóa, **không** resize.

Tách riêng khỏi :mod:`knee_mri.data.volume` để mỗi module một trách nhiệm: ở đây
chỉ có chuyện đọc file và sắp xếp lát cắt đúng thứ tự giải phẫu.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset

from knee_mri.constants import DICOM_SUFFIX
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)


class EmptySeriesError(RuntimeError):
    """Thư mục series không chứa file DICOM đọc được nào."""


def list_dicom_files(series_dir: Path) -> list[Path]:
    """Liệt kê file DICOM trong thư mục, đã sắp theo tên để thứ tự ổn định."""
    return sorted(p for p in series_dir.iterdir() if p.suffix.lower() == DICOM_SUFFIX)


def _slice_sort_key(dataset: FileDataset) -> tuple[float, float]:
    """Khóa sắp xếp: vị trí không gian, dự phòng bằng ``InstanceNumber``.

    ``ImagePositionPatient[2]`` là tọa độ dọc trục bệnh nhân — đáng tin hơn số
    thứ tự ảnh vì phản ánh hình học thật của chuỗi chụp.
    """
    position = getattr(dataset, "ImagePositionPatient", None)
    instance = getattr(dataset, "InstanceNumber", None)
    try:
        primary = float(position[2]) if position is not None else float("inf")
    except (TypeError, ValueError, IndexError):
        primary = float("inf")
    try:
        secondary = float(instance) if instance is not None else 0.0
    except (TypeError, ValueError):
        secondary = 0.0
    return primary, secondary


@lru_cache(maxsize=1)
def _apply_voi_lut() -> Callable[..., np.ndarray]:
    """Trả về hàm ``apply_voi_lut``, tương thích cả pydicom 2.x lẫn 3.x.

    Vị trí của hàm này đổi giữa hai phiên bản; cache kết quả để không phải thử
    import lại cho mỗi lát cắt.
    """
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom < 3.0
        from pydicom.pixel_data_handlers.util import (  # type: ignore[no-redef]
            apply_voi_lut,
        )
    return apply_voi_lut


def slice_pixels(dataset: FileDataset) -> np.ndarray:
    """Lấy mảng pixel của một lát cắt, đã áp VOI LUT nếu file có khai báo.

    Với MRI thì ``RescaleSlope``/``RescaleIntercept`` gần như luôn là 1/0 và
    khái niệm Hounsfield không áp dụng — nên bản cũ đặt tên hàm là ``_to_hu`` là
    sai (lỗi P2-6). Thứ *thực sự* có ý nghĩa cho MRI là VOI LUT / window-center,
    vốn được import trong bản cũ nhưng không hề gọi.
    """
    array = dataset.pixel_array
    if getattr(dataset, "VOILUTSequence", None) or hasattr(dataset, "WindowCenter"):
        try:
            array = _apply_voi_lut()(array, dataset)
        except (ValueError, AttributeError, IndexError) as exc:
            logger.debug("Bỏ qua VOI LUT (%s), dùng pixel thô.", exc)

    array = np.asarray(array, dtype=np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    if slope != 1.0 or intercept != 0.0:
        array = array * slope + intercept

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = array.max() - array  # MONOCHROME1: giá trị cao = tối
    return array


def read_series(series_dir: Path, *, max_slices: int | None = None) -> np.ndarray:
    """Đọc toàn bộ series thành mảng ``(D, H, W)`` float32 ở đơn vị gốc.

    Args:
        series_dir: Thư mục chứa các file ``.dcm`` của một series.
        max_slices: Nếu có, chỉ giữ ``max_slices`` lát ở **giữa** chuỗi. Vùng
            giữa chứa phần lớn giải phẫu đáng quan tâm của đầu gối, còn hai đầu
            thường là không khí và mô mềm ngoại vi.

    Raises:
        EmptySeriesError: khi không đọc được lát nào.
    """
    files = list_dicom_files(series_dir)
    if not files:
        raise EmptySeriesError(f"Không có file DICOM nào trong {series_dir}")

    datasets: list[FileDataset] = []
    for path in files:
        try:
            datasets.append(pydicom.dcmread(path))
        except Exception as exc:  # noqa: BLE001 - file hỏng thì bỏ qua, không dừng cả series
            logger.warning("Bỏ qua file DICOM lỗi %s: %s", path.name, exc)

    if not datasets:
        raise EmptySeriesError(f"Không đọc được file DICOM nào trong {series_dir}")

    datasets.sort(key=_slice_sort_key)
    if max_slices is not None and len(datasets) > max_slices:
        start = (len(datasets) - max_slices) // 2
        datasets = datasets[start : start + max_slices]

    planes = [slice_pixels(ds) for ds in datasets]
    shapes = {plane.shape for plane in planes}
    if len(shapes) > 1:
        # Series trộn nhiều độ phân giải: giữ nhóm shape phổ biến nhất.
        dominant = max(shapes, key=lambda s: sum(p.shape == s for p in planes))
        logger.warning(
            "Series %s có %d shape khác nhau, giữ lại %s.", series_dir.name, len(shapes), dominant
        )
        planes = [plane for plane in planes if plane.shape == dominant]

    return np.stack(planes, axis=0).astype(np.float32)


__all__ = ["EmptySeriesError", "list_dicom_files", "read_series", "slice_pixels"]
