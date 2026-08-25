"""Dựng volume 3D chuẩn hóa, kích thước **cố định** từ một series DICOM.

Bất biến trung tâm của toàn bộ pipeline::

    build_volume(...).shape == cfg.data.target_shape

Bản cũ không có bất biến này: :func:`~knee_mri.utils.mask.roi_crop` trả về shape
tùy ý theo bounding box của mask, khiến (a) ``torch.cat`` trong collate vỡ vì H/W
lệch nhau, và (b) số patch của ViT-3D đổi theo từng batch, kéo theo việc tái tạo
``pos_embed`` trong ``forward`` — hai lỗi P0-6 và P0-5. Cố định shape tại đây xử
lý tận gốc cả hai.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from knee_mri.config.schema import DataCfg
from knee_mri.data.dicom_io import read_series
from knee_mri.data.normalize import normalize, resize_volume
from knee_mri.utils.mask import roi_crop

#: Đọc dư một ít lát so với ``max_slices`` trước khi nội suy, để phép lấy mẫu
#: dọc trục D có đủ dữ liệu nguồn thay vì chỉ nhân bản lát.
_READ_SLICE_MULTIPLIER: int = 3


def build_volume(
    series_dir: Path,
    cfg: DataCfg,
    *,
    mask: np.ndarray | None = None,
    roi_margin: int = 4,
) -> np.ndarray:
    """Đọc một series DICOM thành volume chuẩn hóa có shape cố định.

    Trình tự — thứ tự các bước là cố ý:

    1. Đọc DICOM, sắp lát theo vị trí giải phẫu.
    2. Cắt ROI theo mask (nếu có) — làm **trước** khi resize để độ phân giải
       hiệu dụng dồn vào vùng bệnh lý thay vì vào nền đen.
    3. Resize trilinear trên float32 về ``cfg.target_shape``.
    4. Chuẩn hóa cường độ — làm **sau** resize để thống kê percentile được tính
       trên đúng tập voxel mà model sẽ nhìn thấy.

    Args:
        series_dir: Thư mục chứa file ``.dcm``.
        cfg: Phần ``data`` của cấu hình.
        mask: Mask ROI ``(D, H, W)`` cùng shape với volume thô, hoặc ``None``.
        roi_margin: Số voxel nới thêm quanh bounding box của mask.

    Returns:
        Mảng float32 có shape đúng bằng ``cfg.target_shape``.
    """
    raw = read_series(series_dir, max_slices=cfg.max_slices * _READ_SLICE_MULTIPLIER)
    if mask is not None:
        raw = roi_crop(raw, mask, margin=roi_margin)

    resized = resize_volume(raw, cfg.target_shape)
    volume = normalize(resized, cfg.norm_mode)

    if volume.shape != cfg.target_shape:  # pragma: no cover - bất biến, không nên xảy ra
        raise AssertionError(
            f"build_volume vi phạm bất biến shape: {volume.shape} != {cfg.target_shape}"
        )
    return volume


def empty_volume(cfg: DataCfg) -> np.ndarray:
    """Volume rỗng đúng shape, dùng khi một study không có series đọc được."""
    return np.zeros(cfg.target_shape, dtype=np.float32)


def middle_slices(volume: np.ndarray, count: int) -> np.ndarray:
    """Lấy ``count`` lát cắt phân bố đều quanh giữa volume.

    Dùng cho VLM: đưa vài lát đại diện thay vì cả khối, vừa đủ ngữ cảnh thị giác
    mà không làm nổ số token.
    """
    depth = volume.shape[0]
    count = max(1, min(count, depth))
    lo = depth / 2 - depth / 4
    hi = depth / 2 + depth / 4
    indices = np.clip(np.linspace(lo, hi, count).round().astype(int), 0, depth - 1)
    return volume[indices]


__all__ = ["build_volume", "empty_volume", "middle_slices"]
