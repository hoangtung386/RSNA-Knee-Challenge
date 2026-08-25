"""Chuẩn hóa cường độ MRI và đổi kích thước volume — **nguồn sự thật duy nhất**.

MRI không có thang đo tuyệt đối (khác CT với đơn vị Hounsfield): cùng một mô có
thể cho giá trị pixel rất khác nhau tùy scanner, chuỗi xung và hệ số khuếch đại.
Vì vậy phải chuẩn hóa theo thống kê nội tại của từng volume.

Module này được **cả** pipeline huấn luyện lẫn ``scripts/analyze_normalization.py``
dùng chung, loại bỏ tình trạng hai bản cài đặt song song bị lệch nhau (lỗi P3-8).

Toàn bộ hàm ở đây là numpy thuần — không phụ thuộc PyTorch, nên test chạy được
mà không cần môi trường GPU.
"""

from __future__ import annotations

from typing import Final

import numpy as np

#: Percentile dùng cho chuẩn hóa theo volume. Cắt đuôi để nhiễu xung và artifact
#: kim loại không kéo giãn toàn bộ dải giá trị.
VOLUME_PERCENTILES: Final[tuple[float, float]] = (0.5, 99.5)

#: Percentile dùng cho chuẩn hóa theo từng slice (chế độ cũ, giữ để so sánh).
SLICE_PERCENTILES: Final[tuple[float, float]] = (1.0, 99.0)

#: Ngưỡng cắt cho chế độ z-score.
ZSCORE_CLIP: Final[float] = 3.0

_EPS: Final[float] = 1e-8

#: Dải giá trị đầu ra của từng chế độ — dùng để quy đổi về ``[0, 1]`` khi cần
#: đưa ảnh cho model bên thứ ba (SAM, Gemma) vốn kỳ vọng ảnh hiển thị được.
OUTPUT_RANGES: Final[dict[str, tuple[float, float]]] = {
    "slice_percentile": (0.0, 1.0),
    "volume_percentile": (0.0, 1.0),
    "volume_percentile_pm1": (-1.0, 1.0),
    "volume_zscore": (-ZSCORE_CLIP, ZSCORE_CLIP),
    "none": (0.0, 1.0),
}

NORM_MODES: Final[tuple[str, ...]] = tuple(OUTPUT_RANGES)


class UnknownNormMode(ValueError):
    """Chế độ chuẩn hóa không được hỗ trợ."""

    def __init__(self, mode: str) -> None:
        super().__init__(f"norm_mode {mode!r} không hợp lệ. Hợp lệ: {', '.join(NORM_MODES)}")


def normalize_slice_percentile(volume: np.ndarray) -> np.ndarray:
    """Percentile 1–99 theo **từng slice** → ``[0, 1]``.

    Chế độ này thường *over-normalize*: nó xóa mất chênh lệch cường độ giữa các
    slice, vốn là tín hiệu thật trong chuỗi 3D. Giữ lại để đối chứng.
    """
    low = np.percentile(volume, SLICE_PERCENTILES[0], axis=(1, 2), keepdims=True)
    high = np.percentile(volume, SLICE_PERCENTILES[1], axis=(1, 2), keepdims=True)
    scaled = (volume - low) / (high - low + _EPS)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def normalize_volume_percentile(volume: np.ndarray, *, to_pm1: bool = False) -> np.ndarray:
    """Percentile 0.5–99.5 trên **toàn volume** → ``[0, 1]`` hoặc ``[-1, 1]``.

    Đây là chế độ được khuyến nghị: nhất quán xuyên dataset (không phụ thuộc
    scanner), giữ được biến thiên giữa các slice, và biến thể ``[-1, 1]`` khớp
    đúng định dạng đầu vào của 3DINO-ViT.
    """
    low, high = np.percentile(volume, VOLUME_PERCENTILES)
    scaled = np.clip((volume - low) / (high - low + _EPS), 0.0, 1.0)
    if to_pm1:
        scaled = scaled * 2.0 - 1.0
    return scaled.astype(np.float32)


def normalize_volume_zscore(volume: np.ndarray, *, clip: float = ZSCORE_CLIP) -> np.ndarray:
    """Z-score toàn volume, cắt tại ``±clip``.

    Kém ổn định khi có outlier mạnh (artifact kim loại), vì trung bình và độ lệch
    chuẩn đều bị kéo lệch.
    """
    mean = float(volume.mean())
    std = float(volume.std()) + _EPS
    return np.clip((volume - mean) / std, -clip, clip).astype(np.float32)


def normalize(volume: np.ndarray, mode: str) -> np.ndarray:
    """Điều phối tới hàm chuẩn hóa tương ứng.

    Args:
        volume: Mảng ``(D, H, W)`` ở đơn vị gốc của scanner.
        mode: Một trong :data:`NORM_MODES`.

    Raises:
        UnknownNormMode: khi ``mode`` không hợp lệ — thà lỗi to còn hơn âm thầm
            trả về dữ liệu chưa chuẩn hóa như bản cũ.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if mode == "slice_percentile":
        return normalize_slice_percentile(volume)
    if mode == "volume_percentile":
        return normalize_volume_percentile(volume, to_pm1=False)
    if mode == "volume_percentile_pm1":
        return normalize_volume_percentile(volume, to_pm1=True)
    if mode == "volume_zscore":
        return normalize_volume_zscore(volume)
    if mode == "none":
        return volume
    raise UnknownNormMode(mode)


def to_unit_range(array: np.ndarray, mode: str) -> np.ndarray:
    """Quy đổi mảng đã chuẩn hóa về ``[0, 1]`` bất kể chế độ nào sinh ra nó.

    Bắt buộc dùng trước khi đưa ảnh cho SAM hoặc Gemma. Bản cũ cast thẳng
    ``(volume * 255).astype(np.uint8)`` trên dữ liệu nằm trong ``[-1, 1]``, khiến
    mọi giá trị âm bị *wrap-around* thành số lớn — ảnh gửi cho model là nhiễu
    (lỗi P2-4 và P2-5).
    """
    if mode not in OUTPUT_RANGES:
        raise UnknownNormMode(mode)
    low, high = OUTPUT_RANGES[mode]
    scaled = (np.asarray(array, dtype=np.float32) - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def to_uint8_image(array: np.ndarray, mode: str) -> np.ndarray:
    """Đổi slice đã chuẩn hóa thành ảnh xám uint8 hiển thị được."""
    return (to_unit_range(array, mode) * 255.0).round().astype(np.uint8)


def to_rgb_image(array: np.ndarray, mode: str) -> np.ndarray:
    """Đổi slice ``(H, W)`` thành ảnh RGB ``(H, W, 3)`` uint8 cho SAM/VLM."""
    gray = to_uint8_image(array, mode)
    return np.repeat(gray[..., None], 3, axis=-1)


def _resize_axis(array: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    """Nội suy tuyến tính dọc một trục, quy ước half-pixel center.

    Tương đương ``torch.nn.functional.interpolate(..., align_corners=False)``
    nhưng chỉ dùng numpy, nên lớp dữ liệu không phải phụ thuộc PyTorch.
    """
    in_len = array.shape[axis]
    if in_len == out_len:
        return array
    if in_len == 1:
        return np.repeat(array, out_len, axis=axis)

    scale = in_len / out_len
    centers = (np.arange(out_len, dtype=np.float64) + 0.5) * scale - 0.5
    centers = np.clip(centers, 0.0, in_len - 1)
    low_idx = np.floor(centers).astype(np.intp)
    high_idx = np.minimum(low_idx + 1, in_len - 1)

    weight_shape = [1] * array.ndim
    weight_shape[axis] = out_len
    weight = (centers - low_idx).astype(np.float32).reshape(weight_shape)

    low = np.take(array, low_idx, axis=axis)
    high = np.take(array, high_idx, axis=axis)
    return low * (1.0 - weight) + high * weight


def resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Đổi kích thước volume ``(D, H, W)`` bằng nội suy trilinear trên float32.

    Bản cũ đi đường vòng float → uint8 → ``PIL.Image.resize`` → float, làm mất
    dải động của MRI 12–16 bit xuống còn 8 bit (lỗi P1-6). Ở đây phép nội suy
    chạy thẳng trên float32 nên không mất thông tin ngoài bản thân việc lấy mẫu.

    Args:
        volume: Mảng ``(D, H, W)``.
        target_shape: Kích thước đích ``(D, H, W)``.
    """
    if volume.ndim != 3:
        raise ValueError(f"resize_volume cần mảng 3 chiều (D,H,W), nhận được shape {volume.shape}.")
    result = np.asarray(volume, dtype=np.float32)
    for axis, out_len in enumerate(target_shape):
        result = _resize_axis(result, int(out_len), axis)
    return np.ascontiguousarray(result, dtype=np.float32)


__all__ = [
    "NORM_MODES",
    "OUTPUT_RANGES",
    "SLICE_PERCENTILES",
    "VOLUME_PERCENTILES",
    "ZSCORE_CLIP",
    "UnknownNormMode",
    "normalize",
    "normalize_slice_percentile",
    "normalize_volume_percentile",
    "normalize_volume_zscore",
    "resize_volume",
    "to_rgb_image",
    "to_uint8_image",
    "to_unit_range",
]
