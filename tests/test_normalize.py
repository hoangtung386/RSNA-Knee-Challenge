"""Test chuẩn hóa cường độ và đổi kích thước volume."""

from __future__ import annotations

import numpy as np
import pytest

from knee_mri.data.normalize import (
    NORM_MODES,
    OUTPUT_RANGES,
    UnknownNormMode,
    normalize,
    resize_volume,
    to_rgb_image,
    to_uint8_image,
    to_unit_range,
)


@pytest.fixture
def raw_volume() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((12, 40, 50), dtype=np.float32) * 4000).astype(np.float32)


@pytest.mark.parametrize("mode", NORM_MODES)
def test_normalize_respects_declared_range(raw_volume: np.ndarray, mode: str) -> None:
    """Mỗi chế độ phải cho ra giá trị nằm trong dải nó khai báo."""
    result = normalize(raw_volume, mode)
    low, high = OUTPUT_RANGES[mode]
    if mode != "none":
        assert result.min() >= low - 1e-5
        assert result.max() <= high + 1e-5
    assert result.dtype == np.float32


def test_normalize_rejects_unknown_mode(raw_volume: np.ndarray) -> None:
    """Chế độ sai phải báo lỗi thay vì âm thầm trả về dữ liệu chưa chuẩn hóa."""
    with pytest.raises(UnknownNormMode):
        normalize(raw_volume, "khong_ton_tai")


@pytest.mark.parametrize("mode", NORM_MODES)
def test_to_unit_range_always_zero_one(raw_volume: np.ndarray, mode: str) -> None:
    """P2-4/P2-5: ảnh đưa cho SAM/VLM luôn phải nằm trong [0, 1]."""
    unit = to_unit_range(normalize(raw_volume, mode), mode)
    assert unit.min() >= 0.0
    assert unit.max() <= 1.0


def test_pm1_volume_does_not_wrap_around_as_uint8(raw_volume: np.ndarray) -> None:
    """P2-5: giá trị âm không được wrap-around khi cast uint8.

    Bản cũ làm ``(volume * 255).astype(np.uint8)`` trên dữ liệu [-1, 1]: voxel
    tối nhất (-1.0) thành 1 thay vì 0, và các giá trị âm khác nhảy lên số lớn.
    """
    normalised = normalize(raw_volume, "volume_percentile_pm1")
    assert normalised.min() < 0.0, "fixture phải có giá trị âm để test có ý nghĩa"

    correct = to_uint8_image(normalised, "volume_percentile_pm1")
    buggy = (normalised * 255).astype(np.uint8)

    assert correct.min() == 0
    assert correct.max() == 255
    assert not np.array_equal(correct, buggy)


def test_to_rgb_image_shape(raw_volume: np.ndarray) -> None:
    normalised = normalize(raw_volume, "volume_percentile_pm1")
    rgb = to_rgb_image(normalised[0], "volume_percentile_pm1")
    assert rgb.shape == (*normalised.shape[1:], 3)
    assert rgb.dtype == np.uint8


@pytest.mark.parametrize(
    "source,target",
    [((10, 40, 50), (8, 32, 32)), ((3, 8, 8), (16, 64, 64)), ((8, 32, 32), (8, 32, 32))],
)
def test_resize_volume_hits_exact_shape(source, target) -> None:
    rng = np.random.default_rng(1)
    result = resize_volume(rng.random(source, dtype=np.float32), target)
    assert result.shape == target
    assert result.dtype == np.float32


def test_resize_preserves_constant_field() -> None:
    """Nội suy một trường hằng số phải trả lại đúng hằng số đó."""
    constant = np.full((4, 8, 8), 0.37, dtype=np.float32)
    assert np.allclose(resize_volume(constant, (9, 16, 3)), 0.37, atol=1e-6)


def test_resize_keeps_full_precision() -> None:
    """P1-6: resize không được đi qua uint8 (chỉ 256 mức).

    Dựng volume có nhiều giá trị riêng biệt hơn 256; sau khi resize giữ nguyên
    kích thước, số giá trị riêng biệt phải vẫn vượt 256.
    """
    values = np.linspace(0.0, 1.0, 4096, dtype=np.float32).reshape(4, 32, 32)
    result = resize_volume(values, (4, 32, 32))
    assert len(np.unique(result)) > 256
