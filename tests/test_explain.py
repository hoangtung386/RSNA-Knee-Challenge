"""Test đối chiếu CAM với mask ROI."""

from __future__ import annotations

import numpy as np
import pytest

from knee_mri.explain.overlay import OverlapReport, binarise_cam, cam_mask_overlap


@pytest.fixture
def block_cam() -> np.ndarray:
    """CAM tập trung vào một khối ở giữa."""
    cam = np.zeros((4, 10, 10), dtype=np.float32)
    cam[:, 3:7, 3:7] = 1.0
    return cam


@pytest.fixture
def block_mask() -> np.ndarray:
    mask = np.zeros((4, 10, 10), dtype=np.uint8)
    mask[:, 3:7, 3:7] = 1
    return mask


class TestBinariseCam:
    def test_returns_mask_and_threshold(self, block_cam: np.ndarray) -> None:
        mask, threshold = binarise_cam(block_cam, percentile=80.0)
        assert mask.dtype == np.uint8
        assert mask.shape == block_cam.shape
        assert isinstance(threshold, float)

    def test_higher_percentile_selects_less(self, block_cam: np.ndarray) -> None:
        loose, _ = binarise_cam(block_cam, percentile=50.0)
        strict, _ = binarise_cam(block_cam, percentile=95.0)
        assert strict.sum() <= loose.sum()

    @pytest.mark.parametrize("skew", [1.0, 3.0, 10.0], ids=["đều", "lệch vừa", "lệch mạnh"])
    def test_selected_fraction_is_predictable(self, skew: float) -> None:
        """Percentile luôn chọn đúng tỉ lệ voxel, bất kể phân bố lệch thế nào.

        Đây chính là lý do dùng percentile thay cho ``cam > cam.mean()`` như bản
        cũ: attention rollout có phân bố lệch phải, nên ngưỡng trung bình chọn ra
        một tỉ lệ **không đoán trước được** — thay đổi theo từng ảnh, khiến chỉ
        số IoU giữa các study không so sánh được với nhau.
        """
        rng = np.random.default_rng(0)
        cam = rng.random((4, 20, 20)).astype(np.float32) ** skew

        selected, _ = binarise_cam(cam, percentile=80.0)
        assert selected.mean() == pytest.approx(0.20, abs=0.01)

    def test_mean_threshold_fraction_drifts_with_skew(self) -> None:
        """Đối chứng: ngưỡng trung bình cho tỉ lệ trôi theo độ lệch của phân bố."""
        rng = np.random.default_rng(0)
        fractions = []
        for skew in (1.0, 10.0):
            cam = rng.random((4, 20, 20)).astype(np.float32) ** skew
            fractions.append(float((cam > cam.mean()).mean()))
        assert abs(fractions[0] - fractions[1]) > 0.10


class TestCamMaskOverlap:
    def test_perfect_overlap(self, block_cam: np.ndarray, block_mask: np.ndarray) -> None:
        report = cam_mask_overlap(block_cam, block_mask, percentile=50.0)
        assert isinstance(report, OverlapReport)
        assert report.iou == pytest.approx(1.0)
        assert report.dice == pytest.approx(1.0)

    def test_disjoint_regions_score_zero(self, block_cam: np.ndarray) -> None:
        elsewhere = np.zeros((4, 10, 10), dtype=np.uint8)
        elsewhere[:, 0:2, 0:2] = 1
        report = cam_mask_overlap(block_cam, elsewhere, percentile=50.0)
        assert report is not None
        assert report.iou == pytest.approx(0.0)

    def test_dice_is_at_least_iou(self, block_cam: np.ndarray) -> None:
        """Quan hệ toán học: Dice ≥ IoU với mọi cặp tập hợp."""
        partial = np.zeros((4, 10, 10), dtype=np.uint8)
        partial[:, 5:9, 5:9] = 1
        report = cam_mask_overlap(block_cam, partial, percentile=50.0)
        assert report is not None
        assert report.dice >= report.iou

    @pytest.mark.parametrize(
        "mask",
        [None, np.zeros((4, 10, 10), dtype=np.uint8), np.ones((2, 2, 2), dtype=np.uint8)],
        ids=["không có mask", "mask rỗng", "mask lệch shape"],
    )
    def test_returns_none_when_nothing_to_compare(self, block_cam: np.ndarray, mask) -> None:
        assert cam_mask_overlap(block_cam, mask) is None

    def test_report_repr_is_readable(self, block_cam: np.ndarray, block_mask: np.ndarray) -> None:
        assert "IoU" in repr(cam_mask_overlap(block_cam, block_mask))
