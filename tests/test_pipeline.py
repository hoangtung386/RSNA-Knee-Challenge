"""Test các bước pipeline chạy được **không cần GPU hay model nặng**.

Mỗi bước phải suy giảm êm khi thiếu model (SAM, VLM, 3DINO) thay vì làm hỏng cả
job — đây là điều kiện để một thành viên mới clone repo về là chạy được ngay.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config  # noqa: E402
from knee_mri.constants import LABELS  # noqa: E402
from knee_mri.data.catalog import StudyCatalog  # noqa: E402
from knee_mri.data.dataset import describe_cache  # noqa: E402
from knee_mri.models.sam_masker import SamMaskGenerator, center_bbox  # noqa: E402
from knee_mri.pipeline import generate_masks, precompute_weak_labels  # noqa: E402
from knee_mri.utils.mask import load_mask  # noqa: E402


@pytest.fixture(scope="module")
def real_catalog(session_cfg) -> StudyCatalog:
    if not session_cfg.paths.train_csv.is_file():
        pytest.skip("Không có dataset cục bộ.")
    return StudyCatalog.from_csv(session_cfg.paths.train_csv, session_cfg.paths.train_series_csv)


class TestCenterBbox:
    def test_box_is_centred(self) -> None:
        assert center_bbox(100, 200, 0.5) == [50, 25, 150, 75]

    def test_fraction_controls_size(self) -> None:
        small = center_bbox(100, 100, 0.2)
        large = center_bbox(100, 100, 0.8)
        assert (small[2] - small[0]) < (large[2] - large[0])


class TestMaskStep:
    """SAM không cài được thì phải lùi về mask hộp trung tâm, không được crash."""

    def test_falls_back_when_sam_unavailable(self, cfg: Config, volume: np.ndarray) -> None:
        masker = SamMaskGenerator(cfg, device="cpu")
        mask = masker.generate_3d(volume)
        assert mask.shape == volume.shape
        assert mask.dtype == np.uint8
        assert 0 < mask.sum() < mask.size, "mask fallback phải không rỗng và không phủ hết"

    def test_writes_one_mask_per_selected_series(
        self, cfg: Config, real_catalog: StudyCatalog
    ) -> None:
        uids = real_catalog.studies_with_series()[:2]
        written = generate_masks(cfg, real_catalog, uids, device="cpu")
        assert written > 0
        series = real_catalog.series(uids[0])[0]
        assert load_mask(uids[0], series.series_uid, cfg.paths.masks) is not None

    def test_is_idempotent(self, cfg: Config, real_catalog: StudyCatalog) -> None:
        """Chạy lại không được tính lại — cho phép nối tiếp job bị ngắt giữa chừng."""
        uids = real_catalog.studies_with_series()[:2]
        generate_masks(cfg, real_catalog, uids, device="cpu")
        assert generate_masks(cfg, real_catalog, uids, device="cpu") == 0

    def test_overwrite_forces_recompute(self, cfg: Config, real_catalog: StudyCatalog) -> None:
        uids = real_catalog.studies_with_series()[:1]
        generate_masks(cfg, real_catalog, uids, device="cpu")
        assert generate_masks(cfg, real_catalog, uids, device="cpu", overwrite=True) > 0


class TestWeakLabelStep:
    def test_writes_json_per_study(self, cfg: Config, real_catalog: StudyCatalog) -> None:
        uids = real_catalog.studies_with_series()[:3]
        assert precompute_weak_labels(cfg, real_catalog, uids) == 3

        payload = json.loads((cfg.paths.weak_labels / f"{uids[0]}.json").read_text("utf-8"))
        assert set(payload) == set(LABELS)
        assert all(value in (0, 1) for value in payload.values())

    def test_falls_back_when_vlm_unavailable(self, cfg: Config, real_catalog: StudyCatalog) -> None:
        """VLM lỗi → dùng luật từ khóa, job vẫn hoàn tất."""
        uids = real_catalog.studies_with_series()[:2]
        written = precompute_weak_labels(cfg, real_catalog, uids, use_vlm=True, overwrite=True)
        assert written == 2

    def test_is_idempotent(self, cfg: Config, real_catalog: StudyCatalog) -> None:
        uids = real_catalog.studies_with_series()[:3]
        precompute_weak_labels(cfg, real_catalog, uids)
        assert precompute_weak_labels(cfg, real_catalog, uids) == 0


class TestCacheDiagnostics:
    def test_counts_reflect_what_was_written(self, cfg: Config, real_catalog: StudyCatalog) -> None:
        uids = real_catalog.studies_with_series()[:3]
        precompute_weak_labels(cfg, real_catalog, uids)
        summary = describe_cache(cfg, uids)
        assert summary["studies"] == 3
        assert summary["weak_labels"] == 3
        assert summary["guidance"] == 0  # chưa chạy bước guidance
