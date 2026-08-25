"""Test lớp dữ liệu: catalog, chọn series, dựng volume, chia tập."""

from __future__ import annotations

import numpy as np

from knee_mri.config import Config
from knee_mri.data.catalog import SeriesInfo, StudyCatalog
from knee_mri.data.series_selection import score_series, select_series
from knee_mri.data.splits import make_split
from knee_mri.data.volume import build_volume, empty_volume, middle_slices


class TestCatalog:
    def test_lookup_returns_series(self, catalog: StudyCatalog) -> None:
        assert len(catalog.series("study-a")) == 3

    def test_missing_study_returns_empty(self, catalog: StudyCatalog) -> None:
        assert catalog.series("khong-ton-tai") == []

    def test_report_lookup(self, catalog: StudyCatalog) -> None:
        assert "medial meniscus" in catalog.report("study-a").lower()

    def test_gold_only_for_labelled_studies(self, catalog: StudyCatalog) -> None:
        assert catalog.gold("study-a") is not None
        assert catalog.gold("study-b") is None

    def test_studies_with_series_excludes_empty(self, catalog: StudyCatalog) -> None:
        assert catalog.studies_with_series() == ["study-a", "study-b"]

    def test_gold_matrix_follows_label_order(self, catalog: StudyCatalog) -> None:
        from knee_mri.constants import LABELS

        uids, rows = catalog.gold_matrix(catalog.studies())
        assert uids == ["study-a"]
        assert rows[0][LABELS.index("Effusion")] == 1
        assert rows[0][LABELS.index("ACL")] == 0

    def test_lookups_do_not_touch_disk(self, catalog: StudyCatalog) -> None:
        """P1-1/P1-2: bản cũ đọc lại CSV ở mỗi lần tra cứu.

        Catalog dựng thủ công không có file CSV nào phía sau, nên nếu code còn
        cố đọc file thì test này sẽ hỏng.
        """
        for _ in range(1000):
            catalog.series("study-a")
            catalog.report("study-a")


class TestSeriesSelection:
    def test_prefers_sagittal_fluid_fatsupp(self, catalog: StudyCatalog, cfg: Config) -> None:
        chosen = select_series(catalog, "study-a", cfg.data)
        assert chosen[0].series_uid == "series-a1"

    def test_respects_max_series(self, catalog: StudyCatalog, cfg: Config) -> None:
        assert len(select_series(catalog, "study-a", cfg.data)) == cfg.data.max_series

    def test_limit_argument_overrides_config(self, catalog: StudyCatalog, cfg: Config) -> None:
        assert len(select_series(catalog, "study-a", cfg.data, limit=1)) == 1

    def test_ordering_is_deterministic(self, catalog: StudyCatalog, cfg: Config) -> None:
        first = [s.series_uid for s in select_series(catalog, "study-a", cfg.data)]
        second = [s.series_uid for s in select_series(catalog, "study-a", cfg.data)]
        assert first == second

    def test_score_is_monotonic_in_matches(self, cfg: Config) -> None:
        perfect = SeriesInfo("x", cfg.data.prefer_plane, 1, 1)
        partial = SeriesInfo("y", cfg.data.prefer_plane, 0, 0)
        none = SeriesInfo("z", "Axial", 0, 0)
        assert score_series(perfect, cfg.data) > score_series(partial, cfg.data)
        assert score_series(partial, cfg.data) > score_series(none, cfg.data)


class TestVolumeShapeInvariant:
    """P0-6: mọi volume phải có cùng shape, kể cả sau khi cắt ROI.

    Bản cũ để ``roi_crop`` trả về shape tùy ý, khiến ``torch.cat`` trong collate
    vỡ vì H/W lệch nhau, và số patch của ViT-3D đổi theo từng batch.
    """

    def test_shape_matches_config(self, real_series_dir, cfg: Config) -> None:
        assert build_volume(real_series_dir, cfg.data).shape == cfg.data.target_shape

    def test_shape_unchanged_by_roi_crop(self, real_series_dir, cfg: Config) -> None:
        from knee_mri.data.dicom_io import read_series
        from knee_mri.utils.mask import center_box_mask

        raw = read_series(real_series_dir, max_slices=cfg.data.max_slices * 3)
        cropped = build_volume(real_series_dir, cfg.data, mask=center_box_mask(raw.shape, 0.3))
        assert cropped.shape == cfg.data.target_shape

    def test_mismatched_mask_is_ignored_safely(self, real_series_dir, cfg: Config) -> None:
        """Mask lệch shape thì bỏ qua, không được làm hỏng volume."""
        wrong = np.ones((3, 3, 3), dtype=np.uint8)
        assert build_volume(real_series_dir, cfg.data, mask=wrong).shape == cfg.data.target_shape

    def test_value_range_matches_norm_mode(self, real_series_dir, cfg: Config) -> None:
        volume = build_volume(real_series_dir, cfg.data)
        assert volume.dtype == np.float32
        assert volume.min() >= -1.0 and volume.max() <= 1.0

    def test_empty_volume_matches_target(self, cfg: Config) -> None:
        assert empty_volume(cfg.data).shape == cfg.data.target_shape


class TestMiddleSlices:
    def test_returns_requested_count(self, volume: np.ndarray) -> None:
        assert middle_slices(volume, 4).shape[0] == 4

    def test_clamps_to_available_depth(self, volume: np.ndarray) -> None:
        assert middle_slices(volume, 999).shape[0] == volume.shape[0]

    def test_at_least_one_slice(self, volume: np.ndarray) -> None:
        assert middle_slices(volume, 0).shape[0] == 1


class TestSplits:
    """P1-7: bản cũ không có validation, và 58 nhãn gold không được dùng."""

    def test_gold_studies_are_held_out(self, catalog: StudyCatalog) -> None:
        split = make_split(catalog)
        assert "study-a" in split.val
        assert "study-a" not in split.train

    def test_train_and_val_are_disjoint(self, catalog: StudyCatalog) -> None:
        split = make_split(catalog, extra_val_ratio=0.5)
        assert not set(split.train) & set(split.val)

    def test_limit_shrinks_the_pool(self, catalog: StudyCatalog) -> None:
        split = make_split(catalog, limit=1)
        assert len(split.train) + len(split.val) == 1

    def test_is_reproducible(self, catalog: StudyCatalog) -> None:
        first = make_split(catalog, seed=7, extra_val_ratio=0.5)
        second = make_split(catalog, seed=7, extra_val_ratio=0.5)
        assert first.train == second.train and first.val == second.val
