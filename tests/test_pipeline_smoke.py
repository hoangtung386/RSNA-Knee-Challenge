"""Smoke test đầu-cuối: DICOM thật → volume → batch → train → CAM → submission.

Đây là test quan trọng nhất của suite. Bản cũ **không thể chạy qua nổi bước
import** (``METADATA_DIM`` không tồn tại trong config — lỗi P0-1), nên chưa từng
có bằng chứng nào cho thấy pipeline chạy được đầu-cuối.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config, load_config  # noqa: E402
from knee_mri.data.catalog import StudyCatalog  # noqa: E402
from knee_mri.data.collate import collate_studies, pool_per_study, split_per_study  # noqa: E402
from knee_mri.data.dataset import KneeDataset, describe_cache  # noqa: E402
from knee_mri.inference.predictor import build_submission  # noqa: E402
from knee_mri.training.checkpoint import (  # noqa: E402
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from knee_mri.training.trainer import build_bundle, build_optimizer, compute_losses  # noqa: E402


@pytest.fixture(scope="module")
def real_catalog() -> StudyCatalog:
    """Catalog từ dataset thật, giới hạn vài study cho nhanh."""
    cfg = load_config("test")
    if not cfg.paths.train_csv.is_file():
        pytest.skip("Không có dataset cục bộ.")
    return StudyCatalog.from_csv(cfg.paths.train_csv, cfg.paths.train_series_csv)


@pytest.fixture
def small_dataset(cfg: Config, real_catalog: StudyCatalog) -> KneeDataset:
    """Dataset 2 study, dùng teacher/guidance giả (chưa chạy precompute)."""
    uids = real_catalog.studies_with_series()[:2]
    return KneeDataset(uids, real_catalog, cfg, use_dummy_teacher=True, use_dummy_guidance=True)


class TestDatasetToBatch:
    def test_sample_has_expected_keys(self, small_dataset: KneeDataset) -> None:
        sample = small_dataset[0]
        assert set(sample) == {
            "study_uid",
            "volumes",
            "teacher_feats",
            "metadata",
            "guidance",
            "weak_label",
            "gold_label",
        }

    def test_all_volumes_share_target_shape(self, small_dataset: KneeDataset, cfg: Config) -> None:
        """Bất biến trung tâm — điều kiện để collate hoạt động."""
        for volume in small_dataset[0]["volumes"]:
            assert volume.shape == cfg.data.target_shape

    def test_guidance_dim_matches_config(self, small_dataset: KneeDataset, cfg: Config) -> None:
        assert small_dataset[0]["guidance"].shape == (cfg.model.guidance_dim,)

    def test_collate_produces_stacked_tensors(
        self, small_dataset: KneeDataset, cfg: Config
    ) -> None:
        """P0-6: bản cũ vỡ ở đây khi ROI crop cho ra H/W khác nhau."""
        batch = collate_studies([small_dataset[0], small_dataset[1]])
        total_series = sum(batch["counts"])
        assert batch["volumes"].shape == (total_series, 1, *cfg.data.target_shape)
        assert batch["teacher_feats"].shape == (total_series, cfg.model.teacher_dim)
        assert batch["guidance"].shape == (2, cfg.model.guidance_dim)

    def test_counts_allow_regrouping_per_study(self, small_dataset: KneeDataset) -> None:
        batch = collate_studies([small_dataset[0], small_dataset[1]])
        chunks = split_per_study(batch["volumes"], batch["counts"])
        assert [chunk.shape[0] for chunk in chunks] == batch["counts"]

    def test_pooling_reduces_to_study_level(self, small_dataset: KneeDataset) -> None:
        batch = collate_studies([small_dataset[0], small_dataset[1]])
        pooled = pool_per_study(batch["teacher_feats"], batch["counts"])
        assert pooled.shape[0] == 2


class TestTrainingStep:
    def test_forward_backward_runs(self, cfg: Config, small_dataset: KneeDataset) -> None:
        """Một bước huấn luyện hoàn chỉnh: forward → loss → backward → step."""
        bundle = build_bundle(cfg)
        optimizer = build_optimizer(bundle, cfg)
        batch = collate_studies([small_dataset[0], small_dataset[1]])

        loss, components = compute_losses(bundle, batch, cfg, "cpu")
        assert torch.isfinite(loss)
        assert set(components) == {"total", "kd", "contrast", "cls", "auc"}
        assert all(np.isfinite(value) for value in components.values())

        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in bundle.student.parameters() if p.grad is not None]
        assert grads, "student phải nhận được gradient"
        assert any(g.abs().sum() > 0 for g in grads)
        optimizer.step()

    def test_loss_components_are_all_wired(self, cfg: Config, small_dataset: KneeDataset) -> None:
        """Mỗi thành phần loss phải thực sự đóng góp, không lặng lẽ bằng 0."""
        bundle = build_bundle(cfg)
        batch = collate_studies([small_dataset[0], small_dataset[1]])
        _, components = compute_losses(bundle, batch, cfg, "cpu")
        assert components["kd"] > 0
        assert components["contrast"] > 0
        assert components["cls"] > 0


class TestCheckpointRoundTrip:
    def test_save_then_load_restores_weights(self, cfg: Config, tmp_path) -> None:
        """P1-8: nạp bằng ``weights_only=True`` để không thực thi pickle tùy ý."""
        bundle = build_bundle(cfg)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(CheckpointState(epoch=3, best_metric=0.71, modules=bundle.as_dict()), path)

        restored = build_bundle(cfg)
        before = restored.student.pos_embed.clone()
        epoch, best = load_checkpoint(path, restored.as_dict())

        assert (epoch, pytest.approx(best)) == (3, 0.71)
        assert torch.allclose(restored.student.pos_embed, bundle.student.pos_embed)
        assert not torch.allclose(restored.student.pos_embed, before) or torch.allclose(
            bundle.student.pos_embed, before
        )

    def test_missing_file_raises(self, cfg: Config, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "khong-co.pt", build_bundle(cfg).as_dict())


class TestSubmissionFormat:
    def test_columns_match_sample_submission(self, cfg: Config) -> None:
        import pandas as pd

        from knee_mri.constants import NUM_LABELS

        sample_path = cfg.paths.data_root / "sample_submission.csv"
        if not sample_path.is_file():
            pytest.skip("Không có sample_submission.csv.")

        expected = list(pd.read_csv(sample_path).columns)
        frame = build_submission(["a", "b"], np.full((2, NUM_LABELS), 0.5, dtype=np.float32))
        assert list(frame.columns) == expected
        assert len(frame) == 2


class TestCacheDiagnostics:
    def test_describe_cache_counts_every_artifact(
        self, cfg: Config, real_catalog: StudyCatalog
    ) -> None:
        summary = describe_cache(cfg, real_catalog.studies_with_series()[:5])
        assert summary["studies"] == 5
        assert set(summary) >= {"masks", "teacher_feats", "guidance", "weak_labels"}
