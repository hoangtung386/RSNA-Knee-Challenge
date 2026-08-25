"""Test sinh submission — định dạng phải khớp ``sample_submission.csv`` tuyệt đối."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config  # noqa: E402
from knee_mri.constants import COL_STUDY, LABELS, NUM_LABELS  # noqa: E402
from knee_mri.data.catalog import StudyCatalog  # noqa: E402
from knee_mri.inference.predictor import (  # noqa: E402
    NEUTRAL_PROBABILITY,
    Predictor,
    build_submission,
    predict_submission,
)
from knee_mri.training.checkpoint import BEST_NAME, CheckpointState, save_checkpoint  # noqa: E402
from knee_mri.training.trainer import build_bundle  # noqa: E402


@pytest.fixture
def trained_checkpoint(cfg: Config):
    """Checkpoint chưa huấn luyện nhưng hợp lệ về cấu trúc."""
    cfg.paths.ensure()
    bundle = build_bundle(cfg)
    path = cfg.paths.checkpoints / BEST_NAME
    save_checkpoint(CheckpointState(epoch=1, best_metric=0.5, modules=bundle.as_dict()), path)
    return path


class TestBuildSubmission:
    def test_columns_and_order(self) -> None:
        frame = build_submission(["a", "b"], np.zeros((2, NUM_LABELS), dtype=np.float32))
        assert list(frame.columns) == [COL_STUDY, *LABELS]

    def test_matches_sample_submission_exactly(self, cfg: Config) -> None:
        sample_path = cfg.paths.data_root / "sample_submission.csv"
        if not sample_path.is_file():
            pytest.skip("Không có sample_submission.csv.")
        expected = list(pd.read_csv(sample_path).columns)
        frame = build_submission(["x"], np.zeros((1, NUM_LABELS), dtype=np.float32))
        assert list(frame.columns) == expected

    def test_preserves_probability_values(self) -> None:
        probabilities = np.linspace(0, 1, NUM_LABELS, dtype=np.float32)[None]
        frame = build_submission(["x"], probabilities)
        assert frame.loc[0, LABELS[-1]] == pytest.approx(1.0)


class TestPredictor:
    def test_probabilities_are_in_range(
        self, cfg: Config, catalog: StudyCatalog, trained_checkpoint
    ) -> None:
        predictor = Predictor.from_checkpoint(cfg, checkpoint=trained_checkpoint, device="cpu")
        probabilities = predictor.predict_study(catalog, "study-a")
        assert probabilities.shape == (NUM_LABELS,)
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_unreadable_study_gets_neutral_probability(
        self, cfg: Config, catalog: StudyCatalog, trained_checkpoint
    ) -> None:
        """Study không đọc được series nào vẫn phải có hàng trong submission.

        Bỏ sót hàng khiến file submission bị từ chối; trả xác suất trung tính là
        cách suy giảm đúng.
        """
        predictor = Predictor.from_checkpoint(cfg, checkpoint=trained_checkpoint, device="cpu")
        probabilities = predictor.predict_study(catalog, "study-c")
        assert np.allclose(probabilities, NEUTRAL_PROBABILITY)

    def test_predict_many_returns_one_row_per_study(
        self, cfg: Config, catalog: StudyCatalog, trained_checkpoint
    ) -> None:
        predictor = Predictor.from_checkpoint(cfg, checkpoint=trained_checkpoint, device="cpu")
        scores = predictor.predict_many(catalog, catalog.studies(), log_every=0)
        assert scores.shape == (len(catalog.studies()), NUM_LABELS)

    def test_is_deterministic(self, cfg: Config, catalog: StudyCatalog, trained_checkpoint) -> None:
        """Cùng đầu vào phải cho cùng đầu ra — dropout phải tắt ở chế độ eval."""
        predictor = Predictor.from_checkpoint(cfg, checkpoint=trained_checkpoint, device="cpu")
        first = predictor.predict_many(catalog, ["study-a"], log_every=0)
        second = predictor.predict_many(catalog, ["study-a"], log_every=0)
        assert np.allclose(first, second)


class TestPredictSubmission:
    def test_writes_csv_for_every_test_study(self, cfg: Config, trained_checkpoint) -> None:
        if not cfg.paths.test_csv.is_file():
            pytest.skip("Không có test.csv.")
        destination = predict_submission(cfg, checkpoint=trained_checkpoint, device="cpu")
        frame = pd.read_csv(destination)
        expected = len(pd.read_csv(cfg.paths.test_csv))
        assert len(frame) == expected
        assert frame[list(LABELS)].to_numpy().min() >= 0.0
        assert frame[list(LABELS)].to_numpy().max() <= 1.0
