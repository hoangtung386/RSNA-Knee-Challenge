"""Test metric AUC — đối chiếu với scikit-learn làm chuẩn tham chiếu.

AUC là metric chấm điểm của cuộc thi, nên cài đặt tự viết trong
:mod:`knee_mri.training.metrics` phải khớp scikit-learn tới từng chữ số. Ta tự
cài để metric luôn chạy được kể cả trong môi trường tối giản không có sklearn,
nhưng chính vì thế nó cần được kiểm chứng chặt.
"""

from __future__ import annotations

import numpy as np
import pytest

from knee_mri.constants import LABELS, NUM_LABELS
from knee_mri.training.metrics import AucReport, format_auc_report, macro_auc, roc_auc


class TestRocAuc:
    @pytest.mark.parametrize(
        "truth,score,expected",
        [
            ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
            ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0),
            ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5),
            ([0, 1], [0.3, 0.7], 1.0),
        ],
        ids=["hoàn hảo", "ngược hoàn toàn", "toàn bằng nhau", "tối giản"],
    )
    def test_known_values(self, truth, score, expected) -> None:
        assert roc_auc(np.array(truth), np.array(score)) == pytest.approx(expected)

    def test_ties_use_average_rank(self) -> None:
        """Điểm bằng nhau phải dùng hạng trung bình, giống sklearn."""
        assert roc_auc(np.array([0, 1, 1, 0]), np.array([0.5, 0.5, 0.9, 0.1])) == pytest.approx(
            0.875
        )

    @pytest.mark.parametrize("truth", [[1, 1, 1], [0, 0, 0]])
    def test_single_class_is_undefined(self, truth) -> None:
        """Chỉ một lớp thì AUC không xác định — trả NaN, không phải 0.5 tùy tiện."""
        assert np.isnan(roc_auc(np.array(truth), np.array([0.1, 0.5, 0.9])))

    def test_invariant_to_monotone_rescaling(self) -> None:
        """AUC chỉ phụ thuộc thứ hạng, không phụ thuộc thang đo điểm."""
        rng = np.random.default_rng(0)
        truth = (rng.random(200) < 0.3).astype(int)
        score = rng.random(200)
        assert roc_auc(truth, score) == pytest.approx(roc_auc(truth, score * 100 + 7))

    def test_matches_sklearn(self) -> None:
        """Chuẩn tham chiếu: phải trùng sklearn trên dữ liệu ngẫu nhiên."""
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(42)
        for _ in range(20):
            truth = (rng.random(150) < rng.uniform(0.05, 0.5)).astype(int)
            if truth.min() == truth.max():
                continue
            score = rng.random(150)
            assert roc_auc(truth, score) == pytest.approx(
                sklearn_metrics.roc_auc_score(truth, score), abs=1e-9
            )

    def test_matches_sklearn_with_heavy_ties(self) -> None:
        """Trường hợp dễ sai nhất: rất nhiều điểm trùng nhau."""
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(7)
        truth = (rng.random(300) < 0.2).astype(int)
        score = rng.integers(0, 4, size=300).astype(float)  # chỉ 4 giá trị riêng biệt
        assert roc_auc(truth, score) == pytest.approx(
            sklearn_metrics.roc_auc_score(truth, score), abs=1e-9
        )


class TestMacroAuc:
    def test_averages_over_evaluable_labels(self) -> None:
        truth = np.zeros((10, NUM_LABELS))
        score = np.zeros((10, NUM_LABELS))
        truth[:5, 0] = 1
        score[:5, 0] = 1.0  # nhãn 0: tách hoàn hảo
        truth[:5, 1] = 1
        score[5:, 1] = 1.0  # nhãn 1: ngược hoàn toàn

        report = macro_auc(truth, score)
        assert report.per_label[LABELS[0]] == pytest.approx(1.0)
        assert report.per_label[LABELS[1]] == pytest.approx(0.0)
        assert report.macro == pytest.approx(0.5)

    def test_single_class_labels_are_reported_as_skipped(self) -> None:
        """Nhãn không đo được phải vào ``skipped``, không bị gán 0.5 âm thầm.

        Với 12 nhãn cực hiếm, gán 0.5 cho nhãn không có mẫu dương sẽ kéo macro
        AUC về 0.5 và che mất chất lượng thật của các nhãn đo được.
        """
        truth = np.zeros((10, NUM_LABELS))
        truth[:5, 0] = 1
        report = macro_auc(truth, np.random.default_rng(0).random((10, NUM_LABELS)))
        assert len(report.per_label) == 1
        assert len(report.skipped) == NUM_LABELS - 1
        assert LABELS[0] not in report.skipped

    def test_all_labels_single_class_gives_nan(self) -> None:
        report = macro_auc(
            np.zeros((5, NUM_LABELS)), np.random.default_rng(0).random((5, NUM_LABELS))
        )
        assert np.isnan(report.macro)
        assert report.per_label == {}

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape lệch nhau"):
            macro_auc(np.zeros((5, NUM_LABELS)), np.zeros((4, NUM_LABELS)))

    def test_matches_sklearn_macro(self) -> None:
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(3)
        truth = (rng.random((120, NUM_LABELS)) < 0.35).astype(int)
        score = rng.random((120, NUM_LABELS))
        report = macro_auc(truth, score)
        assert report.macro == pytest.approx(
            sklearn_metrics.roc_auc_score(truth, score, average="macro"), abs=1e-9
        )


class TestFormatting:
    def test_lists_labels_sorted_by_score(self) -> None:
        report = AucReport(macro=0.75, per_label={"ACL": 0.6, "Effusion": 0.9}, skipped=["MCL"])
        text = format_auc_report(report)
        assert text.index("Effusion") < text.index("ACL")
        assert "0.7500" in text
        assert "MCL" in text

    def test_handles_empty_report(self) -> None:
        assert "macro AUC" in format_auc_report(AucReport(macro=float("nan")))
