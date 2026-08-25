"""Metric đánh giá — AUC macro, đúng thứ được cuộc thi chấm."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from knee_mri.constants import LABELS


@dataclass(frozen=True, slots=True)
class AucReport:
    """Kết quả đánh giá AUC."""

    macro: float
    per_label: dict[str, float] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AucReport(macro={self.macro:.4f}, đánh_giá_được={len(self.per_label)}/{len(LABELS)})"
        )


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC nhị phân cho một nhãn, tính bằng thống kê Mann–Whitney U.

    Tự cài đặt thay vì phụ thuộc scikit-learn để metric luôn chạy được kể cả
    trong môi trường tối giản. Xử lý điểm số bằng nhau bằng trung bình hạng,
    giống hệt ``sklearn.metrics.roc_auc_score``.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    positives = y_true > 0.5
    n_pos = int(positives.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    sorted_scores = y_score[order]
    index = 0
    while index < sorted_scores.size:
        stop = index
        while stop + 1 < sorted_scores.size and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        ranks[order[index : stop + 1]] = average_rank
        index = stop + 1

    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    labels: Sequence[str] = LABELS,
) -> AucReport:
    """AUC trung bình trên các nhãn đánh giá được.

    Nhãn chỉ có một lớp trong tập đánh giá thì AUC không xác định — nhãn đó được
    ghi vào ``skipped`` thay vì bị gán 0.5 một cách tùy tiện, để báo cáo phản ánh
    đúng những gì thực sự đo được.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape lệch nhau: y_true {y_true.shape} vs y_score {y_score.shape}")

    per_label: dict[str, float] = {}
    skipped: list[str] = []
    for index, label in enumerate(labels):
        value = roc_auc(y_true[:, index], y_score[:, index])
        if np.isnan(value):
            skipped.append(label)
        else:
            per_label[label] = value

    macro = float(np.mean(list(per_label.values()))) if per_label else float("nan")
    return AucReport(macro=macro, per_label=per_label, skipped=skipped)


def format_auc_report(report: AucReport) -> str:
    """Định dạng báo cáo AUC thành nhiều dòng để ghi log."""
    lines = [f"macro AUC = {report.macro:.4f} (trên {len(report.per_label)} nhãn)"]
    for label, value in sorted(report.per_label.items(), key=lambda item: -item[1]):
        lines.append(f"    {label:<18s} {value:.4f}")
    if report.skipped:
        lines.append(f"    (bỏ qua vì chỉ có một lớp: {', '.join(report.skipped)})")
    return "\n".join(lines)


__all__ = ["AucReport", "format_auc_report", "macro_auc", "roc_auc"]
