"""Index toàn bộ CSV metadata **một lần** vào bộ nhớ.

Bản cũ gọi ``pd.read_csv`` bên trong ``select_series`` và ``load_study_report``,
mà cả hai lại được gọi từ ``Dataset.__getitem__``. Kết quả: parse lại 3.4 MB
``train_series.csv`` và 5.7 MB ``train.csv`` cho **mỗi sample, mỗi epoch** —
hơn 130.000 lần đọc CSV cho một lần huấn luyện 30 epoch (lỗi P1-1 và P1-2).

Ở đây mọi thứ được index sẵn thành dict, tra cứu O(1).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from knee_mri.constants import (
    COL_FAT,
    COL_FLUID,
    COL_PLANE,
    COL_REPORT,
    COL_SERIES,
    COL_STUDY,
    LABELS,
)
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SeriesInfo:
    """Metadata của một series — có sẵn ở **cả** train lẫn test.

    Vì ``test_series.csv`` cũng chứa ba trường này, chúng dùng được làm đầu vào
    phụ trợ lúc inference, khác với report (chỉ có lúc train).
    """

    series_uid: str
    plane: str
    fluid_sensitive: int
    fat_suppression: int


def _coerce_flag(value: object) -> int:
    """Đổi ô CSV thành cờ 0/1, giá trị thiếu hoặc lạ đều thành 0."""
    try:
        return 1 if int(float(value)) == 1 else 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _coerce_label(value: object) -> int | None:
    """Đổi ô nhãn thành 0/1, trả ``None`` khi ô trống (nhãn chưa được gán)."""
    text = str(value).strip()
    if text in {"1", "1.0"}:
        return 1
    if text in {"0", "0.0"}:
        return 0
    return None


class StudyCatalog:
    """Tra cứu series, report và nhãn gold theo ``StudyInstanceUID``.

    Tạo bằng :meth:`from_csv` rồi truyền xuống Dataset; không bao giờ để lớp
    Dataset tự chạm vào file CSV.
    """

    __slots__ = ("_gold", "_reports", "_series", "_study_order")

    def __init__(
        self,
        series_by_study: Mapping[str, Sequence[SeriesInfo]],
        reports: Mapping[str, str],
        gold_labels: Mapping[str, dict[str, int]],
        study_order: Sequence[str],
    ) -> None:
        self._series = dict(series_by_study)
        self._reports = dict(reports)
        self._gold = dict(gold_labels)
        self._study_order = list(study_order)

    # ------------------------------------------------------------------ dựng
    @classmethod
    def from_csv(
        cls,
        study_csv: Path,
        series_csv: Path,
        *,
        labels: Sequence[str] = LABELS,
    ) -> StudyCatalog:
        """Nạp catalog từ cặp CSV study + series.

        Hoạt động với cả bộ train (có cột ``Report`` và 12 cột nhãn) lẫn bộ test
        (chỉ có cột ``StudyInstanceUID``).
        """
        series_df = pd.read_csv(series_csv)
        series_by_study: dict[str, list[SeriesInfo]] = {}
        for study_uid, group in series_df.groupby(COL_STUDY, sort=False):
            series_by_study[str(study_uid)] = [
                SeriesInfo(
                    series_uid=str(row[COL_SERIES]),
                    plane=str(row[COL_PLANE]),
                    fluid_sensitive=_coerce_flag(row[COL_FLUID]),
                    fat_suppression=_coerce_flag(row[COL_FAT]),
                )
                for _, row in group.iterrows()
            ]

        study_df = pd.read_csv(study_csv)
        study_order = [str(uid) for uid in study_df[COL_STUDY]]

        reports: dict[str, str] = {}
        if COL_REPORT in study_df.columns:
            reports = {
                str(uid): str(report)
                for uid, report in zip(
                    study_df[COL_STUDY], study_df[COL_REPORT].fillna(""), strict=True
                )
            }

        gold: dict[str, dict[str, int]] = {}
        present = [col for col in labels if col in study_df.columns]
        if present:
            for _, row in study_df.iterrows():
                values = {col: _coerce_label(row[col]) for col in present}
                if any(value is not None for value in values.values()):
                    gold[str(row[COL_STUDY])] = {
                        col: (value or 0) for col, value in values.items() if value is not None
                    }

        logger.info(
            "Catalog: %d study, %d series, %d report, %d study có nhãn gold.",
            len(study_order),
            len(series_df),
            len(reports),
            len(gold),
        )
        return cls(series_by_study, reports, gold, study_order)

    # ----------------------------------------------------------------- tra cứu
    def series(self, study_uid: str) -> list[SeriesInfo]:
        """Danh sách series của một study (rỗng nếu không có)."""
        return list(self._series.get(study_uid, ()))

    def report(self, study_uid: str) -> str:
        """Radiology report của study, chuỗi rỗng nếu không có."""
        return self._reports.get(study_uid, "")

    def gold(self, study_uid: str) -> dict[str, int] | None:
        """Nhãn cấu trúc do người gán, ``None`` nếu study này chưa được gán."""
        labels = self._gold.get(study_uid)
        return dict(labels) if labels is not None else None

    def studies(self) -> list[str]:
        """Toàn bộ study theo đúng thứ tự trong CSV."""
        return list(self._study_order)

    def studies_with_series(self) -> list[str]:
        """Chỉ những study thực sự có ít nhất một series trong CSV."""
        return [uid for uid in self._study_order if self._series.get(uid)]

    def studies_with_gold(self) -> list[str]:
        """Study có nhãn cấu trúc — tập validation quý nhất của dataset."""
        return [uid for uid in self._study_order if uid in self._gold]

    def gold_matrix(self, study_uids: Iterable[str]) -> tuple[list[str], list[list[int]]]:
        """Ma trận nhãn gold cho các study cho trước.

        Returns:
            ``(uids_có_nhãn, ma_trận)`` với mỗi hàng là 12 giá trị 0/1 theo đúng
            thứ tự :data:`~knee_mri.constants.LABELS`.
        """
        kept: list[str] = []
        rows: list[list[int]] = []
        for uid in study_uids:
            labels = self._gold.get(uid)
            if labels is None:
                continue
            kept.append(uid)
            rows.append([labels.get(name, 0) for name in LABELS])
        return kept, rows

    # ---------------------------------------------------------------- dunder
    def __len__(self) -> int:
        return len(self._study_order)

    def __contains__(self, study_uid: object) -> bool:
        return study_uid in self._series

    def __iter__(self) -> Iterator[str]:
        return iter(self._study_order)

    def __repr__(self) -> str:
        return (
            f"StudyCatalog(studies={len(self._study_order)}, "
            f"reports={len(self._reports)}, gold={len(self._gold)})"
        )


__all__ = ["SeriesInfo", "StudyCatalog"]
