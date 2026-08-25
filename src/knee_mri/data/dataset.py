"""Torch ``Dataset`` cho huấn luyện — **chỉ đọc dữ liệu đã precompute**.

Nguyên tắc bất di bất dịch: ``__getitem__`` không được đọc CSV, không được nạp
model, không được gọi mạng. Bản cũ vi phạm cả ba (đọc lại hai file CSV lớn cho
mỗi sample và khởi tạo một VLM cho mỗi sample — lỗi P1-1, P1-2, P1-3).

Mọi thứ nặng đã được tính sẵn thành file trong ``artifacts/``:

* ``masks/<study>/<series>.npy``        — mask ROI từ SAM
* ``teacher_feats/<study>/<series>.npy`` — đặc trưng 3DINO-ViT
* ``guidance/<study>.npy``               — vector guidance từ VLM
* ``weak_labels/<study>.json``           — nhãn yếu từ report
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from torch.utils.data import Dataset

from knee_mri.config.schema import Config
from knee_mri.constants import LABELS, NUM_LABELS
from knee_mri.data.catalog import StudyCatalog
from knee_mri.data.dicom_io import EmptySeriesError
from knee_mri.data.series_selection import select_series
from knee_mri.data.volume import build_volume, empty_volume
from knee_mri.models.heads import make_metadata_vector
from knee_mri.utils.io import load_json, load_npy
from knee_mri.utils.logging import get_logger
from knee_mri.utils.mask import load_mask

logger = get_logger(__name__)


class StudySample(TypedDict):
    """Một study kèm toàn bộ series đã chọn của nó."""

    study_uid: str
    volumes: list[np.ndarray]
    teacher_feats: list[np.ndarray]
    metadata: list[list[float]]
    guidance: np.ndarray
    weak_label: np.ndarray
    gold_label: np.ndarray | None


class KneeDataset(Dataset[StudySample]):
    """Trả về một study mỗi lần lấy mẫu, gồm nhiều series của study đó.

    Args:
        study_uids: Danh sách study thuộc tập này (train hoặc val).
        catalog: Catalog đã nạp sẵn — Dataset không tự đọc CSV.
        cfg: Cấu hình đầy đủ.
        use_dummy_teacher: Sinh đặc trưng teacher ngẫu nhiên thay vì đọc file.
            Chỉ dùng cho smoke test khi chưa chạy precompute.
        use_dummy_guidance: Tương tự cho guidance.
    """

    def __init__(
        self,
        study_uids: Sequence[str],
        catalog: StudyCatalog,
        cfg: Config,
        *,
        use_dummy_teacher: bool = False,
        use_dummy_guidance: bool = False,
    ) -> None:
        self.study_uids = list(study_uids)
        self.catalog = catalog
        self.cfg = cfg
        self.use_dummy_teacher = use_dummy_teacher
        self.use_dummy_guidance = use_dummy_guidance
        self._rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return len(self.study_uids)

    # ------------------------------------------------------- nạp phần đã cache
    def _teacher_feature(self, study_uid: str, series_uid: str) -> np.ndarray:
        path = self.cfg.paths.teacher_feats / study_uid / f"{series_uid}.npy"
        if path.is_file():
            return load_npy(path)
        if self.use_dummy_teacher:
            return self._rng.standard_normal(self.cfg.model.teacher_dim, dtype=np.float32)
        return np.zeros(self.cfg.model.teacher_dim, dtype=np.float32)

    def _guidance(self, study_uid: str) -> np.ndarray:
        expected = self.cfg.model.guidance_dim
        path = self.cfg.paths.guidance / f"{study_uid}.npy"
        if path.is_file():
            vector = load_npy(path)
            if vector.shape[0] != expected:
                raise ValueError(
                    f"Guidance của {study_uid} có {vector.shape[0]} chiều nhưng cấu hình "
                    f"khai báo {expected}. Xóa cache guidance và precompute lại."
                )
            return vector
        if self.use_dummy_guidance:
            return self._rng.standard_normal(expected, dtype=np.float32)
        return np.zeros(expected, dtype=np.float32)

    def _weak_label(self, study_uid: str) -> np.ndarray:
        path = self.cfg.paths.weak_labels / f"{study_uid}.json"
        if not path.is_file():
            return np.zeros(NUM_LABELS, dtype=np.float32)
        scores = load_json(path)
        return np.array([float(scores.get(name, 0)) for name in LABELS], dtype=np.float32)

    def _gold_label(self, study_uid: str) -> np.ndarray | None:
        scores = self.catalog.gold(study_uid)
        if scores is None:
            return None
        return np.array([float(scores.get(name, 0)) for name in LABELS], dtype=np.float32)

    # ---------------------------------------------------------------- lấy mẫu
    def __getitem__(self, index: int) -> StudySample:
        study_uid = self.study_uids[index]
        volumes: list[np.ndarray] = []
        teacher_feats: list[np.ndarray] = []
        metadata: list[list[float]] = []

        for series in select_series(self.catalog, study_uid, self.cfg.data):
            series_dir = self.cfg.paths.series_dir(study_uid, series.series_uid)
            if series_dir is None:
                continue
            mask = load_mask(study_uid, series.series_uid, self.cfg.paths.masks)
            try:
                volume = build_volume(series_dir, self.cfg.data, mask=mask)
            except (EmptySeriesError, ValueError) as exc:
                logger.warning("Bỏ qua series %s: %s", series.series_uid[:16], exc)
                continue
            volumes.append(volume)
            teacher_feats.append(self._teacher_feature(study_uid, series.series_uid))
            metadata.append(make_metadata_vector(series))

        if not volumes:
            # Study không đọc được series nào: trả mẫu rỗng đúng shape để batch
            # vẫn ghép được, thay vì làm hỏng cả epoch.
            logger.warning("Study %s không có series đọc được, dùng volume rỗng.", study_uid[:16])
            volumes = [empty_volume(self.cfg.data)]
            teacher_feats = [np.zeros(self.cfg.model.teacher_dim, dtype=np.float32)]
            metadata = [[0.0] * self.cfg.model.metadata_dim]

        return StudySample(
            study_uid=study_uid,
            volumes=volumes,
            teacher_feats=teacher_feats,
            metadata=metadata,
            guidance=self._guidance(study_uid),
            weak_label=self._weak_label(study_uid),
            gold_label=self._gold_label(study_uid),
        )

    def __repr__(self) -> str:
        return (
            f"KneeDataset(studies={len(self.study_uids)}, "
            f"target_shape={self.cfg.data.target_shape})"
        )


def describe_cache(cfg: Config, study_uids: Sequence[str]) -> dict[str, Any]:
    """Đếm xem bao nhiêu study đã có sẵn từng loại cache — dùng để chẩn đoán."""
    return {
        "studies": len(study_uids),
        "masks": sum((cfg.paths.masks / uid).is_dir() for uid in study_uids),
        "teacher_feats": sum((cfg.paths.teacher_feats / uid).is_dir() for uid in study_uids),
        "guidance": sum((cfg.paths.guidance / f"{uid}.npy").is_file() for uid in study_uids),
        "weak_labels": sum((cfg.paths.weak_labels / f"{uid}.json").is_file() for uid in study_uids),
    }


__all__ = ["KneeDataset", "StudySample", "describe_cache"]
