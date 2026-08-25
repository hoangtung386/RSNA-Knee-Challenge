"""Sinh dự đoán cho tập test và ghi ``submission.csv``.

Lúc submit chỉ có **ảnh + metadata series** — không có report, không có teacher.
Student đã hấp thụ tri thức từ report qua KD nên tự dự đoán được.

Logic này trước đây chỉ tồn tại trong một cell notebook, không import lại được
và không test được. Nay là module có API rõ ràng, notebook chỉ gọi một dòng.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from knee_mri.config.schema import Config
from knee_mri.constants import COL_STUDY, LABELS, NUM_LABELS
from knee_mri.data.catalog import StudyCatalog
from knee_mri.data.dicom_io import EmptySeriesError
from knee_mri.data.series_selection import select_series
from knee_mri.data.volume import build_volume
from knee_mri.models.heads import ClassifierHead, make_metadata_vector
from knee_mri.models.student import ViT3DStudent
from knee_mri.training.checkpoint import BEST_NAME, load_for_inference
from knee_mri.utils.logging import get_logger
from knee_mri.utils.mask import load_mask

logger = get_logger(__name__)

#: Xác suất mặc định khi một study hoàn toàn không đọc được series nào — trùng
#: với giá trị trung tính của ``sample_submission.csv``.
NEUTRAL_PROBABILITY: float = 0.5


@dataclass(slots=True)
class Predictor:
    """Chạy student + classifier trên volume để ra xác suất mức study."""

    cfg: Config
    student: ViT3DStudent
    classifier: ClassifierHead
    device: str

    @classmethod
    def from_checkpoint(
        cls,
        cfg: Config,
        *,
        checkpoint: Path | None = None,
        device: str | None = None,
    ) -> Predictor:
        """Dựng predictor từ checkpoint đã lưu."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = checkpoint or (cfg.paths.checkpoints / BEST_NAME)
        student = ViT3DStudent(cfg)
        classifier = ClassifierHead(cfg)
        load_for_inference(path, {"student": student, "classifier": classifier}, device=device)
        logger.info("Predictor sẵn sàng trên %s từ %s", device, path.name)
        return cls(cfg=cfg, student=student, classifier=classifier, device=device)

    @torch.no_grad()
    def predict_study(self, catalog: StudyCatalog, study_uid: str) -> np.ndarray:
        """Xác suất 12 nhãn cho một study, trung bình trên các series đã chọn."""
        volumes: list[np.ndarray] = []
        metadata: list[list[float]] = []

        for series in select_series(catalog, study_uid, self.cfg.data):
            series_dir = self.cfg.paths.series_dir(study_uid, series.series_uid)
            if series_dir is None:
                continue
            mask = load_mask(study_uid, series.series_uid, self.cfg.paths.masks)
            try:
                volumes.append(build_volume(series_dir, self.cfg.data, mask=mask))
            except (EmptySeriesError, ValueError) as exc:
                logger.warning("Bỏ qua series %s: %s", series.series_uid[:16], exc)
                continue
            metadata.append(make_metadata_vector(series))

        if not volumes:
            logger.warning(
                "Study %s không có series đọc được → xác suất trung tính.", study_uid[:16]
            )
            return np.full(NUM_LABELS, NEUTRAL_PROBABILITY, dtype=np.float32)

        batch = torch.from_numpy(np.stack(volumes)).unsqueeze(1).to(self.device)
        meta_tensor = torch.tensor(metadata, dtype=torch.float32, device=self.device)
        features, _ = self.student(batch)
        logits = self.classifier(features, meta_tensor)
        return torch.sigmoid(logits.float()).mean(dim=0).cpu().numpy().astype(np.float32)

    def predict_many(
        self, catalog: StudyCatalog, study_uids: Sequence[str], *, log_every: int = 25
    ) -> np.ndarray:
        """Xác suất cho nhiều study, trả về mảng ``(n_study, 12)``."""
        self.student.eval()
        self.classifier.eval()
        rows = []
        for index, study_uid in enumerate(study_uids):
            rows.append(self.predict_study(catalog, study_uid))
            if log_every and index % log_every == 0:
                logger.info("Đã dự đoán %d/%d study", index + 1, len(study_uids))
        return np.stack(rows) if rows else np.zeros((0, NUM_LABELS), dtype=np.float32)


def build_submission(study_uids: Sequence[str], probabilities: np.ndarray) -> pd.DataFrame:
    """Dựng DataFrame đúng định dạng ``sample_submission.csv``."""
    frame = pd.DataFrame(probabilities, columns=list(LABELS))
    frame.insert(0, COL_STUDY, list(study_uids))
    return frame


def predict_submission(
    cfg: Config,
    *,
    checkpoint: Path | None = None,
    output: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> Path:
    """Chạy toàn bộ tập test và ghi ``submission.csv``.

    Returns:
        Đường dẫn file submission đã ghi.
    """
    catalog = StudyCatalog.from_csv(cfg.paths.test_csv, cfg.paths.test_series_csv)
    study_uids = catalog.studies()
    if limit is not None:
        study_uids = study_uids[:limit]

    predictor = Predictor.from_checkpoint(cfg, checkpoint=checkpoint, device=device)
    probabilities = predictor.predict_many(catalog, study_uids)

    destination = output or (cfg.paths.predictions / "submission.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_submission(study_uids, probabilities).to_csv(destination, index=False)
    logger.info("Đã ghi submission cho %d study: %s", len(study_uids), destination)
    return destination


__all__ = [
    "NEUTRAL_PROBABILITY",
    "Predictor",
    "build_submission",
    "predict_submission",
]
