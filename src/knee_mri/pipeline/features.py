"""Precompute đặc trưng teacher: 3DINO-ViT (ảnh) và guidance VLM (text).

Cả hai teacher đều đóng băng, nên đặc trưng của chúng không đổi giữa các epoch.
Tính sẵn một lần rồi cache giúp vòng huấn luyện không phải nạp hai model khổng lồ
vào VRAM cùng lúc với student.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from knee_mri.config.schema import Config
from knee_mri.data.catalog import StudyCatalog
from knee_mri.data.dicom_io import EmptySeriesError
from knee_mri.data.series_selection import representative_series, select_series
from knee_mri.data.volume import build_volume
from knee_mri.models.teachers import Dino3DTeacher, GemmaGuidanceEncoder
from knee_mri.utils.io import save_npy
from knee_mri.utils.logging import get_logger
from knee_mri.utils.mask import load_mask

logger = get_logger(__name__)


def precompute_teacher_features(
    cfg: Config,
    catalog: StudyCatalog,
    study_uids: Sequence[str],
    *,
    device: str = "cuda",
    overwrite: bool = False,
) -> int:
    """Trích và cache đặc trưng 3DINO-ViT cho từng series."""
    cfg.paths.ensure()
    teacher = Dino3DTeacher(cfg, device=device)
    written = 0

    for index, study_uid in enumerate(study_uids):
        for series in select_series(catalog, study_uid, cfg.data):
            destination = cfg.paths.teacher_feats / study_uid / f"{series.series_uid}.npy"
            if destination.is_file() and not overwrite:
                continue
            series_dir = cfg.paths.series_dir(study_uid, series.series_uid)
            if series_dir is None:
                continue
            try:
                volume = build_volume(series_dir, cfg.data)
            except (EmptySeriesError, ValueError) as exc:
                logger.warning("Bỏ qua series %s: %s", series.series_uid[:16], exc)
                continue
            mask = load_mask(study_uid, series.series_uid, cfg.paths.masks)
            save_npy(teacher.extract(volume, mask), destination)
            written += 1

        if index % 25 == 0:
            logger.info("Teacher: %d/%d study, %d đặc trưng", index + 1, len(study_uids), written)

    logger.info("Teacher hoàn tất: %d đặc trưng tại %s", written, cfg.paths.teacher_feats)
    return written


def precompute_guidance(
    cfg: Config,
    catalog: StudyCatalog,
    study_uids: Sequence[str],
    *,
    device: str = "cuda",
    overwrite: bool = False,
    allow_zero_fallback: bool = False,
) -> int:
    """Trích và cache vector guidance (ảnh + report → G) cho từng study.

    Args:
        allow_zero_fallback: Ghi vector 0 khi VLM lỗi. Chỉ bật cho smoke test —
            guidance toàn 0 khiến loss contrastive vô nghĩa.
    """
    cfg.paths.ensure()
    encoder = GemmaGuidanceEncoder(cfg, device=device)
    written = 0
    failures = 0

    for index, study_uid in enumerate(study_uids):
        destination = cfg.paths.guidance / f"{study_uid}.npy"
        if destination.is_file() and not overwrite:
            continue

        series = representative_series(catalog, study_uid, cfg.data)
        if series is None:
            logger.debug("Study %s không có series, bỏ qua guidance.", study_uid[:16])
            continue
        series_dir = cfg.paths.series_dir(study_uid, series.series_uid)
        if series_dir is None:
            continue

        try:
            volume = build_volume(series_dir, cfg.data)
            vector = encoder.encode(volume, catalog.report(study_uid))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if not allow_zero_fallback:
                raise
            if failures <= 3:
                logger.warning("Guidance lỗi trên %s (%s), ghi vector 0.", study_uid[:16], exc)
            vector = np.zeros(cfg.model.guidance_dim, dtype=np.float32)

        save_npy(vector, destination)
        written += 1

        if index % 50 == 0:
            logger.info("Guidance: %d/%d study", index + 1, len(study_uids))

    if failures:
        logger.warning("Guidance thất bại trên %d study (đã ghi vector 0).", failures)
    logger.info("Guidance hoàn tất: %d vector tại %s", written, cfg.paths.guidance)
    return written


__all__ = ["precompute_guidance", "precompute_teacher_features"]
