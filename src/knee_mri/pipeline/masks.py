"""Bước S1: sinh mask ROI bằng VLM + SAM."""

from __future__ import annotations

from collections.abc import Sequence

from knee_mri.config.schema import Config
from knee_mri.data.catalog import StudyCatalog
from knee_mri.data.dicom_io import EmptySeriesError
from knee_mri.data.series_selection import select_series
from knee_mri.data.volume import build_volume
from knee_mri.models.sam_masker import SamMaskGenerator, center_bbox
from knee_mri.utils.logging import get_logger
from knee_mri.utils.mask import mask_path, save_mask

logger = get_logger(__name__)


def generate_masks(
    cfg: Config,
    catalog: StudyCatalog,
    study_uids: Sequence[str],
    *,
    device: str = "cuda",
    overwrite: bool = False,
) -> int:
    """Sinh và lưu mask cho mọi series đã chọn của các study cho trước.

    Bỏ qua series đã có mask **trước khi** đọc DICOM — bản cũ dựng volume rồi mới
    kiểm tra file tồn tại, lãng phí toàn bộ IO cho các series đã xử lý (lỗi P3-6).

    Returns:
        Số mask mới được ghi.
    """
    cfg.paths.ensure()
    masker = SamMaskGenerator(cfg, device=device)
    written = 0

    for index, study_uid in enumerate(study_uids):
        for series in select_series(catalog, study_uid, cfg.data):
            destination = mask_path(cfg.paths.masks, study_uid, series.series_uid)
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

            _, height, width = volume.shape
            mask = masker.generate_3d(volume, bbox=center_bbox(height, width))
            save_mask(mask, study_uid, series.series_uid, cfg.paths.masks)
            written += 1

        if index % 25 == 0:
            logger.info("S1: %d/%d study, đã ghi %d mask", index + 1, len(study_uids), written)

    logger.info("S1 hoàn tất: %d mask mới tại %s", written, cfg.paths.masks)
    return written


__all__ = ["generate_masks"]
