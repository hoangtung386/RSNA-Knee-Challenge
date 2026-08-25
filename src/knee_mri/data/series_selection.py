"""Chọn series đại diện cho một study.

Một study MRI gối thường có nhiều chuỗi xung (T1, T2, PD, STIR...) ở nhiều mặt
phẳng. Không phải chuỗi nào cũng hữu ích như nhau cho 12 bệnh lý mục tiêu, nên
ta xếp hạng theo metadata rồi lấy vài chuỗi tốt nhất.

Hàm ở đây là **hàm thuần**: nhận catalog đã nạp sẵn, không đọc file (đối lập với
bản cũ nhận đường dẫn CSV rồi ``read_csv`` mỗi lần gọi — lỗi P1-1).
"""

from __future__ import annotations

from knee_mri.config.schema import DataCfg
from knee_mri.data.catalog import SeriesInfo, StudyCatalog

#: Trọng số xếp hạng. Mặt phẳng quan trọng nhất (sagittal cho ACL và sụn chêm),
#: kế đến là chuỗi nhạy dịch (thấy tràn dịch, phù tủy xương), rồi xóa mỡ.
_PLANE_WEIGHT: int = 3
_FLUID_WEIGHT: int = 2
_FATSUPP_WEIGHT: int = 1


def score_series(series: SeriesInfo, cfg: DataCfg) -> int:
    """Điểm ưu tiên của một series; càng cao càng đáng dùng."""
    return (
        _PLANE_WEIGHT * int(series.plane == cfg.prefer_plane)
        + _FLUID_WEIGHT * int(series.fluid_sensitive == cfg.prefer_fluid)
        + _FATSUPP_WEIGHT * int(series.fat_suppression == cfg.prefer_fatsupp)
    )


def select_series(
    catalog: StudyCatalog,
    study_uid: str,
    cfg: DataCfg,
    *,
    limit: int | None = None,
) -> list[SeriesInfo]:
    """Trả về các series tốt nhất của một study, đã xếp hạng giảm dần.

    Thứ tự ổn định: khi điểm bằng nhau thì giữ nguyên thứ tự xuất hiện trong CSV,
    nhờ ``sorted`` là thuật toán ổn định. Điều này khiến kết quả tái lập được.

    Args:
        catalog: Catalog đã nạp.
        study_uid: Study cần lấy series.
        cfg: Phần ``data`` của cấu hình (quyết định tiêu chí ưu tiên).
        limit: Số series tối đa; mặc định lấy ``cfg.max_series``.
    """
    available = catalog.series(study_uid)
    if not available:
        return []
    ranked = sorted(available, key=lambda item: score_series(item, cfg), reverse=True)
    return ranked[: limit if limit is not None else cfg.max_series]


def representative_series(catalog: StudyCatalog, study_uid: str, cfg: DataCfg) -> SeriesInfo | None:
    """Series đại diện duy nhất của study — dùng cho guidance và sinh report."""
    chosen = select_series(catalog, study_uid, cfg, limit=1)
    return chosen[0] if chosen else None


__all__ = ["representative_series", "score_series", "select_series"]
