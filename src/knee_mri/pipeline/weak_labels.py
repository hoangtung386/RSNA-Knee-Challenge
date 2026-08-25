"""Precompute nhãn yếu từ report — chạy **một lần**, offline.

Tách hẳn khỏi ``Dataset.__getitem__`` là điểm sửa quan trọng: bản cũ khởi tạo
một VLM cho mỗi sample của DataLoader, nên với ``num_workers > 0`` mỗi worker nạp
một bản model vào VRAM và OOM ngay (lỗi P1-3).
"""

from __future__ import annotations

from collections.abc import Sequence

from knee_mri.config.schema import Config
from knee_mri.data.catalog import StudyCatalog
from knee_mri.labeling.llm_based import GemmaReportParser
from knee_mri.labeling.rule_based import keyword_labeler
from knee_mri.utils.io import save_json
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)


def precompute_weak_labels(
    cfg: Config,
    catalog: StudyCatalog,
    study_uids: Sequence[str],
    *,
    use_vlm: bool = False,
    device: str = "cuda",
    overwrite: bool = False,
) -> int:
    """Sinh nhãn yếu cho từng study và cache ra JSON.

    Args:
        use_vlm: Dùng VLM đọc hiểu report. Nếu ``False`` (hoặc VLM lỗi) thì dùng
            luật từ khóa đa ngôn ngữ — chậm hơn về chất lượng nhưng chạy trên CPU.

    Returns:
        Số file nhãn mới được ghi.
    """
    cfg.paths.ensure()
    parser = GemmaReportParser(cfg.vlm, device=device) if use_vlm else None
    written = 0
    vlm_failures = 0

    for index, study_uid in enumerate(study_uids):
        destination = cfg.paths.weak_labels / f"{study_uid}.json"
        if destination.is_file() and not overwrite:
            continue

        report = catalog.report(study_uid)
        scores = None
        if parser is not None:
            try:
                scores = parser.parse(report)
            except Exception as exc:  # noqa: BLE001 - lùi về luật từ khóa thay vì dừng job
                vlm_failures += 1
                if vlm_failures <= 3:
                    logger.warning("VLM lỗi trên %s (%s), dùng luật từ khóa.", study_uid[:16], exc)
        if scores is None:
            scores = keyword_labeler(report)

        save_json(scores, destination)
        written += 1

        if index % 200 == 0:
            logger.info("Nhãn yếu: %d/%d study", index + 1, len(study_uids))

    if vlm_failures:
        logger.warning("VLM thất bại trên %d study, đã dùng luật từ khóa thay thế.", vlm_failures)
    logger.info("Nhãn yếu hoàn tất: %d file tại %s", written, cfg.paths.weak_labels)
    return written


__all__ = ["precompute_weak_labels"]
