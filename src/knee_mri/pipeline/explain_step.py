"""Bước S3: sinh CAM, đối chiếu với mask ROI và viết báo cáo.

Đây là bước kiểm chứng chất lượng: nếu vùng student chú ý trùng với ROI giải
phẫu, ta có bằng chứng model học đặc trưng thật thay vì bám vào artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from knee_mri.config.schema import Config
from knee_mri.data.catalog import StudyCatalog
from knee_mri.data.dicom_io import EmptySeriesError
from knee_mri.data.normalize import resize_volume
from knee_mri.data.series_selection import representative_series
from knee_mri.data.volume import build_volume
from knee_mri.explain.cam import attention_rollout, attention_to_volume, save_cam
from knee_mri.explain.overlay import OverlapReport, cam_mask_overlap
from knee_mri.models.heads import ClassifierHead, make_metadata_vector
from knee_mri.models.report_generator import GemmaReportGenerator, rule_based_report
from knee_mri.models.student import ViT3DStudent
from knee_mri.training.checkpoint import BEST_NAME, load_for_inference
from knee_mri.utils.logging import get_logger
from knee_mri.utils.mask import load_mask

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    """Kết quả giải thích cho một study."""

    study_uid: str
    series_uid: str
    probabilities: np.ndarray
    overlap: OverlapReport | None
    report_path: Path


@torch.no_grad()
def explain_studies(
    cfg: Config,
    catalog: StudyCatalog,
    study_uids: Sequence[str],
    *,
    checkpoint: Path | None = None,
    device: str | None = None,
    use_vlm: bool = False,
) -> list[ExplanationResult]:
    """Sinh CAM + báo cáo cho từng study, lưu ra ``artifacts/``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.paths.ensure()

    student = ViT3DStudent(cfg)
    classifier = ClassifierHead(cfg)
    load_for_inference(
        checkpoint or (cfg.paths.checkpoints / BEST_NAME),
        {"student": student, "classifier": classifier},
        device=device,
    )
    generator = GemmaReportGenerator(cfg, device=device) if use_vlm else None
    results: list[ExplanationResult] = []

    for study_uid in study_uids:
        series = representative_series(catalog, study_uid, cfg.data)
        if series is None:
            continue
        series_dir = cfg.paths.series_dir(study_uid, series.series_uid)
        if series_dir is None:
            continue

        mask = load_mask(study_uid, series.series_uid, cfg.paths.masks)
        try:
            volume = build_volume(series_dir, cfg.data, mask=mask)
        except (EmptySeriesError, ValueError) as exc:
            logger.warning("Bỏ qua study %s: %s", study_uid[:16], exc)
            continue

        tensor = torch.from_numpy(volume)[None, None].to(device)
        metadata = torch.tensor([make_metadata_vector(series)], dtype=torch.float32, device=device)

        features, attentions = student(tensor, return_attn=True)
        logits = classifier(features, metadata)
        probabilities = torch.sigmoid(logits.float())[0].cpu().numpy()

        overlap: OverlapReport | None = None
        if attentions:
            patch_attention = attention_rollout(attentions)
            cam = attention_to_volume(patch_attention, student.grid, cfg.data.target_shape)
            save_cam(cam, study_uid, series.series_uid, cfg.paths.cams)
            # So CAM với mask đã resize về cùng shape mà student thực sự nhìn.
            if mask is not None:
                resized_mask = (
                    resize_volume(mask.astype(np.float32), cfg.data.target_shape) > 0.5
                ).astype(np.uint8)
                overlap = cam_mask_overlap(cam, resized_mask)
        else:  # pragma: no cover - chỉ xảy ra nếu student bị sửa sai
            logger.error("Student không trả về attention — không dựng được CAM.")

        text = (
            generator.generate(probabilities)
            if generator is not None
            else rule_based_report(probabilities)
        )
        report_path = cfg.paths.reports / f"{study_uid}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        overlap_line = (
            f"CAM/ROI IoU={overlap.iou:.3f} Dice={overlap.dice:.3f}"
            if overlap is not None
            else "CAM/ROI: không có mask để đối chiếu"
        )
        report_path.write_text(f"{text}\n\n{overlap_line}\n", encoding="utf-8")

        results.append(
            ExplanationResult(
                study_uid=study_uid,
                series_uid=series.series_uid,
                probabilities=probabilities,
                overlap=overlap,
                report_path=report_path,
            )
        )

    measured = [r.overlap.iou for r in results if r.overlap is not None]
    if measured:
        logger.info(
            "S3 hoàn tất: %d study | IoU CAM/ROI trung bình = %.3f",
            len(results),
            float(np.mean(measured)),
        )
    else:
        logger.info("S3 hoàn tất: %d study (chưa có mask để đối chiếu).", len(results))
    return results


__all__ = ["ExplanationResult", "explain_studies"]
