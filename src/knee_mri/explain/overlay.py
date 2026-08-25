"""Đối chiếu CAM với mask ROI để kiểm chứng student "nhìn" đúng chỗ.

Nếu vùng student chú ý trùng khớp tốt với ROI do SAM sinh ra ở bước S1, ta có
bằng chứng model học đặc trưng giải phẫu thật chứ không bám vào artifact chụp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OverlapReport:
    """Mức trùng khớp giữa CAM và mask ROI."""

    iou: float
    dice: float
    cam_coverage: float
    threshold: float

    def __repr__(self) -> str:
        return f"OverlapReport(IoU={self.iou:.3f}, Dice={self.dice:.3f})"


def binarise_cam(cam: np.ndarray, *, percentile: float = 80.0) -> tuple[np.ndarray, float]:
    """Nhị phân hóa CAM tại một percentile.

    Dùng percentile thay vì giá trị trung bình (bản cũ dùng ``cam > cam.mean()``)
    vì phân bố attention rất lệch: trung bình bị kéo bởi phần đuôi dài nên cho ra
    vùng dương tính rộng bất thường.

    Returns:
        ``(mask_nhị_phân, ngưỡng_đã_dùng)``.
    """
    threshold = float(np.percentile(cam, percentile))
    return (cam > threshold).astype(np.uint8), threshold


def cam_mask_overlap(
    cam: np.ndarray, mask: np.ndarray, *, percentile: float = 80.0
) -> OverlapReport | None:
    """Tính IoU/Dice giữa CAM đã nhị phân hóa và mask ROI.

    Trả về ``None`` nếu mask rỗng hoặc lệch shape — không có gì để so sánh.
    """
    if mask is None or mask.sum() == 0 or cam.shape != mask.shape:
        return None

    cam_binary, threshold = binarise_cam(cam, percentile=percentile)
    mask_binary = (mask > 0).astype(np.uint8)

    intersection = float((cam_binary & mask_binary).sum())
    union = float((cam_binary | mask_binary).sum())
    total = float(cam_binary.sum() + mask_binary.sum())

    return OverlapReport(
        iou=intersection / union if union > 0 else 0.0,
        dice=2.0 * intersection / total if total > 0 else 0.0,
        cam_coverage=float(cam_binary.mean()),
        threshold=threshold,
    )


__all__ = ["OverlapReport", "binarise_cam", "cam_mask_overlap"]
