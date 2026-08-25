"""Giải thích dự đoán: CAM từ attention và đối chiếu với mask ROI."""

from knee_mri.explain.cam import (
    NoAttentionError,
    attention_rollout,
    attention_to_volume,
    save_cam,
)
from knee_mri.explain.overlay import OverlapReport, binarise_cam, cam_mask_overlap

__all__ = [
    "NoAttentionError",
    "OverlapReport",
    "attention_rollout",
    "attention_to_volume",
    "binarise_cam",
    "cam_mask_overlap",
    "save_cam",
]
