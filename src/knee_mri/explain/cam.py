"""Class Activation Mapping cho student ViT-3D bằng attention rollout.

Attention rollout (Abnar & Zuidema, 2020) nhân dồn ma trận attention qua các
tầng, có tính đến residual connection, để lần ngược xem token CLS thực sự lấy
thông tin từ những patch nào. Ưu điểm so với Grad-CAM: không cần backward pass.

Chỉ hoạt động được nhờ :class:`~knee_mri.models.student.blocks.AttentionBlock`
thực sự trả về trọng số attention — bản cũ dựa vào forward hook nên danh sách
luôn rỗng và toàn bộ tính năng này là code chết (lỗi P0-3).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from knee_mri.utils.io import save_npy


class NoAttentionError(RuntimeError):
    """Không có ma trận attention nào — quên truyền ``return_attn=True``."""


def attention_rollout(attentions: Sequence[Tensor], *, keep_cls: bool = False) -> Tensor:
    """Nhân dồn attention qua các tầng để ra bản đồ quan trọng của CLS.

    Args:
        attentions: Danh sách tensor ``(B, heads, N, N)``, theo thứ tự tầng.
        keep_cls: Giữ lại cột của chính token CLS trong kết quả.

    Returns:
        Tensor ``(B, N-1)`` — mức quan trọng của từng patch, đã chuẩn hóa tổng 1.

    Raises:
        NoAttentionError: khi danh sách rỗng.
    """
    if not attentions:
        raise NoAttentionError("Danh sách attention rỗng. Hãy gọi student với return_attn=True.")

    num_tokens = attentions[0].shape[-1]
    device = attentions[0].device
    identity = torch.eye(num_tokens, device=device).unsqueeze(0)
    rollout = identity.clone()

    for attention in attentions:
        averaged = attention.mean(dim=1)  # trung bình các head -> (B, N, N)
        # Cộng identity để mô hình hóa residual connection, rồi tái chuẩn hóa.
        augmented = averaged + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        rollout = augmented @ rollout

    cls_attention = rollout[:, 0] if keep_cls else rollout[:, 0, 1:]
    return cls_attention / cls_attention.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def attention_to_volume(
    patch_attention: Tensor,
    grid: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Trải bản đồ theo patch trở lại độ phân giải voxel.

    Args:
        patch_attention: Tensor ``(N,)`` hoặc ``(B, N)`` với ``N = prod(grid)``.
        grid: Số patch theo từng chiều ``(nd, nh, nw)``.
        volume_shape: Kích thước volume đích ``(D, H, W)``.

    Returns:
        Mảng float32 ``(D, H, W)``, đã chuẩn hóa về ``[0, 1]``.
    """
    attention = patch_attention.detach().float()
    if attention.ndim == 2:
        attention = attention[0]

    expected = grid[0] * grid[1] * grid[2]
    if attention.numel() != expected:
        raise ValueError(
            f"Số patch không khớp: attention có {attention.numel()} phần tử, "
            f"grid {grid} cần {expected}."
        )

    volume = attention.reshape(1, 1, *grid)
    volume = F.interpolate(volume, size=volume_shape, mode="trilinear", align_corners=False)
    array = volume[0, 0].cpu().numpy().astype(np.float32)

    lowest, highest = float(array.min()), float(array.max())
    if highest > lowest:
        array = (array - lowest) / (highest - lowest)
    return array


def save_cam(cam: np.ndarray, study_uid: str, series_uid: str, cam_dir: Path) -> Path:
    """Ghi bản đồ CAM ra ``<cam_dir>/<study>/<series>.npy``."""
    return save_npy(cam.astype(np.float32), cam_dir / study_uid / f"{series_uid}.npy")


__all__ = ["NoAttentionError", "attention_rollout", "attention_to_volume", "save_cam"]
