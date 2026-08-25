"""Student ViT-3D — model **chỉ dùng ảnh**, là thứ duy nhất chạy lúc inference.

Nhận volume ``(B, 1, D, H, W)``, patchify bằng convolution 3D, cho qua chồng
:class:`~knee_mri.models.student.blocks.AttentionBlock`, xuất đặc trưng CLS.

Điểm sửa so với bản cũ: ``pos_embed`` được cấp phát **một lần trong ``__init__``**
với số patch suy ra từ cấu hình. Bản cũ tạo lại ``nn.Parameter`` ngay trong
``forward`` mỗi khi kích thước đầu vào đổi (lỗi P0-5), khiến tham số mới nằm
ngoài optimizer nên vĩnh viễn bằng 0, đồng thời làm ``state_dict`` đổi shape giữa
các lần lưu.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from knee_mri.config.schema import Config
from knee_mri.models.student.blocks import AttentionBlock, sinusoidal_position_embedding


def grid_size(
    volume_shape: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Số patch theo từng chiều khi patchify volume.

    Raises:
        ValueError: khi kích thước volume không chia hết cho patch. Chia hết là
            bắt buộc: ``nn.Conv3d`` với ``stride == kernel`` sẽ **âm thầm bỏ**
            phần dư ở rìa, làm lệch ánh xạ ngược từ patch về voxel khi dựng CAM.
    """
    for axis, (extent, patch) in enumerate(zip(volume_shape, patch_size, strict=True)):
        if extent % patch != 0:
            raise ValueError(
                f"Chiều {axis} của volume ({extent}) không chia hết cho patch ({patch}). "
                "Hãy chỉnh data.volume_size/data.max_slices hoặc model.patch_size."
            )
    depth, height, width = (
        extent // patch for extent, patch in zip(volume_shape, patch_size, strict=True)
    )
    return depth, height, width


class ViT3DStudent(nn.Module):
    """Vision Transformer 3D nhẹ, học từ hai teacher qua knowledge distillation."""

    def __init__(self, cfg: Config, *, in_channels: int = 1) -> None:
        super().__init__()
        self.patch_size = cfg.model.patch_size
        self.volume_shape = cfg.data.target_shape
        self.grid = grid_size(self.volume_shape, self.patch_size)
        self.dim = cfg.model.student_dim

        num_patches = math.prod(self.grid)
        self.num_patches = num_patches

        self.patch_embed = nn.Conv3d(
            in_channels,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(sinusoidal_position_embedding(num_patches + 1, self.dim))
        self.dropout = nn.Dropout(cfg.model.student_dropout)
        self.blocks = nn.ModuleList(
            AttentionBlock(
                self.dim,
                cfg.model.student_heads,
                cfg.model.student_mlp_dim,
                dropout=cfg.model.student_dropout,
            )
            for _ in range(cfg.model.student_depth)
        )
        self.norm = nn.LayerNorm(self.dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, volume: Tensor, *, return_attn: bool = False) -> tuple[Tensor, list[Tensor]]:
        """Trích đặc trưng CLS từ volume.

        Args:
            volume: Tensor ``(B, 1, D, H, W)`` khớp ``cfg.data.target_shape``.
            return_attn: Trả kèm attention của từng khối (cho CAM).

        Returns:
            ``(cls_feature, attentions)`` — ``cls_feature`` shape ``(B, dim)``;
            ``attentions`` là list rỗng khi ``return_attn=False``.
        """
        if volume.shape[-3:] != self.volume_shape:
            raise ValueError(
                f"Student mong đợi volume {self.volume_shape}, "
                f"nhận được {tuple(volume.shape[-3:])}. "
                "Mọi volume phải đi qua build_volume() để đảm bảo shape cố định."
            )

        tokens = self.patch_embed(volume).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.dropout(tokens)

        attentions: list[Tensor] = []
        for block in self.blocks:
            tokens, weights = block(tokens, return_attn=return_attn)
            if return_attn and weights is not None:
                attentions.append(weights)

        return self.norm(tokens[:, 0]), attentions

    def extra_repr(self) -> str:
        return (
            f"volume_shape={self.volume_shape}, patch_size={self.patch_size}, "
            f"grid={self.grid}, num_patches={self.num_patches}, dim={self.dim}"
        )


__all__ = ["ViT3DStudent", "grid_size"]
