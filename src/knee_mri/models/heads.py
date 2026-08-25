"""Đầu ra của student: phân loại 12 nhãn và các projector cho KD/contrastive."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from knee_mri.config.schema import Config
from knee_mri.constants import METADATA_DIM, NUM_LABELS, PLANES
from knee_mri.data.catalog import SeriesInfo


def make_metadata_vector(series: SeriesInfo) -> list[float]:
    """Mã hóa metadata series thành vector 5 chiều.

    Ba trường này có mặt ở **cả** ``train_series.csv`` lẫn ``test_series.csv``,
    nên dùng được làm đầu vào phụ trợ lúc inference — khác với report vốn chỉ có
    lúc huấn luyện.

    Returns:
        one-hot mặt phẳng (3) + Fluid_Sensitive (1) + Fat_Suppression (1).
    """
    plane_onehot = [float(series.plane == plane) for plane in PLANES]
    return [*plane_onehot, float(series.fluid_sensitive), float(series.fat_suppression)]


class MetadataEncoder(nn.Module):
    """Nâng vector metadata 5 chiều lên cùng số chiều với đặc trưng ảnh."""

    def __init__(self, out_dim: int, *, in_dim: int = METADATA_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, metadata: Tensor) -> Tensor:
        return self.net(metadata)


class ClassifierHead(nn.Module):
    """Sinh 12 logit từ đặc trưng ảnh, tùy chọn hợp nhất với metadata."""

    def __init__(self, cfg: Config, *, num_labels: int = NUM_LABELS) -> None:
        super().__init__()
        dim = cfg.model.student_dim
        self.use_metadata = cfg.model.use_metadata
        self.metadata_encoder = (
            MetadataEncoder(dim, in_dim=cfg.model.metadata_dim) if self.use_metadata else None
        )
        in_dim = dim * 2 if self.use_metadata else dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, feature: Tensor, metadata: Tensor | None = None) -> Tensor:
        """Trả về logit thô ``(B, num_labels)`` — chưa qua sigmoid."""
        if self.metadata_encoder is not None:
            if metadata is None:
                raise ValueError(
                    "ClassifierHead được cấu hình dùng metadata nhưng không nhận được tensor "
                    "metadata. Đặt model.use_metadata=false nếu muốn bỏ nhánh này."
                )
            feature = torch.cat([feature, self.metadata_encoder(metadata)], dim=-1)
        return self.head(feature)


class KDProjector(nn.Module):
    """Chiếu đặc trưng student sang không gian của teacher để so khớp.

    Dùng MLP hai lớp thay vì một phép chiếu tuyến tính: teacher và student có
    kiến trúc rất khác nhau, nên ép khớp tuyến tính là ràng buộc quá chặt và
    thường làm hỏng biểu diễn của student.
    """

    def __init__(self, in_dim: int, out_dim: int, *, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.proj(feature)


class ContrastiveProjector(nn.Module):
    """Chiếu sang không gian contrastive chung rồi L2-normalize (cho InfoNCE)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return F.normalize(self.proj(feature), dim=-1)


__all__ = [
    "ClassifierHead",
    "ContrastiveProjector",
    "KDProjector",
    "MetadataEncoder",
    "make_metadata_vector",
]
