"""Khối transformer trả về ma trận attention để tính CAM.

Bản cũ dùng ``nn.TransformerEncoderLayer`` rồi cố móc attention bằng forward
hook trên ``layer.self_attn``. Cách đó **không bao giờ hoạt động**: lớp đó gọi
``self_attn(..., need_weights=False)``, nên phần tử thứ hai của tuple trả về luôn
là ``None`` và danh sách hook luôn rỗng (lỗi P0-3). Hệ quả dây chuyền là toàn bộ
tính năng CAM ở S3 trở thành code chết, và biến ``cam`` chưa gán gây
``UnboundLocalError`` (lỗi P0-4).

Ở đây khối tự gọi attention nên luôn kiểm soát được ``need_weights``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AttentionBlock(nn.Module):
    """Khối transformer pre-norm, tùy chọn trả về trọng số attention.

    Dùng pre-norm (LayerNorm trước attention/MLP) vì nó ổn định hơn hẳn khi
    huấn luyện ViT sâu mà không cần lịch warmup phức tạp.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, *, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        """Chạy khối.

        Args:
            x: Tensor ``(B, N, D)``.
            return_attn: Có trả về trọng số attention hay không. Đặt ``False``
                lúc huấn luyện để tiết kiệm bộ nhớ (ma trận ``N × N`` cho mỗi
                head là rất tốn).

        Returns:
            ``(output, attn)`` với ``attn`` có shape ``(B, heads, N, N)`` khi
            ``return_attn=True``, ngược lại là ``None``.
        """
        normed = self.norm_attn(x)
        attended, weights = self.attn(
            normed,
            normed,
            normed,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + attended
        x = x + self.mlp(self.norm_mlp(x))
        return x, weights


def sinusoidal_position_embedding(num_positions: int, dim: int) -> Tensor:
    """Positional embedding sin/cos cố định, dùng để khởi tạo ``pos_embed``.

    Khởi tạo có cấu trúc hội tụ nhanh hơn khởi tạo ngẫu nhiên thuần, trong khi
    tham số vẫn học được tiếp.
    """
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    embedding = torch.zeros(num_positions, dim)
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term[: embedding[:, 1::2].shape[1]])
    return embedding.unsqueeze(0)


__all__ = ["AttentionBlock", "sinusoidal_position_embedding"]
