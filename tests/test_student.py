"""Test student ViT-3D — khóa lại hai lỗi P0-3 (attention) và P0-5 (pos_embed)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config  # noqa: E402
from knee_mri.explain.cam import (  # noqa: E402
    NoAttentionError,
    attention_rollout,
    attention_to_volume,
)
from knee_mri.models.student import ViT3DStudent, grid_size  # noqa: E402


@pytest.fixture
def student(cfg: Config) -> ViT3DStudent:
    return ViT3DStudent(cfg)


@pytest.fixture
def batch(cfg: Config) -> torch.Tensor:
    return torch.randn(2, 1, *cfg.data.target_shape)


class TestGridSize:
    def test_computes_patch_grid(self) -> None:
        assert grid_size((64, 128, 128), (8, 16, 16)) == (8, 8, 8)

    def test_rejects_non_divisible_shape(self) -> None:
        """Chia dư phải báo lỗi: Conv3d sẽ âm thầm bỏ rìa, làm lệch CAM."""
        with pytest.raises(ValueError, match="không chia hết"):
            grid_size((8, 30, 32), (4, 8, 8))


class TestAttention:
    """P0-3: bản cũ dùng forward hook nên danh sách attention LUÔN rỗng."""

    def test_returns_attention_for_every_block(
        self, student: ViT3DStudent, batch: torch.Tensor, cfg: Config
    ) -> None:
        _, attentions = student(batch, return_attn=True)
        assert len(attentions) == cfg.model.student_depth
        assert all(attention is not None for attention in attentions)

    def test_attention_has_expected_shape(
        self, student: ViT3DStudent, batch: torch.Tensor, cfg: Config
    ) -> None:
        _, attentions = student(batch, return_attn=True)
        tokens = student.num_patches + 1
        assert attentions[0].shape == (2, cfg.model.student_heads, tokens, tokens)

    def test_attention_rows_sum_to_one(self, student: ViT3DStudent, batch: torch.Tensor) -> None:
        """Softmax attention phải cộng lại thành 1 trên mỗi hàng.

        Bắt buộc gọi ``eval()``: ở chế độ train, dropout được áp lên chính trọng
        số attention nên tổng hàng không còn bằng 1.
        """
        student.eval()
        _, attentions = student(batch, return_attn=True)
        row_sums = attentions[0].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_skipped_when_not_requested(self, student: ViT3DStudent, batch: torch.Tensor) -> None:
        """Mặc định không trả attention — ma trận N×N mỗi head rất tốn bộ nhớ."""
        _, attentions = student(batch)
        assert attentions == []


class TestPositionEmbedding:
    """P0-5: bản cũ tạo lại ``nn.Parameter`` ngay trong ``forward``."""

    def test_is_not_recreated_during_forward(
        self, student: ViT3DStudent, batch: torch.Tensor
    ) -> None:
        identity_before = id(student.pos_embed)
        student(batch)
        student(batch)
        assert id(student.pos_embed) == identity_before

    def test_is_registered_as_parameter(self, student: ViT3DStudent) -> None:
        """Phải nằm trong ``parameters()``, nếu không optimizer sẽ không thấy."""
        assert any(param is student.pos_embed for param in student.parameters())

    def test_receives_gradient(self, student: ViT3DStudent, batch: torch.Tensor) -> None:
        """Bản cũ: pos_embed mới nằm ngoài optimizer nên vĩnh viễn bằng 0.

        Dùng tổng bình phương chứ không phải tổng thường: đầu ra đi qua
        ``LayerNorm`` nên tổng các thành phần luôn bằng 0 theo định nghĩa, khiến
        gradient của phép tổng cũng bằng 0 và test sẽ luôn thất bại một cách vô nghĩa.
        """
        feature, _ = student(batch)
        feature.pow(2).sum().backward()
        assert student.pos_embed.grad is not None
        assert student.pos_embed.grad.abs().sum() > 0

    def test_state_dict_shape_is_stable(
        self, student: ViT3DStudent, batch: torch.Tensor, cfg: Config
    ) -> None:
        """Shape ổn định giữa các lần lưu, nếu không ``load_state_dict`` sẽ vỡ."""
        shape_before = student.state_dict()["pos_embed"].shape
        student(batch)
        student(torch.randn(5, 1, *cfg.data.target_shape))
        assert student.state_dict()["pos_embed"].shape == shape_before


class TestForward:
    def test_output_shape(self, student: ViT3DStudent, batch: torch.Tensor, cfg: Config) -> None:
        feature, _ = student(batch)
        assert feature.shape == (2, cfg.model.student_dim)

    def test_rejects_wrong_volume_shape(self, student: ViT3DStudent) -> None:
        """P0-6: shape sai phải báo lỗi rõ ràng ngay, không đi tiếp âm thầm."""
        with pytest.raises(ValueError, match="mong đợi volume"):
            student(torch.randn(1, 1, 16, 64, 64))


class TestCamPipeline:
    """CAM chỉ chạy được vì attention thực sự tồn tại — chuỗi sửa P0-3 → P0-4."""

    def test_rollout_produces_patch_map(self, student: ViT3DStudent, batch: torch.Tensor) -> None:
        _, attentions = student(batch, return_attn=True)
        rollout = attention_rollout(attentions)
        assert rollout.shape == (2, student.num_patches)
        assert torch.allclose(rollout.sum(dim=-1), torch.ones(2), atol=1e-4)

    def test_rollout_rejects_empty_list(self) -> None:
        with pytest.raises(NoAttentionError):
            attention_rollout([])

    def test_cam_maps_back_to_volume(
        self, student: ViT3DStudent, batch: torch.Tensor, cfg: Config
    ) -> None:
        _, attentions = student(batch, return_attn=True)
        cam = attention_to_volume(
            attention_rollout(attentions), student.grid, cfg.data.target_shape
        )
        assert cam.shape == cfg.data.target_shape
        assert cam.min() >= 0.0 and cam.max() <= 1.0

    def test_cam_rejects_mismatched_grid(
        self, student: ViT3DStudent, batch: torch.Tensor, cfg: Config
    ) -> None:
        _, attentions = student(batch, return_attn=True)
        with pytest.raises(ValueError, match="Số patch không khớp"):
            attention_to_volume(attention_rollout(attentions), (1, 1, 1), cfg.data.target_shape)
