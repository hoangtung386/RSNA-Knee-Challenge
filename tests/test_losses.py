"""Test các hàm loss — đặc biệt khóa lại hai lỗi P0-7 và P0-8."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from knee_mri.training.losses import (  # noqa: E402
    asymmetric_loss,
    auc_margin_loss,
    info_nce,
    kd_cosine_loss,
)


class TestAsymmetricLoss:
    """P0-7: bản cũ cài đặt ASL ngược hướng."""

    def test_easy_negative_costs_less_than_hard_negative(self) -> None:
        """Đây là chính hành vi mà bản cũ làm ngược.

        Mẫu âm dễ (model đoán đúng, p≈0) phải đóng góp loss NHỎ hơn mẫu âm khó
        (model đoán sai, p≈1). Cài đặt cũ dùng ``w_neg = (1-p)**gamma_neg`` cho
        kết quả ngược lại.
        """
        target = torch.zeros(1, 1)
        easy = asymmetric_loss(torch.tensor([[-10.0]]), target)
        hard = asymmetric_loss(torch.tensor([[10.0]]), target)
        assert easy < hard

    def test_easy_positive_costs_less_than_hard_positive(self) -> None:
        target = torch.ones(1, 1)
        easy = asymmetric_loss(torch.tensor([[10.0]]), target)
        hard = asymmetric_loss(torch.tensor([[-10.0]]), target)
        assert easy < hard

    def test_gamma_neg_downweights_easy_negatives(self) -> None:
        """Tăng gamma_neg phải làm mẫu âm dễ đóng góp ÍT hơn nữa.

        Dùng logit -1.0 (p ≈ 0.27, tức trên ngưỡng ``clip``) để phép dịch xác
        suất chưa loại hẳn mẫu — xem
        :meth:`test_probability_shifting_discards_very_easy_negatives`.
        """
        logits = torch.tensor([[-1.0]])
        target = torch.zeros(1, 1)
        weak = asymmetric_loss(logits, target, gamma_neg=0.0)
        strong = asymmetric_loss(logits, target, gamma_neg=4.0)
        assert strong < weak

    def test_probability_shifting_discards_very_easy_negatives(self) -> None:
        """Mẫu âm có ``p < clip`` bị loại hoàn toàn khỏi loss.

        Đây là tác dụng của phép dịch xác suất trong ASL: khi model đã rất chắc
        chắn một mẫu là âm, mẫu đó không còn đóng góp gradient nào nữa. Chính
        tính chất này giúp 12 nhãn cực hiếm (dương tính ~1–2%) không bị chìm
        trong biển mẫu âm.
        """
        target = torch.zeros(1, 1)
        very_easy = asymmetric_loss(torch.tensor([[-6.0]]), target, clip=0.05)
        assert very_easy.item() == pytest.approx(0.0, abs=1e-6)

    def test_gamma_pos_affects_positive_branch(self) -> None:
        """P0-7 phụ: phần dương cũ dùng ``1 - p**g`` thay vì ``(1-p)**g``.

        Hai biểu thức trùng nhau tại gamma_pos=1 nên lỗi bị che giấu; test này
        dùng gamma_pos khác 1 để phát hiện.
        """
        logits = torch.tensor([[2.0]])
        target = torch.ones(1, 1)
        assert asymmetric_loss(logits, target, gamma_pos=0.0) != pytest.approx(
            float(asymmetric_loss(logits, target, gamma_pos=4.0)), rel=1e-3
        )

    def test_is_differentiable(self) -> None:
        logits = torch.randn(4, 12, requires_grad=True)
        asymmetric_loss(logits, torch.zeros(4, 12)).backward()
        assert logits.grad is not None


class TestAucMarginLoss:
    """P0-8: bản cũ trả về ``float`` khi không có cặp positive/negative nào."""

    def test_returns_tensor_when_no_valid_pair(self) -> None:
        """Toàn nhãn âm — không nhãn nào đủ cặp. Vẫn phải là tensor có gradient.

        Bản cũ trả về ``0.0`` (float Python), khiến ``loss.item()`` ném
        ``AttributeError`` và cắt âm thầm nhánh gradient khi cộng vào tổng loss.
        """
        logits = torch.randn(4, 12, requires_grad=True)
        loss = auc_margin_loss(logits, torch.zeros(4, 12))
        assert torch.is_tensor(loss)
        assert loss.requires_grad
        assert loss.item() == pytest.approx(0.0)

    def test_returns_tensor_when_all_positive(self) -> None:
        logits = torch.randn(4, 12, requires_grad=True)
        loss = auc_margin_loss(logits, torch.ones(4, 12))
        assert torch.is_tensor(loss) and loss.requires_grad

    def test_penalises_wrong_ranking(self) -> None:
        """Xếp hạng sai phải chịu loss cao hơn xếp hạng đúng."""
        targets = torch.tensor([[1.0], [0.0]])
        good = auc_margin_loss(torch.tensor([[5.0], [-5.0]]), targets)
        bad = auc_margin_loss(torch.tensor([[-5.0], [5.0]]), targets)
        assert good < bad

    def test_zero_when_margin_satisfied(self) -> None:
        targets = torch.tensor([[1.0], [0.0]])
        loss = auc_margin_loss(torch.tensor([[10.0], [-10.0]]), targets, margin=1.0)
        assert loss.item() == pytest.approx(0.0)


class TestInfoNce:
    def test_perfect_alignment_is_low(self) -> None:
        embedding = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        aligned = info_nce(embedding, embedding)
        shuffled = info_nce(embedding, embedding.flip(0))
        assert aligned < shuffled

    def test_single_sample_returns_grad_tensor(self) -> None:
        """Batch một mẫu không có cặp âm; phải trả tensor 0 chứ không chia cho 0."""
        embedding = torch.nn.functional.normalize(torch.randn(1, 16), dim=-1)
        embedding.requires_grad_(True)
        loss = info_nce(embedding, embedding.detach())
        assert torch.is_tensor(loss) and loss.requires_grad
        assert loss.item() == pytest.approx(0.0)


class TestKdCosineLoss:
    def test_identical_features_give_zero(self) -> None:
        feature = torch.randn(4, 32)
        assert kd_cosine_loss(feature, feature).item() == pytest.approx(0.0, abs=1e-6)

    def test_opposite_features_give_two(self) -> None:
        feature = torch.randn(4, 32)
        assert kd_cosine_loss(feature, -feature).item() == pytest.approx(2.0, abs=1e-5)

    def test_teacher_gradient_is_detached(self) -> None:
        """Teacher đóng băng — gradient không được chảy ngược vào nó."""
        student = torch.randn(4, 32, requires_grad=True)
        teacher = torch.randn(4, 32, requires_grad=True)
        kd_cosine_loss(student, teacher).backward()
        assert student.grad is not None
        assert teacher.grad is None
