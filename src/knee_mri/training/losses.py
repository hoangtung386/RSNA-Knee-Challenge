"""Các hàm mục tiêu cho huấn luyện student.

Bối cảnh: chỉ 58/4407 study có nhãn do người gán, nhưng **mọi** study đều có
report. Chiến lược là bóc tri thức từ 4407 cặp (ảnh, report) thay vì chỉ dựa vào
58 nhãn. Bốn thành phần:

1. :func:`info_nce` — căn chỉnh đặc trưng ảnh của student với guidance text
   trong không gian chung (kiểu CLIP). Đây là động cơ giám sát yếu chính.
2. :func:`asymmetric_loss` — cho 12 nhãn cực mất cân bằng (dương tính ~1–2%).
3. :func:`auc_margin_loss` — tối ưu trực tiếp thứ hạng, tức metric AUC.
4. :func:`kd_cosine_loss` — giữ đặc trưng ảnh mạnh từ 3DINO-ViT.

Tổng: ``L = λ_kd·L_kd + λ_ctr·L_infonce + λ_cls·L_asl + λ_auc·L_auc``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

_EPS: float = 1e-8


def info_nce(
    image_embedding: Tensor, text_embedding: Tensor, *, temperature: float = 0.07
) -> Tensor:
    """InfoNCE đối xứng giữa đặc trưng ảnh và guidance text.

    Mỗi study trong batch là một cặp dương; các study còn lại là cặp âm. Vì thế
    ``batch_size`` càng lớn thì tín hiệu càng mạnh — với batch quá nhỏ, loss này
    gần như không học được gì.

    Args:
        image_embedding: ``(B, D)`` đã L2-normalize.
        text_embedding: ``(B, D)`` đã L2-normalize.
        temperature: Nhiệt độ softmax; nhỏ hơn = phân biệt gắt hơn.
    """
    if image_embedding.shape[0] < 2:
        # Một mẫu thì không có cặp âm nào; trả về 0 nhưng vẫn giữ đồ thị gradient.
        return image_embedding.sum() * 0.0
    logits = image_embedding @ text_embedding.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2


def asymmetric_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
) -> Tensor:
    """Asymmetric Loss (Ben-Baruch et al., 2021) cho multi-label mất cân bằng.

    Ý tưởng: hạ trọng số các mẫu âm **dễ** (model đã đoán đúng là âm với độ tin
    cậy cao) để gradient dồn vào các mẫu dương hiếm và mẫu âm khó.

    Bản cũ cài đặt **ngược hướng**: dùng ``w_neg = (1 - p) ** gamma_neg``, nghĩa
    là mẫu âm dễ (``p ≈ 0``) nhận trọng số ``≈ 1`` (không hề bị hạ) còn mẫu âm
    khó (``p ≈ 1``) nhận trọng số ``≈ 0`` (bị bỏ qua) — đúng ngược lại ý đồ
    (lỗi P0-7). Phần dương cũng sai: ``1 - p ** gamma_pos`` thay vì
    ``(1 - p) ** gamma_pos``; hai biểu thức tình cờ trùng nhau tại
    ``gamma_pos = 1`` nên lỗi bị che giấu ở cấu hình mặc định.

    Args:
        logits: ``(B, C)`` logit thô, chưa qua sigmoid.
        targets: ``(B, C)`` giá trị trong ``{0, 1}``.
        gamma_neg: Số mũ focal cho mẫu âm; càng lớn càng hạ mạnh mẫu âm dễ.
        gamma_pos: Số mũ focal cho mẫu dương.
        clip: Dịch xác suất mẫu âm ("probability shifting") để bỏ hẳn các mẫu âm
            rất dễ ra khỏi loss.
    """
    probs = torch.sigmoid(logits)
    probs_neg = (1.0 - probs + clip).clamp(max=1.0)

    loss_pos = targets * ((1.0 - probs) ** gamma_pos) * torch.log(probs.clamp(min=_EPS))
    loss_neg = (
        (1.0 - targets) * ((1.0 - probs_neg) ** gamma_neg) * torch.log(probs_neg.clamp(min=_EPS))
    )
    return -(loss_pos + loss_neg).sum(dim=1).mean()


def auc_margin_loss(logits: Tensor, targets: Tensor, *, margin: float = 1.0) -> Tensor:
    """Hinge theo cặp, thay thế khả vi cho AUC — metric chính của cuộc thi.

    Với mỗi nhãn, đẩy điểm của mọi mẫu dương cao hơn mọi mẫu âm ít nhất
    ``margin``. Nhãn nào không có đủ cả hai loại trong batch thì bỏ qua.

    Luôn trả về ``Tensor`` gắn với đồ thị tính toán, kể cả khi không nhãn nào đủ
    điều kiện. Bản cũ khởi tạo ``total = 0.0`` (float Python) nên trong trường
    hợp đó trả về float, làm ``loss.item()`` ném ``AttributeError`` và âm thầm
    cắt nhánh gradient khi cộng vào tổng loss (lỗi P0-8).

    Cảnh báo thực tế: với tỉ lệ dương ~1–2% và batch nhỏ, phần lớn batch sẽ không
    có nhãn nào đủ cặp. Muốn loss này thực sự hoạt động cần batch lớn hoặc
    sampler cân bằng.
    """
    total = logits.sum() * 0.0  # tensor giá trị 0 nhưng vẫn nối vào đồ thị
    counted = 0
    for column in range(logits.shape[1]):
        column_targets = targets[:, column]
        positives = logits[column_targets > 0.5, column]
        negatives = logits[column_targets < 0.5, column]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        differences = positives.unsqueeze(1) - negatives.unsqueeze(0)
        total = total + F.relu(margin - differences).mean()
        counted += 1
    return total / max(counted, 1)


def kd_cosine_loss(student_feature: Tensor, teacher_feature: Tensor) -> Tensor:
    """Khoảng cách cosine giữa đặc trưng student và teacher (đã detach).

    Dùng cosine thay vì MSE vì ta muốn khớp *hướng* của biểu diễn chứ không phải
    độ lớn — độ lớn phụ thuộc vào thang đo riêng của từng kiến trúc.
    """
    student = F.normalize(student_feature, dim=-1)
    teacher = F.normalize(teacher_feature.detach(), dim=-1)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


__all__ = ["asymmetric_loss", "auc_margin_loss", "info_nce", "kd_cosine_loss"]
