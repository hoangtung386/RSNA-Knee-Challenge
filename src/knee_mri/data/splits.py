"""Chia tập train/validation.

Bản cũ hoàn toàn không có validation: không cách nào biết checkpoint nào tốt hơn
checkpoint nào (lỗi P1-7). Tệ hơn, 58 study có nhãn do người gán — tài sản đắt
giá nhất của dataset — không được dùng vào việc gì cả.

Chiến lược ở đây: **giữ nguyên 58 study gold làm validation cố định**. Chúng quá
ít để huấn luyện nhưng vừa đủ để xếp hạng tương đối giữa các checkpoint, và đó
là tín hiệu duy nhất ta có về metric thật của cuộc thi (AUC trên nhãn thật).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from knee_mri.data.catalog import StudyCatalog
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Split:
    """Kết quả chia tập."""

    train: list[str]
    val: list[str]

    def __repr__(self) -> str:
        return f"Split(train={len(self.train)}, val={len(self.val)})"


def make_split(
    catalog: StudyCatalog,
    *,
    seed: int = 42,
    holdout_gold: bool = True,
    extra_val_ratio: float = 0.0,
    limit: int | None = None,
) -> Split:
    """Chia study thành tập train và validation.

    Args:
        catalog: Catalog đã nạp.
        seed: Seed cho phần chọn ngẫu nhiên bổ sung.
        holdout_gold: Giữ toàn bộ study có nhãn gold ra khỏi tập train và đưa
            vào validation. Nên để ``True``.
        extra_val_ratio: Tỉ lệ study *không* có nhãn gold được đưa thêm vào
            validation (đánh giá bằng weak label). ``0.0`` = không lấy thêm.
        limit: Nếu có, cắt bớt tổng số study — dùng cho smoke test.

    Returns:
        :class:`Split` với hai danh sách study rời nhau.
    """
    studies = catalog.studies_with_series()
    if limit is not None:
        studies = studies[:limit]

    gold = set(catalog.studies_with_gold()) & set(studies)
    rest = [uid for uid in studies if uid not in gold]

    val = sorted(gold) if holdout_gold else []
    train = list(rest) if holdout_gold else list(studies)

    if extra_val_ratio > 0.0 and train:
        rng = random.Random(seed)
        shuffled = list(train)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * extra_val_ratio))
        val = val + sorted(shuffled[:cut])
        train = sorted(shuffled[cut:])

    if not val:
        logger.warning(
            "Tập validation rỗng (không có study nào có nhãn gold trong phạm vi đã chọn). "
            "Sẽ không đo được AUC — cân nhắc tăng --limit hoặc đặt extra_val_ratio > 0."
        )

    split = Split(train=train, val=val)
    logger.info("%s | gold trong val: %d", split, len(gold))
    return split


__all__ = ["Split", "make_split"]
