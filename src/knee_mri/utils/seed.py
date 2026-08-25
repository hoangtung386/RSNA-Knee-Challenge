"""Cố định seed để kết quả tái lập được."""

from __future__ import annotations

import os
import random

import numpy as np

from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Đặt seed cho ``random``, ``numpy`` và ``torch`` (nếu đã cài).

    Args:
        seed: Giá trị seed.
        deterministic: Bật thuật toán deterministic của cuDNN. Chậm hơn đáng kể,
            chỉ nên dùng khi cần tái lập chính xác từng bit.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("Chưa cài torch, bỏ qua phần seed của torch.")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


__all__ = ["set_seed"]
