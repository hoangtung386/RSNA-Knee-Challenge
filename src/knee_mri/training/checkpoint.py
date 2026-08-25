"""Lưu và nạp checkpoint an toàn.

Hai điểm sửa so với bản cũ:

* ``torch.load`` luôn dùng ``weights_only=True``. Không có cờ này, việc nạp
  checkpoint là thực thi pickle tùy ý — vừa là lỗ hổng bảo mật vừa sẽ thành lỗi
  cứng ở PyTorch phiên bản mới.
* Theo dõi **checkpoint tốt nhất** theo metric validation, thay vì ghi đè một
  file duy nhất mỗi epoch rồi không biết bản nào tốt (lỗi P1-7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

BEST_NAME: str = "student_best.pt"
LAST_NAME: str = "student_last.pt"


@dataclass
class CheckpointState:
    """Trạng thái đầy đủ của một lần huấn luyện."""

    epoch: int = 0
    best_metric: float = float("-inf")
    modules: dict[str, nn.Module] = field(default_factory=dict)
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "modules": {name: module.state_dict() for name, module in self.modules.items()},
        }
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        return payload


def save_checkpoint(state: CheckpointState, path: Path) -> Path:
    """Ghi checkpoint ra đĩa, tạo thư mục cha nếu cần."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.to_dict(), path)
    logger.debug("Đã lưu checkpoint: %s (epoch %d)", path, state.epoch)
    return path


def load_checkpoint(
    path: Path,
    modules: dict[str, nn.Module],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[int, float]:
    """Nạp checkpoint vào các module đã cho.

    Args:
        path: Đường dẫn file checkpoint.
        modules: Dict ``{tên: module}`` cần khôi phục trọng số.
        optimizer: Nếu có, khôi phục cả trạng thái optimizer (để resume).
        scheduler: Nếu có, khôi phục trạng thái scheduler.
        map_location: Thiết bị đích khi nạp.
        strict: Bắt buộc khớp hoàn toàn tên tham số.

    Returns:
        ``(epoch_tiếp_theo, best_metric)``.

    Raises:
        FileNotFoundError: khi file không tồn tại.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=True)
    saved_modules: dict[str, Any] = payload.get("modules", {})
    for name, module in modules.items():
        if name not in saved_modules:
            logger.warning("Checkpoint không chứa module %r, giữ nguyên khởi tạo.", name)
            continue
        module.load_state_dict(saved_modules[name], strict=strict)

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])

    epoch = int(payload.get("epoch", 0))
    best = float(payload.get("best_metric", float("-inf")))
    logger.info("Đã nạp checkpoint %s (epoch %d, best=%.4f)", path.name, epoch, best)
    return epoch, best


def load_for_inference(
    path: Path,
    modules: dict[str, nn.Module],
    *,
    device: str | torch.device = "cpu",
) -> None:
    """Nạp trọng số cho suy luận và chuyển mọi module sang ``eval()``."""
    load_checkpoint(path, modules, map_location=device)
    for module in modules.values():
        module.to(device).eval()


__all__ = [
    "BEST_NAME",
    "LAST_NAME",
    "CheckpointState",
    "load_checkpoint",
    "load_for_inference",
    "save_checkpoint",
]
