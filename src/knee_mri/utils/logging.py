"""Thiết lập logging thống nhất cho toàn bộ package.

Thay thế các lệnh ``print()`` rải rác của bản cũ (lỗi P3-10): có timestamp, có
tên module, và ghi song song ra file trong ``artifacts/logs/`` để lần chạy dài
trên Colab còn dấu vết khi phiên bị ngắt.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_configured = False


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_dir: Path | None = None,
    log_name: str = "run",
) -> None:
    """Cấu hình root logger. An toàn khi gọi nhiều lần (lần sau là no-op).

    Args:
        level: Mức log, ví dụ ``logging.DEBUG`` hoặc ``"INFO"``.
        log_dir: Nếu có, thêm handler ghi ra ``<log_dir>/<log_name>.log``.
        log_name: Tên file log (không gồm đuôi).
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{log_name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # transformers/urllib3 rất ồn ở mức INFO
    for noisy in ("urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Trả về logger theo tên module, ví dụ ``get_logger(__name__)``."""
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
