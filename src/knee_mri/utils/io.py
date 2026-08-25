"""Tiện ích đọc/ghi file dùng chung."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(obj: Any, path: Path) -> Path:
    """Ghi JSON UTF-8 (giữ nguyên dấu tiếng Việt), tự tạo thư mục cha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_npy(array: np.ndarray, path: Path) -> Path:
    """Ghi mảng numpy, tự tạo thư mục cha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path


def load_npy(path: Path, *, dtype: np.dtype | type | None = np.float32) -> np.ndarray:
    """Đọc mảng numpy và ép kiểu (mặc định float32)."""
    array = np.load(path)
    return array.astype(dtype) if dtype is not None else array


__all__ = ["load_json", "load_npy", "save_json", "save_npy"]
