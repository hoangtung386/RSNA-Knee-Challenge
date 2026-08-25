"""Teacher ảnh: 3DINO-ViT (ViT-Large 3D, DINOv2 thích ứng cho ảnh y tế 3D).

Model được huấn luyện self-supervised trên ~100k scan y tế đa cơ quan, nhận
trực tiếp volume ``(1, 112, 112, 112)`` chuẩn hóa ``[-1, 1]`` và xuất vector
1024 chiều — không cần cộng dồn đặc trưng theo từng lát như teacher 2D.

License: 3DINO mang giấy phép CC BY-NC-ND, chỉ dùng cho nghiên cứu phi thương
mại và **không được sửa** code trong ``vendor/3DINO``. Ở đây ta chỉ nạp và gọi.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from knee_mri.config.schema import Config
from knee_mri.utils.logging import get_logger
from knee_mri.utils.mask import roi_crop

logger = get_logger(__name__)


class DinoUnavailableError(RuntimeError):
    """Thiếu repo 3DINO hoặc file weights."""


class Dino3DTeacher:
    """Bọc 3DINO-ViT ở chế độ đóng băng, chỉ suy luận."""

    def __init__(self, cfg: Config, *, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self.dim = cfg.model.teacher_dim
        self.input_size = cfg.dino.input_size
        self._model: Any = None

    # ----------------------------------------------------------------- nạp
    def _check_assets(self) -> tuple[Path, Path]:
        code_dir = self.cfg.paths.dino_code_dir
        weights = self.cfg.dino.weights_path(self.cfg.paths.artifacts_root)
        if not code_dir.is_dir():
            raise DinoUnavailableError(
                f"Không tìm thấy repo 3DINO tại {code_dir}. "
                f"Hãy clone: git clone {self.cfg.dino.repo_url} {code_dir}"
            )
        if not weights.is_file():
            raise DinoUnavailableError(
                f"Không tìm thấy weights 3DINO-ViT tại {weights}. Tải từ HF Hub "
                f"(cần chấp nhận license và có HF_TOKEN) rồi đặt đúng đường dẫn này."
            )
        return code_dir, weights

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        code_dir, weights = self._check_assets()
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

        from dinov2.configs import load_and_merge_config_3d
        from dinov2.eval.setup import build_model_for_eval

        logger.info("Đang nạp 3DINO-ViT từ %s", weights)
        dino_cfg = load_and_merge_config_3d(self.cfg.dino.config_name)
        model = build_model_for_eval(dino_cfg, str(weights))
        self._model = model.to(self.device).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    # ------------------------------------------------------------ tiền xử lý
    @staticmethod
    def normalize_to_pm1(volume: torch.Tensor) -> torch.Tensor:
        """Chuẩn hóa về ``[-1, 1]`` theo percentile 0.05/99.95.

        Dùng ``torch.quantile`` trên tensor đã phẳng; với volume ``112³`` (1.4M
        phần tử) vẫn nằm dưới giới hạn kích thước của hàm này.
        """
        flat = volume.flatten()
        low = torch.quantile(flat, 0.0005)
        high = torch.quantile(flat, 0.9995)
        if high <= low:
            return torch.zeros_like(volume)
        scaled = (volume - low) / (high - low)
        return torch.clamp(scaled * 2.0 - 1.0, -1.0, 1.0)

    def preprocess(self, volume: np.ndarray, mask: np.ndarray | None = None) -> torch.Tensor:
        """Đưa volume về đúng tensor ``(1, 1, S, S, S)`` mà 3DINO mong đợi."""
        if mask is not None:
            volume = roi_crop(volume, mask)
        tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
        tensor = F.interpolate(
            tensor,
            size=(self.input_size,) * 3,
            mode="trilinear",
            align_corners=False,
        )
        return self.normalize_to_pm1(tensor)

    # --------------------------------------------------------------- suy luận
    @torch.no_grad()
    def extract(self, volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Trích vector đặc trưng ``(teacher_dim,)`` từ một volume.

        Args:
            volume: Mảng ``(D, H, W)`` đã chuẩn hóa.
            mask: Mask ROI tùy chọn để tập trung vào vùng bệnh lý.
        """
        self._ensure_loaded()
        tensor = self.preprocess(volume, mask).to(self.device)
        feature = self._model(tensor)
        if feature.ndim > 2:  # một số biến thể trả về token, lấy trung bình
            feature = feature.mean(dim=tuple(range(1, feature.ndim - 1)))
        return feature[0].float().cpu().numpy().astype(np.float32)


__all__ = ["Dino3DTeacher", "DinoUnavailableError"]
