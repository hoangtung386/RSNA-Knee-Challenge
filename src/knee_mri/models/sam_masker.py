"""Sinh mask ROI bằng SAM, được dẫn hướng bởi bounding box từ VLM.

Sửa lỗi P2-4: ảnh được quy đổi về RGB uint8 hợp lệ trước khi đưa vào
``SamProcessor``. Bản cũ truyền thẳng mảng ``(H, W)`` float nằm trong ``[-1, 1]``
trong khi processor mong đợi ảnh RGB — mask sinh ra vì thế vô nghĩa.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from knee_mri.config.schema import Config
from knee_mri.data.normalize import to_rgb_image
from knee_mri.utils.logging import get_logger
from knee_mri.utils.mask import center_box_mask

logger = get_logger(__name__)

#: Tỉ lệ cạnh của hộp trung tâm dùng làm prompt mặc định khi VLM không cho bbox.
_DEFAULT_BOX_FRACTION: float = 0.5


def center_bbox(height: int, width: int, fraction: float = _DEFAULT_BOX_FRACTION) -> list[int]:
    """Bounding box ``[x0, y0, x1, y1]`` ở giữa ảnh."""
    margin_h = int(height * (1.0 - fraction) / 2)
    margin_w = int(width * (1.0 - fraction) / 2)
    return [margin_w, margin_h, width - margin_w, height - margin_h]


class SamMaskGenerator:
    """Bọc Segment Anything để sinh mask 3D cho một volume."""

    def __init__(self, cfg: Config, *, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self.norm_mode = cfg.data.norm_mode
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import SamModel, SamProcessor

        logger.info("Đang nạp SAM %s...", self.cfg.sam.model_id)
        self._processor = SamProcessor.from_pretrained(self.cfg.sam.model_id)
        self._model = SamModel.from_pretrained(self.cfg.sam.model_id).to(self.device).eval()

    def mask_from_bbox(self, plane: np.ndarray, bbox: list[int]) -> np.ndarray:
        """Sinh mask 2D ``(H, W)`` uint8 cho một lát cắt từ bounding box."""
        import torch

        self._ensure_loaded()
        image = to_rgb_image(plane, self.norm_mode)
        inputs = self._processor(image, input_boxes=[[bbox]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return (masks[0][0, 0].numpy() > 0).astype(np.uint8)

    def generate_3d(self, volume: np.ndarray, *, bbox: list[int] | None = None) -> np.ndarray:
        """Sinh mask 3D ``(D, H, W)`` cho toàn volume.

        Chiến lược: phân đoạn lát giữa rồi lan mask đó ra mọi lát. Giả định này
        hợp lý cho ROI ở mức khớp gối (vùng quan tâm gần như không đổi giữa các
        lát kề nhau) và rẻ hơn nhiều so với chạy SAM trên từng lát.

        Nếu SAM không nạp được, trả về hộp trung tâm để pipeline vẫn chạy tiếp.
        """
        depth, height, width = volume.shape
        box = bbox or center_bbox(height, width)
        try:
            mask_2d = self.mask_from_bbox(volume[depth // 2], box)
        except Exception as exc:  # noqa: BLE001 - SAM lỗi thì degrade, không dừng cả job
            logger.warning("SAM thất bại (%s), dùng mask hộp trung tâm.", exc)
            return center_box_mask(volume.shape, _DEFAULT_BOX_FRACTION)

        if mask_2d.shape != (height, width):
            logger.warning(
                "SAM trả mask %s khác kích thước lát %s, dùng mask hộp trung tâm.",
                mask_2d.shape,
                (height, width),
            )
            return center_box_mask(volume.shape, _DEFAULT_BOX_FRACTION)

        if mask_2d.sum() == 0:
            logger.debug("SAM trả mask rỗng, dùng mask hộp trung tâm.")
            return center_box_mask(volume.shape, _DEFAULT_BOX_FRACTION)

        return np.broadcast_to(mask_2d[None], (depth, height, width)).copy()


__all__ = ["SamMaskGenerator", "center_bbox"]
