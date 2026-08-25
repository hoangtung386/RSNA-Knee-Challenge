"""Teacher text: VLM đa modality đọc (ảnh + report) sinh vector guidance ``G``.

``G`` mã hóa tri thức lâm sàng nằm trong report. Student chỉ nhìn ảnh nhưng được
huấn luyện để căn chỉnh với ``G``, nên lúc inference — khi không còn report —
tri thức đó đã được "hấp thụ" vào trọng số của student.

Sửa lỗi P2-5 của bản cũ: các lát cắt được quy đổi qua
:func:`~knee_mri.data.normalize.to_unit_range` trước khi cast uint8. Bản cũ làm
``(volume * 255).astype(np.uint8)`` trên dữ liệu nằm trong ``[-1, 1]``, khiến mọi
voxel âm bị *wrap-around* thành số lớn — ảnh gửi cho VLM là nhiễu thuần túy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from knee_mri.config.schema import Config
from knee_mri.data.normalize import to_uint8_image
from knee_mri.data.volume import middle_slices
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

_GUIDANCE_PROMPT = (
    "You are a musculoskeletal radiologist. Below are representative MRI slices "
    "of a knee examination together with its radiology report. Internally "
    "summarise the clinically salient findings — ligaments, menisci, cartilage, "
    "effusion, bone marrow — into a single rich representation."
)


class GemmaGuidanceEncoder:
    """Sinh embedding guidance bằng cách mean-pool hidden state cuối của VLM."""

    def __init__(self, cfg: Config, *, device: str = "cuda") -> None:
        self.cfg = cfg
        self.vlm = cfg.vlm
        self.device = device
        self.norm_mode = cfg.data.norm_mode
        self.expected_dim = cfg.model.guidance_dim
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Đang nạp VLM guidance %s...", self.vlm.model_id)
        self._processor = AutoProcessor.from_pretrained(self.vlm.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.vlm.model_id,
            torch_dtype=getattr(torch, self.vlm.dtype),
            device_map="auto",
            attn_implementation="eager",
        )
        self._model.eval()

    @property
    def hidden_size(self) -> int:
        """Số chiều hidden state thật của model đã nạp."""
        self._ensure_loaded()
        return int(self._model.config.text_config.hidden_size)

    def _to_pil_images(self, volume: np.ndarray) -> list[Any]:
        """Lấy vài lát giữa volume và đổi thành ảnh RGB hợp lệ."""
        from PIL import Image

        slices = middle_slices(volume, self.vlm.guidance_max_slices)
        return [
            Image.fromarray(to_uint8_image(plane, self.norm_mode)).convert("RGB")
            for plane in slices
        ]

    def encode(self, volume: np.ndarray, report: str) -> np.ndarray:
        """Sinh vector guidance ``(guidance_dim,)`` cho một study.

        Raises:
            ValueError: khi số chiều thật khác ``model.guidance_dim`` trong cấu
                hình. Thà lỗi to ở bước precompute còn hơn để lẫn file ``.npy``
                lệch chiều rồi vỡ giữa chừng lúc huấn luyện (lỗi P2-7).
        """
        import torch

        self._ensure_loaded()
        images = self._to_pil_images(volume)
        content: list[dict[str, str]] = [{"type": "image"} for _ in images]
        content.append(
            {
                "type": "text",
                "text": (
                    f"{_GUIDANCE_PROMPT}\n\nReport:\n{(report or '')[: self.vlm.max_report_chars]}"
                ),
            }
        )
        prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        inputs = self._processor(text=prompt, images=images, return_tensors="pt", padding=True).to(
            self._model.device
        )

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]  # (1, seq, hidden)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        vector = pooled[0].float().cpu().numpy().astype(np.float32)

        if vector.shape[0] != self.expected_dim:
            raise ValueError(
                f"Guidance có {vector.shape[0]} chiều nhưng cấu hình khai báo "
                f"model.guidance_dim={self.expected_dim}. Hãy sửa configs/base.yaml "
                f"cho khớp với {self.vlm.model_id}."
            )
        return vector


__all__ = ["GemmaGuidanceEncoder"]
