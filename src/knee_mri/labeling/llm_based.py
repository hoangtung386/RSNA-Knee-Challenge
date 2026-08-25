"""Gán nhãn yếu bằng VLM (Gemma) — chính xác hơn luật từ khóa nhưng tốn GPU.

Quan trọng: lớp này **chỉ** được dùng ở bước precompute offline
(``scripts/precompute_weak_labels.py``), không bao giờ trong ``Dataset.__getitem__``.
Bản cũ khởi tạo một ``GemmaReportParser`` mới cho *mỗi sample* của DataLoader
(lỗi P1-3) — với ``num_workers > 0`` thì mỗi worker nạp một bản Gemma vào VRAM và
OOM ngay lập tức.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from knee_mri.config.schema import VlmCfg
from knee_mri.constants import LABELS
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = (
    "You are a musculoskeletal radiologist. Read the knee MRI report below "
    "(it may be in English, Spanish, Dutch, German, French, or another language) "
    "and output ONLY a JSON object with exactly these {n} boolean keys, "
    "true when the finding is present and false otherwise:\n"
    "{keys}\n\n"
    "Report:\n{report}\n\nJSON:"
)


def extract_json_labels(text: str, *, labels: Sequence[str] = LABELS) -> dict[str, int]:
    """Trích dict nhãn 0/1 từ đầu ra tự do của LLM.

    Quét **mọi** khối ``{...}`` cân bằng ngoặc thay vì dùng regex không tham lam
    ``\\{.*?\\}`` như bản cũ — biểu thức đó cắt ngang JSON có ngoặc lồng nhau và
    thường trả về mảnh vỡ không parse được.
    """
    fallback = dict.fromkeys(labels, 0)
    for candidate in _balanced_json_blocks(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return {name: _coerce_bool(parsed.get(name)) for name in labels}
    logger.debug("Không tìm thấy JSON hợp lệ trong đầu ra LLM (%d ký tự).", len(text))
    return fallback


def _balanced_json_blocks(text: str) -> list[str]:
    """Liệt kê các đoạn ``{...}`` có ngoặc cân bằng, dài nhất trước."""
    blocks: list[str] = []
    stack: list[int] = []
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            if not stack:
                blocks.append(text[start : index + 1])
    return sorted(blocks, key=len, reverse=True)


def _coerce_bool(value: Any) -> int:
    """Đổi giá trị JSON tùy ý thành 0/1, chấp nhận cả ``"true"``/``1``/``"yes"``."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "yes", "1", "present", "positive"})
    return 0


class GemmaReportParser:
    """Parse radiology report thành vector nhãn 12 chiều bằng Gemma.

    Model được nạp **lười** ở lần dùng đầu tiên và giữ lại cho các lần sau, nên
    một instance xử lý được cả nghìn report mà chỉ nạp weights một lần.
    """

    def __init__(self, cfg: VlmCfg, *, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Đang nạp VLM %s (dtype=%s)...", self.cfg.model_id, self.cfg.dtype)
        self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model_id,
            torch_dtype=getattr(torch, self.cfg.dtype),
            device_map="auto",
            attn_implementation="eager",
        )
        self._model.eval()

    def build_prompt(self, report: str) -> str:
        return _PROMPT_TEMPLATE.format(
            n=len(LABELS),
            keys=", ".join(LABELS),
            report=(report or "")[: self.cfg.max_report_chars],
        )

    def parse(self, report: str) -> dict[str, int]:
        """Trả về dict ``{nhãn: 0/1}``. Ném ngoại lệ nếu model lỗi."""
        import torch

        self._ensure_loaded()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": self.build_prompt(report)}]}
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False
            )
        text = self._processor.decode(generated[0], skip_special_tokens=True)
        return extract_json_labels(text)


__all__ = ["GemmaReportParser", "extract_json_labels"]
