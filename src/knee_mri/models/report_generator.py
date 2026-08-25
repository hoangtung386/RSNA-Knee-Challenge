"""Sinh báo cáo X-quang bằng lời từ vector xác suất dự đoán (bước S3).

Đây là thành phần *giải thích*, không tham gia vào metric của cuộc thi: nó biến
12 con số thành một đoạn văn mà bác sĩ đọc được, giúp kiểm tra xem model có đang
"nghĩ" hợp lý hay không.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from knee_mri.config.schema import Config
from knee_mri.constants import LABELS
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

#: Ngưỡng coi một nhãn là "có mặt" khi diễn giải thành lời.
DEFAULT_THRESHOLD: float = 0.5


def summarise_findings(
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    labels: Sequence[str] = LABELS,
) -> list[tuple[str, float]]:
    """Liệt kê các nhãn vượt ngưỡng, sắp theo xác suất giảm dần."""
    findings = [
        (label, float(prob))
        for label, prob in zip(labels, probabilities, strict=True)
        if prob >= threshold
    ]
    return sorted(findings, key=lambda item: item[1], reverse=True)


def rule_based_report(
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    labels: Sequence[str] = LABELS,
) -> str:
    """Báo cáo dạng mẫu, không cần model — luôn khả dụng.

    Dùng làm fallback khi VLM chưa sẵn sàng, để bước S3 vẫn cho ra sản phẩm.
    """
    findings = summarise_findings(probabilities, threshold=threshold, labels=labels)
    if not findings:
        ranked = sorted(
            zip(labels, probabilities, strict=True), key=lambda item: item[1], reverse=True
        )
        top = ", ".join(f"{name} ({prob:.2f})" for name, prob in ranked[:3])
        return (
            "IMPRESSION: No finding reached the reporting threshold. "
            f"Highest-scoring candidates were: {top}."
        )
    listed = "; ".join(f"{name} (p={prob:.2f})" for name, prob in findings)
    return f"IMPRESSION: Findings suggestive of {listed}."


class GemmaReportGenerator:
    """Sinh báo cáo bằng VLM, tự lùi về mẫu rule-based nếu model lỗi."""

    def __init__(self, cfg: Config, *, device: str = "cuda") -> None:
        self.cfg = cfg
        self.vlm = cfg.vlm
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Đang nạp VLM sinh report %s...", self.vlm.model_id)
        self._processor = AutoProcessor.from_pretrained(self.vlm.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.vlm.model_id,
            torch_dtype=getattr(torch, self.vlm.dtype),
            device_map="auto",
            attn_implementation="eager",
        )
        self._model.eval()

    def build_prompt(
        self, probabilities: Sequence[float] | np.ndarray, *, threshold: float = DEFAULT_THRESHOLD
    ) -> str:
        findings = summarise_findings(probabilities, threshold=threshold)
        if not findings:
            return (
                "You are a musculoskeletal radiologist. A knee MRI model reported no "
                "abnormality above threshold. Write a concise normal impression."
            )
        listed = ", ".join(f"{name} (probability {prob:.2f})" for name, prob in findings)
        return (
            "You are a musculoskeletal radiologist. A knee MRI model predicts the "
            f"following findings: {listed}. Write a concise radiology impression. "
            "State that these are model predictions requiring confirmation."
        )

    def generate(
        self, probabilities: Sequence[float] | np.ndarray, *, threshold: float = DEFAULT_THRESHOLD
    ) -> str:
        """Sinh báo cáo; nếu VLM lỗi thì trả về bản rule-based."""
        probabilities = np.asarray(probabilities, dtype=np.float32).tolist()
        try:
            import torch

            self._ensure_loaded()
            prompt_text = self.build_prompt(probabilities, threshold=threshold)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(text=prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs, max_new_tokens=self.vlm.max_new_tokens, do_sample=False
                )
            return self._processor.decode(generated[0], skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001 - báo cáo là phần phụ trợ, không được làm hỏng job
            logger.warning("VLM sinh report thất bại (%s), dùng mẫu rule-based.", exc)
            return rule_based_report(probabilities, threshold=threshold)


__all__ = [
    "DEFAULT_THRESHOLD",
    "GemmaReportGenerator",
    "rule_based_report",
    "summarise_findings",
]
