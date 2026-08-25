"""Các bước của pipeline, mỗi bước là một hàm gọi được độc lập.

Thay cho ``src/pipeline.py`` cũ vốn trộn lẫn orchestration, logic nghiệp vụ và
phân tích tham số dòng lệnh trong cùng một file.

Thứ tự chạy::

    generate_masks              (S1)
    precompute_weak_labels      (S1)
    precompute_teacher_features (S2, chuẩn bị)
    precompute_guidance         (S2, chuẩn bị)
    train                       (S2)
    explain_studies             (S3)
    predict_submission          (submit)
"""

from knee_mri.pipeline.explain_step import ExplanationResult, explain_studies
from knee_mri.pipeline.features import precompute_guidance, precompute_teacher_features
from knee_mri.pipeline.masks import generate_masks
from knee_mri.pipeline.weak_labels import precompute_weak_labels

__all__ = [
    "ExplanationResult",
    "explain_studies",
    "generate_masks",
    "precompute_guidance",
    "precompute_teacher_features",
    "precompute_weak_labels",
]
