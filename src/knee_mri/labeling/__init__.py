"""Sinh nhãn yếu 12 chiều từ radiology report.

Hai đường:

* :func:`~knee_mri.labeling.rule_based.keyword_labeler` — luật từ khóa đa ngôn
  ngữ, chạy tức thì trên CPU, không cần model. Dùng làm baseline và fallback.
* :class:`~knee_mri.labeling.llm_based.GemmaReportParser` — VLM đọc hiểu report,
  chính xác hơn nhưng cần GPU. Chỉ dùng ở bước precompute offline.
"""

from knee_mri.labeling.llm_based import GemmaReportParser, extract_json_labels
from knee_mri.labeling.rule_based import keyword_labeler, label_vector

__all__ = [
    "GemmaReportParser",
    "extract_json_labels",
    "keyword_labeler",
    "label_vector",
]
