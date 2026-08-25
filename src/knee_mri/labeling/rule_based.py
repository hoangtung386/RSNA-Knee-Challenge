"""Gán nhãn yếu bằng luật từ khóa — baseline không cần model.

Cách tiếp cận: tách report thành **mệnh đề**, rồi với mỗi nhãn kiểm tra xem có
mệnh đề nào (a) chứa thuật ngữ của nhãn đó và (b) không bị phủ định hay không.

Tách theo mệnh đề là điểm khác biệt then chốt so với bản cũ, vốn dò phủ định
trong một cửa sổ cứng 25 ký tự trước vị trí khớp. Cửa sổ cố định vừa cắt ngang
mệnh đề (bỏ sót phủ định ở đầu câu dài), vừa nhặt nhầm phủ định của câu trước.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from knee_mri.constants import LABELS
from knee_mri.labeling.keywords import (
    ANATOMY_PATTERNS,
    DIRECT_LABEL_PATTERNS,
    LATERAL_LABEL_RULES,
    NEGATION_PATTERN,
    SIDE_PATTERNS,
)

#: Dấu kết thúc mệnh đề: chấm câu, xuống dòng, dấu chấm phẩy, gạch đầu dòng.
_CLAUSE_SPLIT: re.Pattern[str] = re.compile(r"[.;:\n\r]+|(?:^|\s)[-•>*]\s+")

#: Số ký tự tối đa của một mệnh đề; đoạn dài hơn được cắt tiếp theo dấu phẩy để
#: phủ định không "phủ sóng" quá xa khỏi phạm vi thực của nó.
_MAX_CLAUSE_LEN: int = 160


def split_clauses(report: str) -> list[str]:
    """Tách report thành các mệnh đề đã chuẩn hóa chữ thường."""
    if not report:
        return []
    clauses: list[str] = []
    for chunk in _CLAUSE_SPLIT.split(report.lower()):
        piece = (chunk or "").strip()
        if not piece:
            continue
        if len(piece) <= _MAX_CLAUSE_LEN:
            clauses.append(piece)
        else:
            clauses.extend(part.strip() for part in piece.split(",") if part.strip())
    return clauses


#: Khoảng cách tối đa (ký tự) giữa phát hiện và một từ phủ định đứng **sau** nó
#: để vẫn coi là phủ định — bắt lối nói hậu tố *"ACL normal"*, *"meniscos
#: normales"*, *"Kreuzband intakt"*.
_TRAILING_NEGATION_GAP: int = 30

#: Dấu ngắt ý. Nếu xuất hiện giữa phát hiện và từ phủ định đứng sau thì đó là hai
#: khẳng định riêng biệt (*"Effusion present, no fracture"*), không phải phủ định.
_CLAUSE_BREAKS: frozenset[str] = frozenset(",()/")


def negation_positions(clause: str) -> list[int]:
    """Vị trí bắt đầu của mọi dấu hiệu phủ định trong mệnh đề."""
    return [match.start() for match in NEGATION_PATTERN.finditer(clause)]


def is_negated(clause: str, finding_at: int | None = None) -> bool:
    """Phát hiện tại vị trí ``finding_at`` trong mệnh đề có bị phủ định không.

    Quy tắc, theo thứ tự áp dụng:

    1. Không có dấu hiệu phủ định nào → không phủ định.
    2. Có dấu hiệu đứng **trước** phát hiện → phủ định. Đây là dạng áp đảo:
       *"no evidence of tear"*, *"sin derrame"*, *"kein Knochenödem"*. Xét theo
       vị trí tương đối cũng xử lý đúng câu hỗn hợp như *"Effusion present, no
       fracture"* — nhãn Effusion dương, nhãn Fracture âm.
    3. Dấu hiệu đứng ngay **sau** phát hiện, cách không quá
       :data:`_TRAILING_NEGATION_GAP` ký tự và không có dấu ngắt ý xen giữa →
       phủ định. Bắt lối nói hậu tố mà vẫn giữ đúng *"Effusion present, no
       fracture"* (dấu phẩy tách hai khẳng định).

    Args:
        clause: Mệnh đề đã chuẩn hóa chữ thường.
        finding_at: Vị trí bắt đầu của thuật ngữ bệnh lý. ``None`` nghĩa là hỏi
            chung "mệnh đề này có mang phủ định không".
    """
    positions = negation_positions(clause)
    if not positions:
        return False
    if finding_at is None:
        return True
    if any(position < finding_at for position in positions):
        return True

    trailing = [position for position in positions if position >= finding_at]
    if not trailing:
        return False
    gap = clause[finding_at : trailing[0]]
    return len(gap) <= _TRAILING_NEGATION_GAP and not (_CLAUSE_BREAKS & set(gap))


def _label_match_position(clause: str, label: str) -> int | None:
    """Vị trí sớm nhất mà mệnh đề đề cập tới bệnh lý của nhãn.

    Trả về ``None`` nếu mệnh đề không nhắc tới nhãn này.
    """
    rule = LATERAL_LABEL_RULES.get(label)
    if rule is not None:
        anatomy_key, side = rule
        anatomy = ANATOMY_PATTERNS[anatomy_key].search(clause)
        side_match = SIDE_PATTERNS[side].search(clause)
        if anatomy is None or side_match is None:
            return None
        return min(anatomy.start(), side_match.start())

    pattern = DIRECT_LABEL_PATTERNS.get(label)
    if pattern is None:
        return None
    match = pattern.search(clause)
    return match.start() if match else None


def keyword_labeler(report: str, *, labels: Sequence[str] = LABELS) -> dict[str, int]:
    """Sinh nhãn yếu 0/1 cho từng bệnh lý từ một report.

    Args:
        report: Nội dung radiology report (bất kỳ ngôn ngữ nào trong từ điển).
        labels: Danh sách nhãn cần sinh, mặc định toàn bộ 12 nhãn.

    Returns:
        Dict ``{tên_nhãn: 0 hoặc 1}`` đủ mọi nhãn trong ``labels``.
    """
    result = dict.fromkeys(labels, 0)
    if not isinstance(report, str) or not report.strip():
        return result

    for clause in split_clauses(report):
        for label in labels:
            if result[label]:
                continue
            position = _label_match_position(clause, label)
            if position is not None and not is_negated(clause, position):
                result[label] = 1
    return result


def label_vector(report: str, *, labels: Sequence[str] = LABELS) -> list[int]:
    """Như :func:`keyword_labeler` nhưng trả về vector theo đúng thứ tự nhãn."""
    scores = keyword_labeler(report, labels=labels)
    return [scores[label] for label in labels]


__all__ = [
    "is_negated",
    "keyword_labeler",
    "label_vector",
    "negation_positions",
    "split_clauses",
]
