"""Test gán nhãn yếu — khóa lại P2-1 (sụn chêm), P2-2 (MCL/LCL), P2-3 (phủ định)."""

from __future__ import annotations

import pytest

from knee_mri.constants import LABELS
from knee_mri.labeling.llm_based import extract_json_labels
from knee_mri.labeling.rule_based import (
    is_negated,
    keyword_labeler,
    label_vector,
    split_clauses,
)


def positives(report: str) -> set[str]:
    return {label for label, value in keyword_labeler(report).items() if value}


class TestOutputContract:
    def test_returns_all_labels(self) -> None:
        assert set(keyword_labeler("bất kỳ")) == set(LABELS)

    @pytest.mark.parametrize("bad_input", ["", "   ", None, 42])
    def test_handles_missing_report(self, bad_input) -> None:
        assert keyword_labeler(bad_input) == dict.fromkeys(LABELS, 0)

    def test_label_vector_follows_label_order(self) -> None:
        report = "Large joint effusion."
        vector = label_vector(report)
        assert len(vector) == len(LABELS)
        assert vector[LABELS.index("Effusion")] == 1


class TestMeniscusLaterality:
    """P2-1: bản cũ dùng chung từ khóa ``"menisc"`` cho cả hai nhãn sụn chêm.

    Hệ quả là ``Medial Meniscus`` và ``Lateral Meniscus`` LUÔN bằng nhau — đo
    trên 58 study có nhãn gold thì trùng khớp 58/58 (100%).
    """

    def test_medial_only(self) -> None:
        found = positives("Tear of the medial meniscus.")
        assert "Medial Meniscus" in found
        assert "Lateral Meniscus" not in found

    def test_lateral_only(self) -> None:
        found = positives("Torn lateral meniscus.")
        assert "Lateral Meniscus" in found
        assert "Medial Meniscus" not in found

    def test_spanish_laterality(self) -> None:
        """*interno* = trong (medial), *externo* = ngoài (lateral)."""
        assert "Medial Meniscus" in positives("Rotura de menisco interno.")
        assert "Lateral Meniscus" in positives("Rotura de menisco externo.")

    def test_german_compound_word(self) -> None:
        """Tiếng Đức ghép cả giải phẫu lẫn bên vào một từ."""
        assert "Medial Meniscus" in positives("Innenmeniskusriss.")
        assert "Lateral Meniscus" in positives("Aussenmeniskusriss.")

    def test_dutch_compound_word(self) -> None:
        assert "Lateral Meniscus" in positives("Scheur van de buitenmeniscus.")

    def test_generic_mention_flags_neither(self) -> None:
        """Nhắc chung chung, không nói bên nào, thì không suy ra bên nào cả."""
        found = positives("Meniscal degeneration is present.")
        assert "Medial Meniscus" not in found
        assert "Lateral Meniscus" not in found


class TestCollateralLigament:
    """P2-2: bản cũ xếp ``"lcl"`` vào nhãn ``MCL`` — hai dây chằng khác nhau."""

    def test_mcl_detected(self) -> None:
        assert "MCL" in positives("Sprain of the medial collateral ligament.")

    def test_lcl_does_not_trigger_mcl(self) -> None:
        assert "MCL" not in positives("LCL is thickened.")

    def test_german_innenband(self) -> None:
        assert "MCL" in positives("Innenband verdickt.")


class TestMultilingualNegation:
    """P2-3: bản cũ chỉ nhận phủ định tiếng Anh, dữ liệu thì đa ngôn ngữ.

    Khảo sát ``train.csv``: Anh ~1.6k, Tây Ban Nha ~0.7k, Hà Lan ~0.3k,
    Đức ~0.24k, Pháp ~0.08k.
    """

    @pytest.mark.parametrize(
        "report",
        [
            "No evidence of joint effusion.",
            "Sin derrame articular.",
            "Geen vocht in het gewricht.",
            "Kein Gelenkerguss.",
            "Pas de épanchement articulaire.",
        ],
        ids=["en", "es", "nl", "de", "fr"],
    )
    def test_negated_effusion_is_not_flagged(self, report: str) -> None:
        assert "Effusion" not in positives(report)

    def test_positive_effusion_is_flagged(self) -> None:
        assert "Effusion" in positives("Large joint effusion is present.")

    def test_negation_scope_stops_at_clause_boundary(self) -> None:
        """Phủ định chỉ áp cho mệnh đề của nó, không lan sang câu sau."""
        found = positives("No ACL tear. Large joint effusion.")
        assert "ACL" not in found
        assert "Effusion" in found

    def test_mixed_assertion_in_one_clause(self) -> None:
        """*"Effusion present, no fracture"* — một dương, một âm."""
        found = positives("Effusion present, no fracture.")
        assert "Effusion" in found
        assert "Fracture" not in found

    def test_trailing_negation(self) -> None:
        """Lối nói hậu tố: *"ACL normal"* là phủ định."""
        found = positives("ACL normal. Baker cyst present.")
        assert "ACL" not in found
        assert "Baker's" in found

    def test_coordinated_negation(self) -> None:
        """Một phủ định phủ nhiều phát hiện liệt kê sau nó."""
        found = positives("No evidence of tear, effusion, or fracture.")
        assert found == set()

    def test_german_comma_separated_negation(self) -> None:
        assert "Contusion" not in positives("Homogenes Knochenmarksignal, kein Knochenödem.")


class TestClauseHelpers:
    def test_split_lowercases_and_drops_empty(self) -> None:
        assert split_clauses("A. B;; C\n") == ["a", "b", "c"]

    def test_is_negated_without_position(self) -> None:
        assert is_negated("no evidence of tear") is True
        assert is_negated("large effusion") is False


class TestJsonExtraction:
    def test_parses_plain_object(self) -> None:
        assert extract_json_labels('{"ACL": true, "Effusion": false}')["ACL"] == 1

    def test_handles_nested_braces(self) -> None:
        """Bản cũ dùng regex ``\\{.*?\\}`` nên cắt ngang JSON có ngoặc lồng."""
        assert extract_json_labels('{"meta": {"x": 1}, "ACL": true}')["ACL"] == 1

    def test_ignores_surrounding_prose(self) -> None:
        assert extract_json_labels('Here you go: {"Fracture": 1} hope that helps')["Fracture"] == 1

    @pytest.mark.parametrize("value", ["true", "yes", "1", 1, True, "present"])
    def test_accepts_truthy_variants(self, value) -> None:
        import json

        assert extract_json_labels(json.dumps({"ACL": value}))["ACL"] == 1

    def test_returns_all_zeros_on_garbage(self) -> None:
        assert extract_json_labels("không có json ở đây") == dict.fromkeys(LABELS, 0)
