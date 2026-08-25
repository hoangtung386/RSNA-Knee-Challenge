"""Test phần logic thuần của các wrapper model nặng.

Bản thân 3DINO-ViT, SAM và VLM cần GPU cùng weights đã tải, nên không test được
trong CI. Nhưng phần *quanh* chúng — tiền xử lý, dựng prompt, thông báo lỗi,
đường suy giảm khi thiếu model — là logic thường xuyên bị sửa và hoàn toàn kiểm
chứng được. Đây chính là chỗ nằm của các lỗi P2-4 và P2-5.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from knee_mri.config import Config  # noqa: E402
from knee_mri.constants import LABELS, NUM_LABELS  # noqa: E402
from knee_mri.models.report_generator import (  # noqa: E402
    GemmaReportGenerator,
    rule_based_report,
    summarise_findings,
)
from knee_mri.models.teachers.dino3d import Dino3DTeacher, DinoUnavailableError  # noqa: E402
from knee_mri.models.teachers.gemma_guidance import GemmaGuidanceEncoder  # noqa: E402


class TestDino3DPreprocessing:
    def test_output_matches_expected_input_shape(self, cfg: Config, volume: np.ndarray) -> None:
        """3DINO nhận đúng khối lập phương ``input_size³``."""
        teacher = Dino3DTeacher(cfg, device="cpu")
        tensor = teacher.preprocess(volume)
        size = cfg.dino.input_size
        assert tensor.shape == (1, 1, size, size, size)

    def test_output_is_in_pm1_range(self, cfg: Config, volume: np.ndarray) -> None:
        tensor = Dino3DTeacher(cfg, device="cpu").preprocess(volume)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_constant_volume_does_not_divide_by_zero(self, cfg: Config) -> None:
        """Volume phẳng (toàn khí, hoặc series hỏng) không được sinh NaN."""
        flat = np.full(cfg.data.target_shape, 0.5, dtype=np.float32)
        tensor = Dino3DTeacher(cfg, device="cpu").preprocess(flat)
        assert torch.isfinite(tensor).all()

    def test_roi_mask_is_applied_before_resize(self, cfg: Config, volume: np.ndarray) -> None:
        """Cắt ROI đổi nội dung nhưng shape đầu ra vẫn cố định."""
        mask = np.zeros_like(volume, dtype=np.uint8)
        mask[:, : volume.shape[1] // 2, :] = 1
        teacher = Dino3DTeacher(cfg, device="cpu")
        without = teacher.preprocess(volume)
        with_mask = teacher.preprocess(volume, mask)
        assert with_mask.shape == without.shape
        assert not torch.allclose(with_mask, without)

    def test_missing_repo_gives_actionable_error(self, cfg: Config) -> None:
        """Thông báo lỗi phải nói rõ cần làm gì, kèm lệnh clone."""
        with pytest.raises(DinoUnavailableError, match="git clone"):
            Dino3DTeacher(cfg, device="cpu")._ensure_loaded()

    def test_weights_path_honours_config(self, cfg: Config) -> None:
        """``dino.weight_file`` phải thực sự quyết định đường dẫn weights.

        Trước đây tên file được viết cứng trong một property của ``Paths``, nên
        giá trị khai trong YAML bị nuốt im lặng.
        """
        from dataclasses import replace

        custom = replace(cfg.dino, weight_file="weights_khac.pth")
        assert custom.weights_path(cfg.paths.artifacts_root).name == "weights_khac.pth"


class TestGuidanceEncoder:
    def test_slices_are_valid_rgb_images(self, cfg: Config, volume: np.ndarray) -> None:
        """P2-5: volume trong [-1, 1] phải quy đổi đúng, không wrap-around uint8."""
        images = GemmaGuidanceEncoder(cfg, device="cpu")._to_pil_images(volume)
        assert len(images) == cfg.vlm.guidance_max_slices
        for image in images:
            assert image.mode == "RGB"
            pixels = np.asarray(image)
            assert pixels.dtype == np.uint8
            assert pixels.min() >= 0 and pixels.max() <= 255

    def test_dark_voxels_map_to_dark_pixels(self, cfg: Config) -> None:
        """Voxel tối nhất (-1) phải ra pixel 0, không phải một số lớn.

        Cast thẳng ``(volume * 255).astype(np.uint8)`` trên dữ liệu âm khiến -1.0
        thành 1 và -0.5 thành 129 — ảnh gửi cho VLM là nhiễu.
        """
        volume = np.full(cfg.data.target_shape, -1.0, dtype=np.float32)
        volume[..., 0] = 1.0  # cần một điểm sáng để dải giá trị không suy biến
        image = np.asarray(GemmaGuidanceEncoder(cfg, device="cpu")._to_pil_images(volume)[0])
        assert image[..., 0].min() == 0


class TestReportGeneration:
    def test_summarise_keeps_only_findings_above_threshold(self) -> None:
        probabilities = np.zeros(NUM_LABELS)
        probabilities[0] = 0.9
        probabilities[1] = 0.2
        findings = summarise_findings(probabilities, threshold=0.5)
        assert [name for name, _ in findings] == [LABELS[0]]

    def test_summarise_sorts_by_confidence(self) -> None:
        probabilities = np.full(NUM_LABELS, 0.6)
        probabilities[3] = 0.99
        findings = summarise_findings(probabilities, threshold=0.5)
        assert findings[0][0] == LABELS[3]

    def test_report_mentions_every_positive_finding(self) -> None:
        probabilities = np.zeros(NUM_LABELS)
        probabilities[[0, 7]] = 0.8
        report = rule_based_report(probabilities)
        assert LABELS[0] in report and LABELS[7] in report

    def test_negative_report_still_names_top_candidates(self) -> None:
        """Không phát hiện gì thì vẫn nêu ứng viên cao nhất — hữu ích khi rà soát."""
        probabilities = np.linspace(0.0, 0.4, NUM_LABELS)
        report = rule_based_report(probabilities)
        assert "No finding reached" in report
        assert LABELS[-1] in report

    def test_prompt_reflects_findings(self, cfg: Config) -> None:
        generator = GemmaReportGenerator(cfg, device="cpu")
        probabilities = np.zeros(NUM_LABELS)
        probabilities[2] = 0.95
        assert LABELS[2] in generator.build_prompt(probabilities)

    def test_prompt_for_normal_case(self, cfg: Config) -> None:
        prompt = GemmaReportGenerator(cfg, device="cpu").build_prompt(np.zeros(NUM_LABELS))
        assert "normal impression" in prompt

    def test_generate_falls_back_when_vlm_unavailable(self, cfg: Config) -> None:
        """VLM lỗi thì vẫn phải trả về một báo cáo, không được ném ngoại lệ."""
        probabilities = np.zeros(NUM_LABELS)
        probabilities[0] = 0.9
        report = GemmaReportGenerator(cfg, device="cpu").generate(probabilities)
        assert LABELS[0] in report


class TestWeakLabelPrompt:
    def test_prompt_lists_all_twelve_labels(self, cfg: Config) -> None:
        from knee_mri.labeling.llm_based import GemmaReportParser

        prompt = GemmaReportParser(cfg.vlm, device="cpu").build_prompt("một report nào đó")
        for label in LABELS:
            assert label in prompt

    def test_prompt_truncates_long_reports(self, cfg: Config) -> None:
        """Report dài phải bị cắt để không nổ số token."""
        from knee_mri.labeling.llm_based import GemmaReportParser

        parser = GemmaReportParser(cfg.vlm, device="cpu")
        # Ký tự đánh dấu không xuất hiện trong template, nên đếm được chính xác.
        marker = "\u00a7"
        prompt = parser.build_prompt(marker * (cfg.vlm.max_report_chars * 3))
        assert prompt.count(marker) == cfg.vlm.max_report_chars
