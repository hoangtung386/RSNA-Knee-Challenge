"""Schema cấu hình dạng dataclass bất biến.

Toàn bộ cấu hình được nạp một lần thành object :class:`Config` rồi **truyền
tường minh** xuống các hàm. Đây là điểm mấu chốt sửa lỗi P0-9 của bản cũ: khi
config là hằng số module-level, ``from module import CONSTANT`` sao chép giá trị
lúc import và mọi ghi đè sau đó (ví dụ trong notebook Colab) bị nuốt im lặng.

Các dataclass đều ``frozen=True`` nên không thể sửa nhầm sau khi tạo; muốn đổi
giá trị phải dùng :func:`dataclasses.replace` một cách có chủ đích.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knee_mri.constants import (
    TEST_CSV,
    TEST_SERIES_CSV,
    TEST_SERIES_DIR,
    TRAIN_CSV,
    TRAIN_SERIES_CSV,
    TRAIN_SERIES_DIR,
)


@dataclass(frozen=True, slots=True)
class Paths:
    """Đường dẫn dataset và artifacts. Mọi đường dẫn con là property dẫn xuất."""

    data_root: Path
    artifacts_root: Path
    dino_code_dir: Path

    # ---- dataset ----
    @property
    def train_csv(self) -> Path:
        return self.data_root / TRAIN_CSV

    @property
    def train_series_csv(self) -> Path:
        return self.data_root / TRAIN_SERIES_CSV

    @property
    def test_csv(self) -> Path:
        return self.data_root / TEST_CSV

    @property
    def test_series_csv(self) -> Path:
        return self.data_root / TEST_SERIES_CSV

    # ---- artifacts ----
    @property
    def masks(self) -> Path:
        return self.artifacts_root / "masks"

    @property
    def checkpoints(self) -> Path:
        return self.artifacts_root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.artifacts_root / "logs"

    @property
    def cams(self) -> Path:
        return self.artifacts_root / "cams"

    @property
    def reports(self) -> Path:
        return self.artifacts_root / "reports"

    @property
    def guidance(self) -> Path:
        return self.artifacts_root / "guidance"

    @property
    def teacher_feats(self) -> Path:
        return self.artifacts_root / "teacher_feats"

    @property
    def weak_labels(self) -> Path:
        return self.artifacts_root / "weak_labels"

    @property
    def predictions(self) -> Path:
        return self.artifacts_root / "predictions"

    def series_dir(self, study_uid: str, series_uid: str) -> Path | None:
        """Tìm thư mục chứa DICOM của một series, thử cả train lẫn test.

        Trả về ``None`` nếu không tồn tại ở cả hai nơi.
        """
        for sub in (TRAIN_SERIES_DIR, TEST_SERIES_DIR):
            candidate = self.data_root / sub / study_uid / series_uid
            if candidate.is_dir():
                return candidate
        return None

    def ensure(self) -> None:
        """Tạo các thư mục artifacts.

        Gọi **tường minh** từ entrypoint, không phải như side effect lúc import
        (lỗi P3-2 của bản cũ khiến mọi lần chạy test đều tạo thư mục rác).
        """
        for directory in (
            self.masks,
            self.checkpoints,
            self.logs,
            self.cams,
            self.reports,
            self.guidance,
            self.teacher_feats,
            self.weak_labels,
            self.predictions,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class DataCfg:
    """Tham số dựng volume và chọn series."""

    volume_size: tuple[int, int]
    max_slices: int
    max_series: int
    norm_mode: str
    prefer_plane: str
    prefer_fluid: int
    prefer_fatsupp: int

    @property
    def target_shape(self) -> tuple[int, int, int]:
        """Shape ``(D, H, W)`` mà **mọi** volume phải có sau khi dựng.

        Đây là bất biến quan trọng nhất của pipeline: nó cố định số patch của
        ViT-3D, cho phép collate đơn giản bằng ``np.stack`` và loại bỏ hoàn toàn
        lớp lỗi P0-5/P0-6 của bản cũ.
        """
        return (self.max_slices, *self.volume_size)


@dataclass(frozen=True, slots=True)
class ModelCfg:
    """Kích thước các thành phần model."""

    patch_size: tuple[int, int, int]
    student_dim: int
    student_depth: int
    student_heads: int
    student_mlp_dim: int
    student_dropout: float
    teacher_dim: int
    guidance_dim: int
    contrast_dim: int
    metadata_dim: int
    use_metadata: bool


@dataclass(frozen=True, slots=True)
class AslCfg:
    gamma_neg: float
    gamma_pos: float
    clip: float


@dataclass(frozen=True, slots=True)
class LossWeights:
    kd: float
    contrast: float
    cls: float
    auc: float


@dataclass(frozen=True, slots=True)
class TrainCfg:
    """Tham số vòng huấn luyện."""

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    grad_clip: float
    num_workers: int
    amp_dtype: str
    log_every: int
    loss: LossWeights
    temp: float
    asl: AslCfg
    auc_margin: float


@dataclass(frozen=True, slots=True)
class VlmCfg:
    """Gemma multimodal — dùng cho weak label, guidance và sinh report."""

    model_id: str
    dtype: str
    guidance_max_slices: int
    max_report_chars: int
    max_new_tokens: int


@dataclass(frozen=True, slots=True)
class SamCfg:
    model_id: str


@dataclass(frozen=True, slots=True)
class DinoCfg:
    """3DINO-ViT — teacher ảnh, license CC BY-NC-ND (không sửa code vendor)."""

    repo_url: str
    config_name: str
    weight_file: str
    input_size: int

    def weights_path(self, artifacts_root: Path) -> Path:
        """Đường dẫn file weights, dựng từ ``weight_file`` trong cấu hình.

        Trước đây đây là property cứng trên :class:`Paths` với tên file viết
        thẳng trong code, nên ``dino.weight_file`` trong YAML bị nuốt im lặng —
        đúng loại lỗi mà đợt refactor phải loại bỏ.
        """
        return artifacts_root / self.weight_file


@dataclass(frozen=True, slots=True)
class Config:
    """Cấu hình đầy đủ của một lần chạy."""

    env: str
    seed: int
    paths: Paths
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    vlm: VlmCfg
    sam: SamCfg
    dino: DinoCfg


__all__ = [
    "AslCfg",
    "Config",
    "DataCfg",
    "DinoCfg",
    "LossWeights",
    "ModelCfg",
    "Paths",
    "SamCfg",
    "TrainCfg",
    "VlmCfg",
]
