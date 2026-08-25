"""Nạp cấu hình từ YAML thành :class:`~knee_mri.config.schema.Config`.

Thứ tự merge (sau đè trước)::

    configs/base.yaml  →  configs/env.<env>.yaml  →  overrides truyền vào

Ví dụ dùng trong notebook Colab — override có tác dụng thật vì ``cfg`` được
truyền tường minh xuống mọi hàm::

    cfg = load_config("colab", overrides={"paths": {"data_root": "/content/..."}})
    cfg.paths.ensure()
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from knee_mri.config.schema import (
    AslCfg,
    Config,
    DataCfg,
    DinoCfg,
    LossWeights,
    ModelCfg,
    Paths,
    SamCfg,
    TrainCfg,
    VlmCfg,
)

#: Thư mục gốc của project (chứa ``configs/``, ``src/``, ``dataset/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

#: Thư mục chứa các file YAML cấu hình.
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

#: Biến môi trường để ép môi trường mà không sửa code.
ENV_VAR: str = "KNEE_MRI_ENV"


class ConfigError(RuntimeError):
    """Cấu hình thiếu khóa, sai kiểu, hoặc file YAML không tồn tại."""


def detect_env() -> str:
    """Đoán môi trường đang chạy.

    Ưu tiên biến môi trường ``KNEE_MRI_ENV``; nếu không có thì phát hiện Colab
    qua điểm mount Drive; mặc định là ``local``.
    """
    forced = os.environ.get(ENV_VAR)
    if forced:
        return forced
    if Path("/content/drive").exists():
        return "colab"
    return "local"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Không tìm thấy file cấu hình: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"File cấu hình {path} phải là mapping ở cấp cao nhất.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge đệ quy: dict lồng nhau được hợp nhất, giá trị vô hướng bị ghi đè."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Thiếu khóa cấu hình bắt buộc: {context}.{key}")
    return mapping[key]


def _resolve_path(raw: str | Path) -> Path:
    """Đường dẫn tương đối được hiểu là tương đối với gốc project."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _build_paths(raw: dict[str, Any]) -> Paths:
    return Paths(
        data_root=_resolve_path(_require(raw, "data_root", "paths")),
        artifacts_root=_resolve_path(_require(raw, "artifacts_root", "paths")),
        dino_code_dir=_resolve_path(raw.get("dino_code_dir", "vendor/3DINO")),
    )


def _build_data(raw: dict[str, Any]) -> DataCfg:
    height, width = _require(raw, "volume_size", "data")
    return DataCfg(
        volume_size=(int(height), int(width)),
        max_slices=int(_require(raw, "max_slices", "data")),
        max_series=int(raw.get("max_series", 4)),
        norm_mode=str(_require(raw, "norm_mode", "data")),
        prefer_plane=str(raw.get("prefer_plane", "Sagittal")),
        prefer_fluid=int(raw.get("prefer_fluid", 1)),
        prefer_fatsupp=int(raw.get("prefer_fatsupp", 1)),
    )


def _build_model(raw: dict[str, Any]) -> ModelCfg:
    patch = tuple(int(v) for v in _require(raw, "patch_size", "model"))
    if len(patch) != 3:
        raise ConfigError(f"model.patch_size phải có 3 phần tử (D,H,W), nhận được {patch}.")
    patch_d, patch_h, patch_w = patch
    return ModelCfg(
        patch_size=(patch_d, patch_h, patch_w),
        student_dim=int(_require(raw, "student_dim", "model")),
        student_depth=int(raw.get("student_depth", 6)),
        student_heads=int(raw.get("student_heads", 8)),
        student_mlp_dim=int(raw.get("student_mlp_dim", 2048)),
        student_dropout=float(raw.get("student_dropout", 0.1)),
        teacher_dim=int(_require(raw, "teacher_dim", "model")),
        guidance_dim=int(_require(raw, "guidance_dim", "model")),
        contrast_dim=int(raw.get("contrast_dim", 512)),
        metadata_dim=int(raw.get("metadata_dim", 5)),
        use_metadata=bool(raw.get("use_metadata", True)),
    )


def _build_train(raw: dict[str, Any]) -> TrainCfg:
    loss_raw = raw.get("loss", {})
    asl_raw = raw.get("asl", {})
    return TrainCfg(
        batch_size=int(_require(raw, "batch_size", "train")),
        epochs=int(_require(raw, "epochs", "train")),
        lr=float(_require(raw, "lr", "train")),
        weight_decay=float(raw.get("weight_decay", 0.05)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.05)),
        grad_clip=float(raw.get("grad_clip", 1.0)),
        num_workers=int(raw.get("num_workers", 4)),
        amp_dtype=str(raw.get("amp_dtype", "bfloat16")).lower(),
        log_every=int(raw.get("log_every", 20)),
        loss=LossWeights(
            kd=float(loss_raw.get("kd", 1.0)),
            contrast=float(loss_raw.get("contrast", 1.0)),
            cls=float(loss_raw.get("cls", 1.0)),
            auc=float(loss_raw.get("auc", 0.5)),
        ),
        temp=float(raw.get("temp", 0.07)),
        asl=AslCfg(
            gamma_neg=float(asl_raw.get("gamma_neg", 4.0)),
            gamma_pos=float(asl_raw.get("gamma_pos", 1.0)),
            clip=float(asl_raw.get("clip", 0.05)),
        ),
        auc_margin=float(raw.get("auc_margin", 1.0)),
    )


def _build_vlm(raw: dict[str, Any]) -> VlmCfg:
    return VlmCfg(
        model_id=str(_require(raw, "model_id", "vlm")),
        dtype=str(raw.get("dtype", "bfloat16")),
        guidance_max_slices=int(raw.get("guidance_max_slices", 4)),
        max_report_chars=int(raw.get("max_report_chars", 4000)),
        max_new_tokens=int(raw.get("max_new_tokens", 256)),
    )


def _build_sam(raw: dict[str, Any]) -> SamCfg:
    return SamCfg(model_id=str(_require(raw, "model_id", "sam")))


def _build_dino(raw: dict[str, Any]) -> DinoCfg:
    return DinoCfg(
        repo_url=str(raw.get("repo_url", "https://github.com/AICONSlab/3DINO")),
        config_name=str(raw.get("config_name", "train/vit3d_highres")),
        weight_file=str(raw.get("weight_file", "3dino_vit_weights.pth")),
        input_size=int(raw.get("input_size", 112)),
    )


def load_config(
    env: str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
) -> Config:
    """Nạp và hợp nhất cấu hình.

    Args:
        env: Tên môi trường (``local`` / ``colab`` / ``test``). ``None`` = tự phát hiện.
        overrides: Dict lồng nhau ghi đè lên YAML, áp dụng sau cùng.
        config_dir: Thư mục chứa YAML, mặc định là ``<project>/configs``.

    Raises:
        ConfigError: khi thiếu file hoặc thiếu khóa bắt buộc.
    """
    env = env or detect_env()
    directory = config_dir or CONFIG_DIR

    raw = _read_yaml(directory / "base.yaml")
    env_file = directory / f"env.{env}.yaml"
    if env_file.is_file():
        raw = _deep_merge(raw, _read_yaml(env_file))
    elif env not in {"local", "colab", "test"}:
        raise ConfigError(f"Không tìm thấy cấu hình cho môi trường {env!r}: {env_file}")

    if overrides:
        raw = _deep_merge(raw, overrides)

    return Config(
        env=env,
        seed=int(raw.get("seed", 42)),
        paths=_build_paths(_require(raw, "paths", "root")),
        data=_build_data(_require(raw, "data", "root")),
        model=_build_model(_require(raw, "model", "root")),
        train=_build_train(_require(raw, "train", "root")),
        vlm=_build_vlm(_require(raw, "vlm", "root")),
        sam=_build_sam(_require(raw, "sam", "root")),
        dino=_build_dino(raw.get("dino", {})),
    )


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Chuyển các chuỗi ``a.b.c=value`` trên CLI thành dict lồng nhau.

    Giá trị được parse bằng YAML nên ``42``/``true``/``[1,2]`` ra đúng kiểu.

    >>> parse_overrides(["train.epochs=5", "data.max_slices=16"])
    {'train': {'epochs': 5}, 'data': {'max_slices': 16}}
    """
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"Override phải có dạng khóa=giá_trị, nhận được: {item!r}")
        dotted, _, raw_value = item.partition("=")
        node = result
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw_value)
    return result


__all__ = [
    "CONFIG_DIR",
    "PROJECT_ROOT",
    "ConfigError",
    "detect_env",
    "load_config",
    "parse_overrides",
]
