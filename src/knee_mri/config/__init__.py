"""Cấu hình dạng dataclass bất biến, nạp từ YAML."""

from knee_mri.config.loader import (
    CONFIG_DIR,
    PROJECT_ROOT,
    ConfigError,
    detect_env,
    load_config,
    parse_overrides,
)
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

__all__ = [
    "CONFIG_DIR",
    "PROJECT_ROOT",
    "AslCfg",
    "Config",
    "ConfigError",
    "DataCfg",
    "DinoCfg",
    "LossWeights",
    "ModelCfg",
    "Paths",
    "SamCfg",
    "TrainCfg",
    "VlmCfg",
    "detect_env",
    "load_config",
    "parse_overrides",
]
