"""Fixture dùng chung cho test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from knee_mri.config import Config, load_config  # noqa: E402
from knee_mri.data.catalog import SeriesInfo, StudyCatalog  # noqa: E402


@pytest.fixture(scope="session")
def session_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Cấu hình ``test`` dùng chung cả phiên, artifacts trong thư mục tạm.

    Volume tí hon (8×32×32) nên mọi test model chạy trong vài trăm mili giây.
    """
    artifacts = tmp_path_factory.mktemp("artifacts-session")
    return load_config("test", overrides={"paths": {"artifacts_root": str(artifacts)}})


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Cấu hình ``test`` **riêng cho từng test**, artifacts trong thư mục tạm.

    Cô lập theo từng test là cần thiết cho các bước pipeline có ghi cache: nếu
    dùng chung thư mục, test kiểm tra tính idempotent sẽ phụ thuộc thứ tự chạy.
    """
    return load_config("test", overrides={"paths": {"artifacts_root": str(tmp_path / "artifacts")}})


@pytest.fixture
def volume(cfg: Config) -> np.ndarray:
    """Volume giả đúng shape mục tiêu, có cấu trúc không gian rõ ràng."""
    depth, height, width = cfg.data.target_shape
    grid = np.indices((depth, height, width)).astype(np.float32)
    array = np.sin(grid[0] / 3) + np.cos(grid[1] / 5) + np.sin(grid[2] / 7)
    return (array / np.abs(array).max()).astype(np.float32)


@pytest.fixture
def catalog() -> StudyCatalog:
    """Catalog nhỏ dựng trực tiếp, không cần file CSV."""
    series = {
        "study-a": [
            SeriesInfo("series-a1", "Sagittal", 1, 1),
            SeriesInfo("series-a2", "Axial", 0, 0),
            SeriesInfo("series-a3", "Coronal", 1, 0),
        ],
        "study-b": [SeriesInfo("series-b1", "Axial", 0, 1)],
        "study-c": [],
    }
    reports = {
        "study-a": "Tear of the medial meniscus. Large joint effusion.",
        "study-b": "Sin derrame articular. Meniscos normales.",
        "study-c": "",
    }
    gold = {"study-a": {"Medial Meniscus": 1, "Effusion": 1, "ACL": 0}}
    return StudyCatalog(series, reports, gold, ["study-a", "study-b", "study-c"])


@pytest.fixture(scope="session")
def real_series_dir() -> Path:
    """Thư mục DICOM thật đầu tiên trong dataset, hoặc skip nếu không có."""
    root = PROJECT_ROOT / "dataset" / "train_series"
    if not root.is_dir():
        pytest.skip("Không có dataset cục bộ.")
    for study in sorted(root.iterdir()):
        for series in sorted(study.iterdir()):
            if any(series.glob("*.dcm")):
                return series
    pytest.skip("Không tìm thấy series DICOM nào.")
    raise AssertionError("unreachable")
