#!/usr/bin/env python3
"""So sánh các phương án chuẩn hóa cường độ MRI trên dữ liệu thật.

MRI không có thang đo tuyệt đối, nên lựa chọn chuẩn hóa ảnh hưởng trực tiếp tới
chất lượng đặc trưng. Script đo ba chỉ số trên một mẫu series ngẫu nhiên:

* **Cross-series consistency** — độ lệch chuẩn của giá trị trung bình sau chuẩn
  hóa, tính xuyên các series. Càng nhỏ càng nhất quán giữa các scanner.
* **Flicker** — biến thiên giữa các lát trong cùng volume. Quá nhỏ nghĩa là đã
  *over-normalize*, xóa mất tín hiệu 3D thật.
* **Outlier bị cắt** — tỉ lệ voxel nằm ngoài khoảng percentile.

Toàn bộ phép chuẩn hóa lấy từ :mod:`knee_mri.data.normalize` — không còn bản sao
riêng như trước (lỗi P3-8), nên script và pipeline không thể lệch nhau.

Ví dụ::

    python scripts/analyze_normalization.py --max-series 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee_mri.config import load_config  # noqa: E402
from knee_mri.constants import COL_PLANE, COL_SERIES, COL_STUDY, TRAIN_SERIES_DIR  # noqa: E402
from knee_mri.data.dicom_io import EmptySeriesError, read_series  # noqa: E402
from knee_mri.data.normalize import (  # noqa: E402
    VOLUME_PERCENTILES,
    normalize_slice_percentile,
    normalize_volume_percentile,
    normalize_volume_zscore,
)
from knee_mri.utils.logging import setup_logging  # noqa: E402

#: Các phương án được so sánh.
CANDIDATES = {
    "slice_pctl": normalize_slice_percentile,
    "vol_pctl": lambda volume: normalize_volume_percentile(volume, to_pm1=False),
    "vol_pctl_pm1": lambda volume: normalize_volume_percentile(volume, to_pm1=True),
    "vol_zscore": normalize_volume_zscore,
}


def measure(raw: np.ndarray, normalised: np.ndarray) -> dict[str, float]:
    """Ba chỉ số đánh giá một phương án chuẩn hóa trên một volume."""
    slice_means = normalised.reshape(normalised.shape[0], -1).mean(axis=1)
    low, high = np.percentile(raw, VOLUME_PERCENTILES)
    return {
        "mean": float(normalised.mean()),
        "std": float(normalised.std()),
        "flicker": float(slice_means.std() / (abs(slice_means.mean()) + 1e-8)),
        "clipped": float(((raw < low) | (raw > high)).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default=None, help="Môi trường cấu hình.")
    parser.add_argument("--max-series", type=int, default=200, help="Số series lấy mẫu.")
    parser.add_argument("--max-slices", type=int, default=48, help="Số lát đọc mỗi series.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.env)
    series_df = pd.read_csv(cfg.paths.train_series_csv)
    sample = series_df.sample(min(args.max_series, len(series_df)), random_state=args.seed)

    records: list[dict[str, float | str]] = []
    for _, row in sample.iterrows():
        series_dir = cfg.paths.data_root / TRAIN_SERIES_DIR / row[COL_STUDY] / row[COL_SERIES]
        if not series_dir.is_dir():
            continue
        try:
            raw = read_series(series_dir, max_slices=args.max_slices)
        except EmptySeriesError:
            continue

        record: dict[str, float | str] = {"plane": row[COL_PLANE]}
        for name, transform in CANDIDATES.items():
            for metric, value in measure(raw, transform(raw)).items():
                record[f"{name}_{metric}"] = value
        records.append(record)

        if len(records) % 50 == 0:
            print(f"  ... đã phân tích {len(records)} series")

    if not records:
        print("Không đọc được series nào — kiểm tra lại đường dẫn dataset.")
        return 1

    frame = pd.DataFrame(records)
    print(f"\nTổng số series đã phân tích: {len(frame)}\n")

    print("== Cross-series consistency (std của mean; NHỎ hơn = nhất quán hơn) ==")
    print(frame[[f"{name}_mean" for name in CANDIDATES]].std().round(4).to_string())

    print("\n== Flicker (biến thiên giữa các lát; quá NHỎ = over-normalize) ==")
    print(frame[[f"{name}_flicker" for name in CANDIDATES]].mean().round(4).to_string())

    print("\n== Tỉ lệ voxel bị cắt ==")
    print(frame[[f"{name}_clipped" for name in CANDIDATES]].mean().round(5).to_string())

    print(
        "\n=> KHUYẾN NGHỊ: percentile theo volume → [-1, 1] (vol_pctl_pm1)\n"
        "   - nhất quán xuyên dataset, không phụ thuộc scanner;\n"
        "   - giữ được biến thiên 3D, không over-normalize như per-slice;\n"
        "   - khớp đúng định dạng đầu vào của 3DINO-ViT.\n"
        f"   Đang cấu hình: data.norm_mode = {cfg.data.norm_mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
