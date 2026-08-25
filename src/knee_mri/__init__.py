"""RSNA Knee MRI — phát hiện bất thường đa nhãn bằng text-guided knowledge distillation.

Package này được tổ chức theo trách nhiệm:

- :mod:`knee_mri.config`     — schema + loader cho cấu hình YAML
- :mod:`knee_mri.data`       — đọc DICOM, chuẩn hóa, index CSV, Dataset/collate
- :mod:`knee_mri.labeling`   — sinh weak label 12 chiều từ radiology report
- :mod:`knee_mri.models`     — student ViT-3D, các teacher, SAM, report generator
- :mod:`knee_mri.training`   — loss, metric, vòng huấn luyện, checkpoint
- :mod:`knee_mri.explain`    — CAM và đối chiếu với mask
- :mod:`knee_mri.inference`  — sinh submission từ ảnh + metadata
- :mod:`knee_mri.utils`      — IO, mask, logging, seed
"""

from knee_mri.constants import LABELS, METADATA_DIM, NUM_LABELS

__version__ = "0.1.0"

__all__ = ["LABELS", "METADATA_DIM", "NUM_LABELS", "__version__"]
