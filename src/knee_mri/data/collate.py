"""Ghép các study thành batch.

Một study có số series thay đổi, nên batch được làm phẳng theo series và kèm
mảng ``counts`` để tách ngược về từng study khi cần tổng hợp ở mức study.

Nhờ :func:`~knee_mri.data.volume.build_volume` bảo đảm mọi volume có cùng shape,
hàm ghép ở đây chỉ là ``np.stack`` — không còn phép pad động và không còn giả
định ngầm ``H = vols[0].shape[2]`` vốn làm ``torch.cat`` vỡ khi ROI crop cho ra
kích thước khác nhau (lỗi P0-6).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor

from knee_mri.data.dataset import StudySample


class KneeBatch(TypedDict):
    """Batch đã ghép. Chiều ``N`` là tổng số series trong batch."""

    volumes: Tensor  # (N, 1, D, H, W)
    teacher_feats: Tensor  # (N, teacher_dim)
    metadata: Tensor  # (N, metadata_dim)
    guidance: Tensor  # (B, guidance_dim)
    weak_labels: Tensor  # (B, num_labels)
    gold_labels: Tensor | None  # (B, num_labels) hoặc None
    gold_mask: Tensor  # (B,) bool — study nào có nhãn gold thật
    counts: list[int]  # số series của từng study
    study_uids: list[str]


def collate_studies(batch: Sequence[StudySample]) -> KneeBatch:
    """Ghép danh sách study thành một :class:`KneeBatch`."""
    volumes = np.stack([volume for sample in batch for volume in sample["volumes"]])
    teacher_feats = np.stack([feat for sample in batch for feat in sample["teacher_feats"]])
    metadata = np.stack(
        [np.asarray(meta, dtype=np.float32) for sample in batch for meta in sample["metadata"]]
    )

    guidance = np.stack([sample["guidance"] for sample in batch])
    weak_labels = np.stack([sample["weak_label"] for sample in batch])

    gold_present = [sample["gold_label"] is not None for sample in batch]
    gold_labels: Tensor | None = None
    if any(gold_present):
        filler = np.zeros_like(weak_labels[0])
        gold_labels = torch.from_numpy(
            np.stack(
                [
                    sample["gold_label"] if sample["gold_label"] is not None else filler
                    for sample in batch
                ]
            )
        )

    return KneeBatch(
        volumes=torch.from_numpy(volumes).unsqueeze(1),
        teacher_feats=torch.from_numpy(teacher_feats),
        metadata=torch.from_numpy(metadata),
        guidance=torch.from_numpy(guidance),
        weak_labels=torch.from_numpy(weak_labels),
        gold_labels=gold_labels,
        gold_mask=torch.tensor(gold_present, dtype=torch.bool),
        counts=[len(sample["volumes"]) for sample in batch],
        study_uids=[sample["study_uid"] for sample in batch],
    )


def split_per_study(tensor: Tensor, counts: Sequence[int]) -> list[Tensor]:
    """Tách tensor phẳng theo series về danh sách theo từng study."""
    chunks: list[Tensor] = []
    offset = 0
    for count in counts:
        chunks.append(tensor[offset : offset + count])
        offset += count
    return chunks


def pool_per_study(tensor: Tensor, counts: Sequence[int]) -> Tensor:
    """Gộp các series của cùng một study bằng trung bình.

    Dự đoán được chấm ở mức study, nên mọi loss ở mức study (contrastive, phân
    loại, AUC) đều phải gộp trước khi so với nhãn.
    """
    return torch.stack([chunk.mean(dim=0) for chunk in split_per_study(tensor, counts)])


__all__ = ["KneeBatch", "collate_studies", "pool_per_study", "split_per_study"]
