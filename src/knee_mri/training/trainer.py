"""Vòng huấn luyện student bằng text-guided knowledge distillation.

Student (ViT-3D, **chỉ ảnh**) học đồng thời bốn mục tiêu:

1. Bắt chước đặc trưng ảnh của 3DINO-ViT (KD).
2. Căn chỉnh với guidance text của VLM qua InfoNCE — đây là cách tri thức trong
   report được chuyển vào một model không bao giờ đọc report.
3. Phân loại 12 nhãn bằng Asymmetric Loss trên nhãn yếu.
4. Tối ưu thứ hạng bằng AUC-margin.

Những gì bản cũ thiếu và nay đã có: mixed precision, gradient clipping, LR
scheduler có warmup, vòng validation đo AUC trên 58 study nhãn gold, theo dõi
checkpoint tốt nhất, và ``num_workers`` lấy từ cấu hình (lỗi P1-4, P1-5, P1-7, P1-8).
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from knee_mri.config.schema import Config
from knee_mri.data.collate import KneeBatch, collate_studies, pool_per_study
from knee_mri.data.dataset import KneeDataset
from knee_mri.models.heads import ClassifierHead, ContrastiveProjector, KDProjector
from knee_mri.models.student import ViT3DStudent
from knee_mri.training.checkpoint import (
    BEST_NAME,
    LAST_NAME,
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from knee_mri.training.losses import asymmetric_loss, auc_margin_loss, info_nce, kd_cosine_loss
from knee_mri.training.metrics import AucReport, format_auc_report, macro_auc
from knee_mri.utils.logging import get_logger

logger = get_logger(__name__)

_AMP_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class EpochStats:
    """Tổng hợp loss của một epoch."""

    total: float = 0.0
    kd: float = 0.0
    contrast: float = 0.0
    cls: float = 0.0
    auc: float = 0.0
    steps: int = 0

    def update(self, **values: float) -> None:
        for name, value in values.items():
            setattr(self, name, getattr(self, name) + value)
        self.steps += 1

    def averaged(self) -> dict[str, float]:
        divisor = max(self.steps, 1)
        return {
            name: getattr(self, name) / divisor
            for name in ("total", "kd", "contrast", "cls", "auc")
        }

    def format(self) -> str:
        values = self.averaged()
        return " ".join(f"{name}={value:.4f}" for name, value in values.items())


@dataclass
class TrainingBundle:
    """Toàn bộ module có tham số học được của một lần huấn luyện."""

    student: ViT3DStudent
    image_projector: KDProjector
    image_contrast: ContrastiveProjector
    text_contrast: ContrastiveProjector
    classifier: ClassifierHead

    def as_dict(self) -> dict[str, nn.Module]:
        return {
            "student": self.student,
            "image_projector": self.image_projector,
            "image_contrast": self.image_contrast,
            "text_contrast": self.text_contrast,
            "classifier": self.classifier,
        }

    def parameters(self) -> Iterator[nn.Parameter]:
        for module in self.as_dict().values():
            yield from module.parameters()

    def to(self, device: str | torch.device) -> TrainingBundle:
        for module in self.as_dict().values():
            module.to(device)
        return self

    def train(self) -> None:
        for module in self.as_dict().values():
            module.train()

    def eval(self) -> None:
        for module in self.as_dict().values():
            module.eval()


def build_bundle(cfg: Config) -> TrainingBundle:
    """Khởi tạo mọi module học được từ cấu hình.

    Số chiều guidance lấy thẳng từ cấu hình. Bản cũ lấy nó bằng cách gọi
    ``next(iter(loader))`` — tiêu tốn nguyên một batch (đọc DICOM, dựng volume)
    chỉ để đọc một con số (lỗi P1-4).
    """
    return TrainingBundle(
        student=ViT3DStudent(cfg),
        image_projector=KDProjector(cfg.model.student_dim, cfg.model.teacher_dim),
        image_contrast=ContrastiveProjector(cfg.model.student_dim, cfg.model.contrast_dim),
        text_contrast=ContrastiveProjector(cfg.model.guidance_dim, cfg.model.contrast_dim),
        classifier=ClassifierHead(cfg),
    )


def build_optimizer(bundle: TrainingBundle, cfg: Config) -> torch.optim.AdamW:
    """AdamW với weight decay **không** áp lên bias và tham số chuẩn hóa.

    Áp decay lên bias/LayerNorm là sai về mặt lý thuyết (chúng không phải trọng
    số kết nối) và thực nghiệm cho thấy làm chậm hội tụ của ViT.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for module in bundle.as_dict().values():
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith(".bias") or "cls_token" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config, *, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup tuyến tính rồi cosine decay.

    Warmup rất quan trọng với ViT: những bước đầu tiên có gradient lớn và nhiễu,
    lao thẳng vào learning rate đầy đủ thường làm hỏng attention ngay từ đầu.
    """
    total_steps = max(1, steps_per_epoch * cfg.train.epochs)
    warmup_steps = max(1, int(total_steps * cfg.train.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_loader(dataset: KneeDataset, cfg: Config, *, shuffle: bool) -> DataLoader:
    """DataLoader dùng ``num_workers`` từ cấu hình (bản cũ hardcode 0 — lỗi P1-5)."""
    workers = cfg.train.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_studies,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _autocast(cfg: Config, device: str):
    """Context mixed precision, hoặc no-op khi tắt/không có CUDA."""
    dtype = _AMP_DTYPES.get(cfg.train.amp_dtype)
    enabled = dtype is not None and device.startswith("cuda")
    return torch.autocast(device_type="cuda" if enabled else "cpu", dtype=dtype, enabled=enabled)


def compute_losses(
    bundle: TrainingBundle, batch: KneeBatch, cfg: Config, device: str
) -> tuple[Tensor, dict[str, float]]:
    """Tính tổng loss và các thành phần cho một batch."""
    volumes = batch["volumes"].to(device, non_blocking=True)
    teacher_feats = batch["teacher_feats"].to(device, non_blocking=True)
    guidance = batch["guidance"].to(device, non_blocking=True)
    metadata = batch["metadata"].to(device, non_blocking=True)
    weak_labels = batch["weak_labels"].to(device, non_blocking=True)
    counts = batch["counts"]

    features, _ = bundle.student(volumes)

    loss_kd = kd_cosine_loss(bundle.image_projector(features), teacher_feats)

    study_features = pool_per_study(features, counts)
    image_embedding = bundle.image_contrast(study_features)
    text_embedding = bundle.text_contrast(guidance)
    loss_contrast = info_nce(image_embedding, text_embedding, temperature=cfg.train.temp)

    logits = bundle.classifier(features, metadata)
    study_logits = pool_per_study(logits, counts)
    loss_cls = asymmetric_loss(
        study_logits,
        weak_labels,
        gamma_neg=cfg.train.asl.gamma_neg,
        gamma_pos=cfg.train.asl.gamma_pos,
        clip=cfg.train.asl.clip,
    )
    loss_auc = auc_margin_loss(study_logits, weak_labels, margin=cfg.train.auc_margin)

    weights = cfg.train.loss
    total = (
        weights.kd * loss_kd
        + weights.contrast * loss_contrast
        + weights.cls * loss_cls
        + weights.auc * loss_auc
    )
    components = {
        "total": float(total.detach()),
        "kd": float(loss_kd.detach()),
        "contrast": float(loss_contrast.detach()),
        "cls": float(loss_cls.detach()),
        "auc": float(loss_auc.detach()),
    }
    return total, components


@torch.no_grad()
def evaluate(
    bundle: TrainingBundle, loader: DataLoader, cfg: Config, device: str
) -> AucReport | None:
    """Đo AUC macro trên tập validation, chỉ dùng study có nhãn gold.

    Trả về ``None`` nếu không có study nào có nhãn gold — khi đó không có gì để
    đo và ta nói rõ điều đó thay vì báo một con số vô nghĩa.
    """
    bundle.eval()
    scores: list[np.ndarray] = []
    truths: list[np.ndarray] = []

    for batch in loader:
        gold = batch["gold_labels"]
        if gold is None:
            continue
        volumes = batch["volumes"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        with _autocast(cfg, device):
            features, _ = bundle.student(volumes)
            logits = bundle.classifier(features, metadata)
        probabilities = torch.sigmoid(pool_per_study(logits.float(), batch["counts"]))

        mask = batch["gold_mask"]
        scores.append(probabilities[mask].cpu().numpy())
        truths.append(gold[mask].cpu().numpy())

    if not scores:
        return None
    return macro_auc(np.concatenate(truths), np.concatenate(scores))


@dataclass
class TrainResult:
    """Kết quả trả về sau khi huấn luyện xong."""

    bundle: TrainingBundle
    best_metric: float
    best_checkpoint: Path
    history: list[dict[str, float]] = field(default_factory=list)


def train(
    cfg: Config,
    train_dataset: KneeDataset,
    val_dataset: KneeDataset | None = None,
    *,
    device: str | None = None,
    resume: bool = False,
) -> TrainResult:
    """Huấn luyện student. Trả về :class:`TrainResult` với checkpoint tốt nhất."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.paths.ensure()

    bundle = build_bundle(cfg).to(device)
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    val_loader = make_loader(val_dataset, cfg, shuffle=False) if val_dataset else None

    optimizer = build_optimizer(bundle, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp_dtype == "float16")

    start_epoch = 0
    best_metric = float("-inf")
    last_path = cfg.paths.checkpoints / LAST_NAME
    best_path = cfg.paths.checkpoints / BEST_NAME
    if resume and last_path.is_file():
        start_epoch, best_metric = load_checkpoint(
            last_path,
            bundle.as_dict(),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )

    logger.info(
        "Bắt đầu huấn luyện trên %s | %d study train, %d study val | %d epoch",
        device,
        len(train_dataset),
        len(val_dataset) if val_dataset else 0,
        cfg.train.epochs,
    )

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, cfg.train.epochs):
        bundle.train()
        stats = EpochStats()
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, device):
                loss, components = compute_losses(bundle, batch, cfg, device)

            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(bundle.parameters()), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            stats.update(**components)
            if step % cfg.train.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d | %s | lr=%.2e",
                    epoch,
                    step,
                    len(train_loader),
                    " ".join(f"{k}={v:.4f}" for k, v in components.items()),
                    scheduler.get_last_lr()[0],
                )

        elapsed = time.perf_counter() - epoch_start
        record = stats.averaged()
        record["epoch"] = float(epoch)
        record["seconds"] = elapsed
        logger.info("epoch %d xong sau %.1fs | %s", epoch, elapsed, stats.format())

        if val_loader is not None:
            report = evaluate(bundle, val_loader, cfg, device)
            if report is None:
                logger.warning("Tập validation không có study nào có nhãn gold — bỏ qua đo AUC.")
            else:
                record["val_macro_auc"] = report.macro
                logger.info("Validation:\n%s", format_auc_report(report))
                if report.macro > best_metric:
                    best_metric = report.macro
                    save_checkpoint(
                        CheckpointState(
                            epoch=epoch + 1,
                            best_metric=best_metric,
                            modules=bundle.as_dict(),
                            optimizer=optimizer,
                            scheduler=scheduler,
                        ),
                        best_path,
                    )
                    logger.info("→ checkpoint tốt nhất mới: macro AUC = %.4f", best_metric)

        history.append(record)
        save_checkpoint(
            CheckpointState(
                epoch=epoch + 1,
                best_metric=best_metric,
                modules=bundle.as_dict(),
                optimizer=optimizer,
                scheduler=scheduler,
            ),
            last_path,
        )

    if not best_path.is_file():
        # Không có validation: lấy checkpoint cuối làm "tốt nhất" để bước sau dùng được.
        save_checkpoint(
            CheckpointState(
                epoch=cfg.train.epochs, best_metric=best_metric, modules=bundle.as_dict()
            ),
            best_path,
        )

    return TrainResult(
        bundle=bundle, best_metric=best_metric, best_checkpoint=best_path, history=history
    )


__all__ = [
    "EpochStats",
    "TrainResult",
    "TrainingBundle",
    "build_bundle",
    "build_optimizer",
    "build_scheduler",
    "evaluate",
    "make_loader",
    "train",
]
