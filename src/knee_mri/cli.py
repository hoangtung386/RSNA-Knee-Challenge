"""Giao diện dòng lệnh cho mọi bước của pipeline.

Mỗi bước là một entrypoint riêng (khai báo trong ``pyproject.toml``) và cũng gọi
được qua ``python -m knee_mri.cli <lệnh>``. Tất cả dùng chung bộ tham số cơ bản
nên cách chạy nhất quán::

    knee-train --env colab --limit 200 --set train.epochs=10

Thay cho ``src/pipeline.py`` cũ vốn nhồi mọi bước vào một ``argparse`` duy nhất
với các cờ đan xen khó hiểu.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from knee_mri.config import Config, load_config, parse_overrides
from knee_mri.data.catalog import StudyCatalog
from knee_mri.utils.logging import get_logger, setup_logging
from knee_mri.utils.seed import set_seed

logger = get_logger(__name__)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Thêm các tham số dùng chung cho mọi lệnh."""
    parser.add_argument(
        "--env",
        default=None,
        help="Môi trường cấu hình (local/colab/test). Mặc định: tự phát hiện.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KHOA=GIA_TRI",
        help="Ghi đè cấu hình, ví dụ --set train.epochs=10 (lặp lại được).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số study (smoke test).")
    parser.add_argument("--device", default=None, help="cuda / cpu. Mặc định: tự phát hiện.")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè kết quả đã cache.")
    parser.add_argument("--log-level", default="INFO", help="DEBUG / INFO / WARNING / ERROR.")


def prepare(args: argparse.Namespace) -> tuple[Config, StudyCatalog, list[str]]:
    """Nạp cấu hình, khởi tạo log/seed và dựng catalog tập train."""
    cfg = load_config(args.env, overrides=parse_overrides(args.overrides))
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO), log_dir=cfg.paths.logs)
    set_seed(cfg.seed)
    cfg.paths.ensure()
    logger.info("Môi trường=%s | data=%s", cfg.env, cfg.paths.data_root)

    catalog = StudyCatalog.from_csv(cfg.paths.train_csv, cfg.paths.train_series_csv)
    study_uids = catalog.studies_with_series()
    if args.limit is not None:
        study_uids = study_uids[: args.limit]
    return cfg, catalog, study_uids


def _resolve_device(args: argparse.Namespace) -> str:
    if args.device:
        return args.device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------- lệnh
def prepare_masks(argv: Sequence[str] | None = None) -> int:
    """S1 — sinh mask ROI bằng SAM."""
    parser = argparse.ArgumentParser(description=prepare_masks.__doc__)
    add_common_arguments(parser)
    args = parser.parse_args(argv)

    from knee_mri.pipeline import generate_masks

    cfg, catalog, study_uids = prepare(args)
    generate_masks(cfg, catalog, study_uids, device=_resolve_device(args), overwrite=args.overwrite)
    return 0


def precompute_weak_labels(argv: Sequence[str] | None = None) -> int:
    """Sinh nhãn yếu 12 chiều từ report (luật từ khóa hoặc VLM)."""
    parser = argparse.ArgumentParser(description=precompute_weak_labels.__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--use-vlm", action="store_true", help="Dùng VLM thay vì luật từ khóa (cần GPU)."
    )
    args = parser.parse_args(argv)

    from knee_mri.pipeline import precompute_weak_labels as run

    cfg, catalog, study_uids = prepare(args)
    run(
        cfg,
        catalog,
        study_uids,
        use_vlm=args.use_vlm,
        device=_resolve_device(args),
        overwrite=args.overwrite,
    )
    return 0


def precompute_teacher_feats(argv: Sequence[str] | None = None) -> int:
    """Trích và cache đặc trưng ảnh từ 3DINO-ViT."""
    parser = argparse.ArgumentParser(description=precompute_teacher_feats.__doc__)
    add_common_arguments(parser)
    args = parser.parse_args(argv)

    from knee_mri.pipeline import precompute_teacher_features

    cfg, catalog, study_uids = prepare(args)
    precompute_teacher_features(
        cfg, catalog, study_uids, device=_resolve_device(args), overwrite=args.overwrite
    )
    return 0


def precompute_guidance(argv: Sequence[str] | None = None) -> int:
    """Trích và cache vector guidance (ảnh + report → G) từ VLM."""
    parser = argparse.ArgumentParser(description=precompute_guidance.__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--allow-zero-fallback",
        action="store_true",
        help="Ghi vector 0 khi VLM lỗi (chỉ dùng cho smoke test).",
    )
    args = parser.parse_args(argv)

    from knee_mri.pipeline import precompute_guidance as run

    cfg, catalog, study_uids = prepare(args)
    run(
        cfg,
        catalog,
        study_uids,
        device=_resolve_device(args),
        overwrite=args.overwrite,
        allow_zero_fallback=args.allow_zero_fallback,
    )
    return 0


def train(argv: Sequence[str] | None = None) -> int:
    """S2 — huấn luyện student bằng text-guided knowledge distillation."""
    parser = argparse.ArgumentParser(description=train.__doc__)
    add_common_arguments(parser)
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint gần nhất.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Chạy thử với teacher/guidance giả, không cần precompute.",
    )
    parser.add_argument(
        "--extra-val-ratio",
        type=float,
        default=0.0,
        help="Tỉ lệ study không có nhãn gold đưa thêm vào validation.",
    )
    args = parser.parse_args(argv)

    from knee_mri.data.dataset import KneeDataset, describe_cache
    from knee_mri.data.splits import make_split
    from knee_mri.training.trainer import train as run_training

    cfg, catalog, study_uids = prepare(args)
    logger.info("Trạng thái cache: %s", describe_cache(cfg, study_uids))

    split = make_split(
        catalog,
        seed=cfg.seed,
        extra_val_ratio=args.extra_val_ratio,
        limit=args.limit,
    )
    train_dataset = KneeDataset(
        split.train,
        catalog,
        cfg,
        use_dummy_teacher=args.smoke,
        use_dummy_guidance=args.smoke,
    )
    val_dataset = (
        KneeDataset(
            split.val, catalog, cfg, use_dummy_teacher=args.smoke, use_dummy_guidance=args.smoke
        )
        if split.val
        else None
    )

    result = run_training(
        cfg, train_dataset, val_dataset, device=_resolve_device(args), resume=args.resume
    )
    logger.info(
        "Huấn luyện xong. Macro AUC tốt nhất = %.4f | checkpoint: %s",
        result.best_metric,
        result.best_checkpoint,
    )
    return 0


def evaluate(argv: Sequence[str] | None = None) -> int:
    """Đo AUC macro của một checkpoint trên các study có nhãn gold."""
    parser = argparse.ArgumentParser(description=evaluate.__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Đường dẫn checkpoint.")
    args = parser.parse_args(argv)

    import numpy as np

    from knee_mri.inference.predictor import Predictor
    from knee_mri.training.metrics import format_auc_report, macro_auc

    cfg, catalog, _ = prepare(args)
    gold_uids = catalog.studies_with_gold()
    if args.limit is not None:
        gold_uids = gold_uids[: args.limit]
    if not gold_uids:
        logger.error("Không có study nào có nhãn gold để đánh giá.")
        return 1

    uids, truth_rows = catalog.gold_matrix(gold_uids)
    predictor = Predictor.from_checkpoint(
        cfg, checkpoint=args.checkpoint, device=_resolve_device(args)
    )
    scores = predictor.predict_many(catalog, uids)
    report = macro_auc(np.array(truth_rows, dtype=np.float32), scores)
    logger.info("Đánh giá trên %d study có nhãn gold:\n%s", len(uids), format_auc_report(report))
    return 0


def predict(argv: Sequence[str] | None = None) -> int:
    """Sinh submission.csv từ tập test (chỉ ảnh + metadata)."""
    parser = argparse.ArgumentParser(description=predict.__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Đường dẫn checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Nơi ghi submission.csv.")
    args = parser.parse_args(argv)

    from knee_mri.inference.predictor import predict_submission

    cfg = load_config(args.env, overrides=parse_overrides(args.overrides))
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO), log_dir=cfg.paths.logs)
    set_seed(cfg.seed)
    cfg.paths.ensure()

    destination = predict_submission(
        cfg,
        checkpoint=args.checkpoint,
        output=args.output,
        device=_resolve_device(args),
        limit=args.limit,
    )
    logger.info("Submission: %s", destination)
    return 0


def explain(argv: Sequence[str] | None = None) -> int:
    """S3 — sinh CAM, đối chiếu với mask ROI và viết báo cáo."""
    parser = argparse.ArgumentParser(description=explain.__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Đường dẫn checkpoint.")
    parser.add_argument("--use-vlm", action="store_true", help="Sinh báo cáo bằng VLM.")
    args = parser.parse_args(argv)

    from knee_mri.pipeline import explain_studies

    cfg, catalog, study_uids = prepare(args)
    explain_studies(
        cfg,
        catalog,
        study_uids,
        checkpoint=args.checkpoint,
        device=_resolve_device(args),
        use_vlm=args.use_vlm,
    )
    return 0


#: Kiểu của một hàm lệnh: nhận danh sách tham số, trả mã thoát.
Command = Callable[[Sequence[str] | None], int]

COMMANDS: dict[str, Command] = {
    "masks": prepare_masks,
    "weak-labels": precompute_weak_labels,
    "teacher-feats": precompute_teacher_feats,
    "guidance": precompute_guidance,
    "train": train,
    "evaluate": evaluate,
    "predict": predict,
    "explain": explain,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Điều phối ``python -m knee_mri.cli <lệnh> [tham số...]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        available = "\n".join(f"  {name:<15s} {func.__doc__}" for name, func in COMMANDS.items())
        print(f"Cách dùng: python -m knee_mri.cli <lệnh> [tham số...]\n\nCác lệnh:\n{available}")
        return 0 if not argv else 2
    command = COMMANDS[argv[0]]
    return command(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
