#!/usr/bin/env python3
"""Entrypoint mỏng cho `knee_mri.cli.precompute_teacher_feats`.

Chạy được cả hai cách::

    python scripts/precompute_teacher_feats.py --env local --limit 8
    knee-precompute-teacher-feats --env local --limit 8    # sau khi pip install -e .
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee_mri.cli import precompute_teacher_feats  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(precompute_teacher_feats())
