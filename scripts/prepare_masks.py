#!/usr/bin/env python3
"""Entrypoint mỏng cho `knee_mri.cli.prepare_masks`.

Chạy được cả hai cách::

    python scripts/prepare_masks.py --env local --limit 8
    knee-prepare-masks --env local --limit 8    # sau khi pip install -e .
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee_mri.cli import prepare_masks  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(prepare_masks())
