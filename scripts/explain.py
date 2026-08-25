#!/usr/bin/env python3
"""Entrypoint mỏng cho `knee_mri.cli.explain`.

Chạy được cả hai cách::

    python scripts/explain.py --env local --limit 8
    knee-explain --env local --limit 8    # sau khi pip install -e .
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee_mri.cli import explain  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(explain())
