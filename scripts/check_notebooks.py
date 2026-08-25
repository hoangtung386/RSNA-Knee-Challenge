#!/usr/bin/env python3
"""Kiểm tra mọi cell code trong notebook parse được thành Python hợp lệ.

Notebook không được test bởi ``pytest`` nên rất dễ trôi khỏi code thật — đúng
điều đã xảy ra với bản trước refactor, nơi ba cell tham chiếu tới một API chưa
từng tồn tại (``from src.config import cfg``, ``cfg.OUT``, ``cfg.META_DIM``).

Script này là chốt chặn tối thiểu trong CI. Nó **không** chạy notebook (cần GPU
và dataset), chỉ xác nhận cú pháp.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

#: Dòng bắt đầu bằng các ký tự này là lệnh shell / magic của IPython, không phải
#: Python thuần. Thay bằng ``pass`` để giữ nguyên thụt lề khi parse.
IPYTHON_PREFIXES: tuple[str, ...] = ("!", "%")


def to_plain_python(source: str) -> str:
    """Đổi nguồn của một cell thành Python parse được."""
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(IPYTHON_PREFIXES):
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def check_notebook(path: Path) -> list[str]:
    """Trả về danh sách mô tả lỗi; rỗng nghĩa là notebook hợp lệ."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        try:
            ast.parse(to_plain_python("".join(cell["source"])))
        except SyntaxError as error:
            errors.append(f"{path}: cell {index}: {error}")
    return errors


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(p) for p in (argv or [])] or sorted(root.glob("notebooks/*.ipynb"))
    if not paths:
        print("Không tìm thấy notebook nào.")
        return 0

    all_errors: list[str] = []
    for path in paths:
        errors = check_notebook(path)
        all_errors.extend(errors)
        status = "OK" if not errors else f"{len(errors)} LỖI"
        cells = len(json.loads(path.read_text(encoding="utf-8"))["cells"])
        print(f"  {path.name:<36s} {cells:>2d} cell — {status}")

    for error in all_errors:
        print(f"    {error}", file=sys.stderr)
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
