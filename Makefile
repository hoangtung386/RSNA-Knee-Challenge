# Lệnh tắt cho công việc thường ngày. Chạy `make` để xem danh sách.
.DEFAULT_GOAL := help
.PHONY: help install lint format test test-fast cov notebooks check smoke clean

PYTHON ?= python
ENV    ?= local

help:  ## Hiện danh sách lệnh
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Cài package ở chế độ editable kèm dependency dev
	$(PYTHON) -m pip install -e ".[dev,viz,eval]"
	pre-commit install

lint:  ## Kiểm tra lint và định dạng (không sửa)
	ruff check src tests scripts
	ruff format --check src tests scripts

format:  ## Tự sửa lint và định dạng lại code
	ruff check --fix src tests scripts
	ruff format src tests scripts

test:  ## Chạy toàn bộ test
	pytest

test-fast:  ## Chạy test, bỏ qua các test chậm
	pytest -m "not slow"

cov:  ## Chạy test kèm báo cáo độ phủ
	pytest --cov=knee_mri --cov-report=term-missing

notebooks:  ## Kiểm tra cú pháp mọi cell code trong notebook
	$(PYTHON) scripts/check_notebooks.py

check: lint test notebooks  ## Chạy đủ bộ kiểm tra như CI

smoke:  ## Chạy thử pipeline đầu-cuối trên tập nhỏ (không cần GPU)
	$(PYTHON) scripts/prepare_masks.py          --env $(ENV) --limit 8
	$(PYTHON) scripts/precompute_weak_labels.py --env $(ENV) --limit 8
	$(PYTHON) scripts/train.py                  --env $(ENV) --limit 8 --smoke --set train.epochs=1
	$(PYTHON) scripts/explain.py                --env $(ENV) --limit 3
	$(PYTHON) scripts/predict.py                --env $(ENV)

clean:  ## Xóa file tạm và cache
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +
