"""Test nạp cấu hình — khóa lại lỗi P0-9 (ghi đè bị nuốt im lặng)."""

from __future__ import annotations

from pathlib import Path

import pytest

from knee_mri.config import Config, ConfigError, load_config, parse_overrides


class TestLoading:
    @pytest.mark.parametrize("env", ["local", "colab", "test"])
    def test_every_env_loads(self, env: str) -> None:
        assert load_config(env).env == env

    def test_unknown_env_raises(self) -> None:
        with pytest.raises(ConfigError, match="Không tìm thấy cấu hình"):
            load_config("moi_truong_khong_ton_tai")

    def test_env_file_overrides_base(self) -> None:
        """``env.test.yaml`` thu nhỏ volume so với ``base.yaml``."""
        assert load_config("test").data.max_slices < load_config("colab").data.max_slices


class TestOverridesActuallyApply:
    """P0-9: bản cũ dùng hằng số module-level nên ghi đè không có tác dụng.

    ``from src.config import DATA_ROOT`` sao chép *giá trị* lúc import; gán lại
    ``cfg.DATA_ROOT`` sau đó không đổi được bản sao. Notebook Colab ghi đè 6
    đường dẫn theo cách này và tất cả đều bị nuốt im lặng.
    """

    def test_path_override_takes_effect(self) -> None:
        cfg = load_config("local", overrides={"paths": {"data_root": "/tmp/du-lieu-khac"}})
        assert cfg.paths.data_root == Path("/tmp/du-lieu-khac")

    def test_nested_override_preserves_siblings(self) -> None:
        """Ghi đè một khóa lồng nhau không được xóa các khóa anh em."""
        base = load_config("test")
        cfg = load_config("test", overrides={"train": {"epochs": 99}})
        assert cfg.train.epochs == 99
        assert cfg.train.batch_size == base.train.batch_size

    def test_derived_paths_follow_override(self) -> None:
        """Mọi đường dẫn con được suy ra động, nên bám theo giá trị đã ghi đè."""
        cfg = load_config("local", overrides={"paths": {"artifacts_root": "/tmp/kq"}})
        assert cfg.paths.masks == Path("/tmp/kq/masks")
        assert cfg.paths.checkpoints == Path("/tmp/kq/checkpoints")


class TestParseOverrides:
    def test_builds_nested_dict(self) -> None:
        assert parse_overrides(["train.epochs=5"]) == {"train": {"epochs": 5}}

    def test_infers_types_via_yaml(self) -> None:
        parsed = parse_overrides(["a.b=true", "a.c=1.5", "a.d=[1, 2]", "a.e=text"])["a"]
        assert parsed == {"b": True, "c": 1.5, "d": [1, 2], "e": "text"}

    def test_merges_multiple_keys_under_same_parent(self) -> None:
        parsed = parse_overrides(["train.epochs=5", "train.lr=0.01"])
        assert parsed == {"train": {"epochs": 5, "lr": 0.01}}

    def test_rejects_malformed_item(self) -> None:
        with pytest.raises(ConfigError, match="khóa=giá_trị"):
            parse_overrides(["thieu_dau_bang"])

    def test_empty_input(self) -> None:
        assert parse_overrides(None) == {}


class TestImmutability:
    def test_config_is_frozen(self, cfg: Config) -> None:
        """Frozen dataclass chặn việc sửa nhầm cấu hình giữa chừng."""
        with pytest.raises((AttributeError, TypeError)):
            cfg.seed = 1  # type: ignore[misc]


class TestPaths:
    def test_target_shape_combines_slices_and_size(self, cfg: Config) -> None:
        assert cfg.data.target_shape == (cfg.data.max_slices, *cfg.data.volume_size)

    def test_ensure_creates_directories(self, tmp_path: Path) -> None:
        """Thư mục chỉ được tạo khi gọi ``ensure()``, không phải lúc import (P3-2)."""
        cfg = load_config("test", overrides={"paths": {"artifacts_root": str(tmp_path / "kq")}})
        assert not cfg.paths.masks.exists()
        cfg.paths.ensure()
        assert cfg.paths.masks.is_dir()
        assert cfg.paths.weak_labels.is_dir()

    def test_series_dir_returns_none_when_missing(self, cfg: Config) -> None:
        assert cfg.paths.series_dir("khong-co", "khong-co") is None
