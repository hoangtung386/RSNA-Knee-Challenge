"""Test giao diện dòng lệnh — mặt tiếp xúc chính khi vận hành pipeline."""

from __future__ import annotations

import pytest

from knee_mri import cli


class TestCommandRegistry:
    def test_every_command_is_callable(self) -> None:
        assert cli.COMMANDS
        assert all(callable(func) for func in cli.COMMANDS.values())

    def test_every_command_is_documented(self) -> None:
        """Docstring hiện ra trong bảng trợ giúp, nên không được để trống."""
        for name, func in cli.COMMANDS.items():
            assert func.__doc__, f"lệnh {name!r} thiếu docstring"

    def test_registry_matches_pyproject_entrypoints(self) -> None:
        """Mỗi entrypoint trong ``pyproject.toml`` phải trỏ tới hàm có thật."""
        from pathlib import Path

        # tomllib chỉ có sẵn từ Python 3.11; repo hỗ trợ từ 3.10 nên bỏ qua êm.
        tomllib = pytest.importorskip("tomllib")

        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8")
        )
        for script, target in pyproject["project"]["scripts"].items():
            module, _, function = target.partition(":")
            assert module == "knee_mri.cli", f"{script} trỏ tới module lạ: {module}"
            assert hasattr(cli, function), f"{script} trỏ tới hàm không tồn tại: {function}"

    def test_help_is_printed_without_arguments(self, capsys: pytest.CaptureFixture) -> None:
        assert cli.main([]) == 0
        output = capsys.readouterr().out
        assert "Cách dùng" in output
        for name in cli.COMMANDS:
            assert name in output

    def test_unknown_command_exits_non_zero(self, capsys: pytest.CaptureFixture) -> None:
        assert cli.main(["lenh-khong-ton-tai"]) == 2
        capsys.readouterr()


class TestArgumentParsing:
    @pytest.mark.parametrize("command", sorted(cli.COMMANDS))
    def test_help_flag_exits_cleanly(self, command: str) -> None:
        """``--help`` phải in trợ giúp rồi thoát mã 0, không nạp model nặng."""
        with pytest.raises(SystemExit) as exit_info:
            cli.main([command, "--help"])
        assert exit_info.value.code == 0

    def test_set_flag_reaches_config(self) -> None:
        from knee_mri.config import load_config, parse_overrides

        overrides = parse_overrides(["train.epochs=7", "data.max_slices=16"])
        cfg = load_config("test", overrides=overrides)
        assert cfg.train.epochs == 7
        assert cfg.data.max_slices == 16


class TestEndToEndCommand:
    def test_weak_labels_command_writes_files(self, tmp_path, monkeypatch) -> None:
        """Chạy thật một lệnh nhẹ, đầu-cuối, qua đúng đường mà người dùng gõ."""
        from knee_mri.config import load_config

        probe = load_config("test")
        if not probe.paths.train_csv.is_file():
            pytest.skip("Không có dataset cục bộ.")

        artifacts = tmp_path / "artifacts"
        exit_code = cli.precompute_weak_labels(
            ["--env", "test", "--limit", "3", "--set", f"paths.artifacts_root={artifacts}"]
        )
        assert exit_code == 0
        assert len(list((artifacts / "weak_labels").glob("*.json"))) == 3
