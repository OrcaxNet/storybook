"""FLO-209 / FLO-210：`book version` 子命令测试。

验收项：
- `book version` 退出 0，单行输出 ``storybook X.Y.Z``；
- ``X.Y.Z == storybook.__version__ == pyproject.toml [project] version``；
- `book version --json` 输出含 name/version 的稳定 JSON；
- `book --help` Commands 段含 `version` 且非 hidden（见 test_cli_architecture.py）。
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from click.testing import CliRunner

from storybook import __version__
from storybook.cli import cli


def _pyproject_version() -> str:
    root = Path(__file__).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_version_output_matches_package_and_pyproject():
    result = CliRunner().invoke(cli, ["version"], prog_name="book")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == f"storybook {__version__}"
    # 单一事实源：storybook.__version__ 与 pyproject.toml [project] version 一致
    assert __version__ == _pyproject_version()
    assert result.stdout.strip() == f"storybook {_pyproject_version()}"


def test_version_json_output_stable():
    result = CliRunner().invoke(cli, ["version", "--json"], prog_name="book")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == {"name": "storybook", "version": __version__}
    # 稳定 JSON：字段顺序无关紧要，键值稳定
    assert set(payload) == {"name", "version"}


def test_version_json_stdout_is_pure():
    """JSON stdout 无兼容提示 / 无 warning 污染。"""
    result = CliRunner().invoke(cli, ["version", "--json"], prog_name="book")
    assert result.exit_code == 0, result.output
    assert "兼容" not in result.stdout
    assert json.loads(result.stdout)["version"] == __version__


def test_version_is_visible_in_help_and_not_hidden():
    result = CliRunner().invoke(cli, ["--help"], prog_name="book")
    assert result.exit_code == 0, result.output
    assert "version" in result.output
    cmd = cli.commands["version"]
    assert getattr(cmd, "hidden", False) is False
