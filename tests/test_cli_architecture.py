"""FLO-180：`book` canonical 入口与 CLI 信息架构测试。

覆盖验收项：
- `book --help` 只显示 canonical 顶层信息架构，主要任务不超过两层；
- `book run` 四种模式与所有冲突组合 fail-fast；默认 `run` 与 `run --once` 等价；
- canonical / legacy 对照：相同业务调用、退出码、结构化输出；JSON stdout 无 warning；
- 新生成 schedule 配置调用 `book run --watch`，旧配置入口仍可运行；
- clean install 同时生成 `book` 与 `storybook` entrypoint。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from storybook import config
from storybook.cli import cli
from storybook.profiles import PlatformRoots, ProfileRegistry
from storybook.setup_adapters import Launcher
from storybook.setup_manager import SetupManager


CANONICAL_TOP_LEVEL = {
    "admin", "doctor", "init", "mcp", "memory", "profile", "run",
    "search", "source", "status",
}
LEGACY_TOP_LEVEL = {
    "setup", "process", "dream", "import-data", "sources",
    "list", "show", "forget", "stats", "sync", "prime",
}


def _help_names(prog_name="book") -> set[str]:
    """从 ``--help`` 的 Commands 段解析可见命令名。"""
    result = CliRunner().invoke(cli, ["--help"], prog_name=prog_name)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Commands:")
    except StopIteration:
        return set()
    names: set[str] = set()
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        match = re.match(r"^  (\S+)", line)
        if match:
            names.add(match.group(1))
    return names


def _invoke(args, prog_name="book", **kwargs):
    return CliRunner().invoke(cli, args, prog_name=prog_name, **kwargs)


# ═══════════════════════════════════════════════
#  信息架构
# ═══════════════════════════════════════════════


def test_book_help_shows_only_canonical_top_level():
    names = _help_names("book")
    assert names == CANONICAL_TOP_LEVEL


def test_legacy_commands_are_hidden_from_help():
    names = _help_names("book")
    assert not (names & LEGACY_TOP_LEVEL)
    for name in LEGACY_TOP_LEVEL:
        cmd = cli.commands[name]
        assert getattr(cmd, "hidden", False) is True, f"{name} 应默认从 help 隐藏"


def test_canonical_top_level_commands_are_not_hidden():
    for name in CANONICAL_TOP_LEVEL:
        assert getattr(cli.commands[name], "hidden", False) is False, name


def test_storybook_help_also_hides_legacy_commands():
    names = _help_names("storybook")
    assert names == CANONICAL_TOP_LEVEL


def test_memory_group_subcommands():
    group = cli.commands["memory"]
    assert set(group.commands) == {"list", "show", "forget"}
    assert all(not getattr(c, "hidden", False) for c in group.commands.values())


def test_source_group_subcommands_and_legacy_alias():
    group = cli.commands["source"]
    assert set(group.commands) == {"list", "enable", "disable", "reset"}
    alias = cli.commands["sources"]
    assert set(alias.commands) == {
        "list", "enable", "disable", "reset-checkpoint"
    }


def test_admin_group_subcommands():
    group = cli.commands["admin"]
    assert set(group.commands) == {
        "init-db", "migration", "index", "benchmark", "eval", "uninstall"
    }
    migration = group.commands["migration"]
    assert set(migration.commands) == {
        "discover", "run", "rollback", "status", "delete-backup"
    }


def test_major_tasks_are_at_most_two_levels_deep():
    """canonical 顶层命令下面最多一层分组；admin migration 是唯一三层例外（低频维护）。"""
    for name, cmd in cli.commands.items():
        if getattr(cmd, "hidden", False):
            continue
        # 只有 memory / source / profile / admin 是分组，且其子命令必须是叶子命令
        if isinstance(cmd, type(cli)) and name != "admin":
            for sub in cmd.commands.values():
                assert not isinstance(sub, type(cli)), f"{name} 的子命令不应再分组"


# ═══════════════════════════════════════════════
#  book run 模式与冲突
# ═══════════════════════════════════════════════


def _fake_cycle_result():
    return {
        "status": "ok", "imported": 1, "updated": 0, "total": 2,
        "success": 2, "failed": 0, "duration_s": 0.1, "sources": [],
    }


def test_run_conflict_combinations_fail_fast(monkeypatch):
    monkeypatch.setattr("storybook.cli.dreamd.run_dream_cycle_once",
                        lambda **kwargs: _fake_cycle_result())
    for args in [
        ["run", "--watch", "--daemon"],
        ["run", "--watch", "--session", "5"],
        ["run", "--daemon", "--session", "5"],
        ["run", "--once", "--watch"],
        ["run", "--once", "--daemon"],
    ]:
        result = _invoke(args)
        assert result.exit_code == 2, args
        assert "Error:" in result.stderr, args


def test_run_default_equals_run_once(monkeypatch):
    calls = []
    monkeypatch.setattr("storybook.cli.dreamd.setup_dream_logging",
                        lambda *a, **k: None)

    def fake_cycle(**kwargs):
        calls.append(kwargs)
        return _fake_cycle_result()

    monkeypatch.setattr("storybook.cli.dreamd.run_dream_cycle_once", fake_cycle)

    default = _invoke(["run"])
    once = _invoke(["run", "--once"])

    assert default.exit_code == 0, default.output
    assert once.exit_code == 0, once.output
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["import_new"] is True
    assert calls[0]["verbose"] is True


def test_run_modes_dispatch_to_correct_business_call(monkeypatch):
    calls = []
    monkeypatch.setattr("storybook.cli.dreamd.setup_dream_logging",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        "storybook.cli.dreamd.run_dream_cycle_once",
        lambda **kwargs: calls.append(("cycle", kwargs)) or _fake_cycle_result(),
    )
    monkeypatch.setattr(
        "storybook.cli.dreamd.watch_loop",
        lambda **kwargs: calls.append(("watch", kwargs)),
    )
    monkeypatch.setattr(
        "storybook.cli.dreamd.dream_daemon",
        lambda **kwargs: calls.append(("daemon", kwargs)),
    )
    monkeypatch.setattr(
        "storybook.cli.processor.process_session",
        lambda sid: calls.append(("session", sid)) or 42,
    )

    assert _invoke(["run", "--watch", "--interval", "30"]).exit_code == 0
    assert _invoke(["run", "--daemon", "--interval", "60"]).exit_code == 0
    assert _invoke(["run", "--session", "9"]).exit_code == 0

    kinds = [entry[0] for entry in calls]
    assert kinds == ["watch", "daemon", "session"]
    assert calls[0][1]["poll_interval"] == 30
    assert calls[1][1]["interval"] == 60
    assert calls[2][1] == 9


# ═══════════════════════════════════════════════
#  canonical / legacy 对照
# ═══════════════════════════════════════════════


def test_memory_list_parity_with_top_level_list(monkeypatch, tmp_path):
    from storybook import store
    # 空库时两条路径输出一致（相同业务函数）
    canonical = _invoke(["memory", "list"])
    legacy = _invoke(["list"], prog_name="storybook")
    assert canonical.exit_code == 0
    assert legacy.exit_code == 0
    # canonical 无 stderr 提示；legacy 有 stderr 提示且 stdout 一致
    assert canonical.stderr == ""
    assert "兼容" in legacy.stderr
    assert canonical.stdout == legacy.stdout


def test_run_once_parity_with_dream_once(monkeypatch):
    calls = []
    monkeypatch.setattr("storybook.cli.dreamd.setup_dream_logging",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        "storybook.cli.dreamd.run_dream_cycle_once",
        lambda **kwargs: calls.append(kwargs) or _fake_cycle_result(),
    )
    canonical = _invoke(["run", "--once"])
    legacy = _invoke(["dream", "--once"], prog_name="storybook")
    assert canonical.exit_code == 0
    assert legacy.exit_code == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[1]["import_new"] is True
    assert canonical.stdout == legacy.stdout


def test_run_session_parity_with_process_session(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "storybook.cli.processor.process_session",
        lambda sid: calls.append(sid) or 7,
    )
    canonical = _invoke(["run", "--session", "7"])
    legacy = _invoke(["process", "--session", "7"], prog_name="storybook")
    assert canonical.exit_code == 0
    assert legacy.exit_code == 0
    assert calls == [7, 7]
    assert canonical.stdout == legacy.stdout


def test_source_list_parity_with_sources_alias(monkeypatch):
    monkeypatch.setattr(
        "storybook.cli.source_manager.list_sources",
        lambda: [
            {"name": "codex", "available": True, "enabled": True,
             "status": "ok", "adapter_version": "1.0"},
        ],
    )
    canonical = _invoke(["source", "list", "--json"])
    legacy = _invoke(["sources", "list", "--json"], prog_name="storybook")
    assert canonical.exit_code == 0
    assert legacy.exit_code == 0
    # JSON stdout 无 warning（兼容提示只进 stderr）
    assert json.loads(canonical.stdout) == json.loads(legacy.stdout)
    assert canonical.stderr == ""
    assert "兼容" in legacy.stderr


def test_setup_parity_with_init_dry_run(monkeypatch):
    """setup（隐藏兼容 alias）与 book init 走同一 onboarding 业务函数。"""
    manager = _FakeSetupManager()
    monkeypatch.setattr("storybook.cli.SetupManager", lambda: manager)

    init_result = _invoke(
        ["init", "--dry-run", "--json", "--skip-models", "--agent", "codex"],
        prog_name="book",
    )
    setup_result = _invoke(
        ["setup", "--dry-run", "--json", "--skip-models", "--agent", "codex"]
    )

    assert init_result.exit_code == 0, init_result.output
    assert setup_result.exit_code == 0, setup_result.output
    assert json.loads(init_result.stdout) == json.loads(setup_result.stdout)
    assert "兼容" in setup_result.stderr


def test_legacy_alias_hint_never_pollutes_json_stdout():
    """`storybook` executable 的兼容提示只进 stderr；JSON stdout 保持纯净。"""
    result = _invoke(["status", "--json"], prog_name="storybook")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "status" in payload
    assert "兼容" in result.stderr
    assert "兼容" not in result.stdout


# ═══════════════════════════════════════════════
#  生成配置调用 book
# ═══════════════════════════════════════════════


def test_schedule_uses_book_run_watch(tmp_path, monkeypatch):
    roots = PlatformRoots(
        config=tmp_path / "cfg", data=tmp_path / "data",
        cache=tmp_path / "cache", state=tmp_path / "state",
        logs=tmp_path / "logs",
    )
    home = tmp_path / "home"
    registry = ProfileRegistry(roots.config / "profiles.json", roots=roots,
                               environ={"HOME": str(home)})
    old_registry = config.PROFILE_REGISTRY
    config.PROFILE_REGISTRY = registry
    config.refresh_profile(create=False)
    try:
        manager = SetupManager(
            home=home, environ={"HOME": str(home)},
            launcher=Launcher("/opt/storybook/bin/book"), roots=roots,
        )
        payload = manager._write_schedule()
        text = Path(payload["path"]).read_bytes().decode("utf-8")
        assert "/opt/storybook/bin/book" in text
        assert "run" in text
        assert "--watch" in text
        assert "process" not in text
    finally:
        config.PROFILE_REGISTRY = old_registry
        config.refresh_profile(create=False)


def test_pyproject_defines_both_book_and_storybook_entrypoints():
    import tomllib
    root = Path(__file__).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    scripts = data["project"]["scripts"]
    assert scripts["book"] == "storybook.cli:cli"
    assert scripts["storybook"] == "storybook.cli:cli"


# ═══════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════


class _FakeSetupManager:
    """最小 SetupManager 桩，支持 dry-run JSON 计划输出。"""

    environ = {}
    state_path = Path("/tmp/nonexistent/setup-state.json")

    def plan(self, agents, provider_config=None):
        class Plan:
            @staticmethod
            def as_dict():
                return {
                    "profile": {
                        "action": "create", "display_name": "default",
                        "sync_state": "local_only",
                    },
                    "adapters": [
                        {
                            "display_name": "Codex", "selected": True,
                            "changed": True,
                            "targets": ["/x/.codex/config.toml"],
                        }
                    ],
                    "embedding": {
                        "preset": "ollama", "adapter": "ollama",
                        "model": "embedding-v1", "dimension": 1024,
                    },
                    "models": ["generation-v1"],
                    "legacy_databases": [],
                }

        return Plan()

    def execute(self, **kwargs):  # pragma: no cover
        raise AssertionError("dry-run 不应执行")
