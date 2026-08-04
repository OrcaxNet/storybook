"""一键 setup / uninstall 的隔离功能测试。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import click
from click.testing import CliRunner

from storybook import config
from storybook.cli import cli
from storybook.profiles import PlatformRoots, ProfileRegistry
from storybook.setup_adapters import Launcher
from storybook.setup_manager import SetupError, SetupManager


def _roots(tmp_path: Path) -> PlatformRoots:
    return PlatformRoots(
        config=tmp_path / "storybook-config",
        data=tmp_path / "storybook-data",
        cache=tmp_path / "storybook-cache",
        state=tmp_path / "storybook-state",
        logs=tmp_path / "storybook-logs",
    )


@pytest.fixture
def isolated_setup(tmp_path, monkeypatch):
    roots = _roots(tmp_path)
    registry = ProfileRegistry(
        roots.config / "profiles.json", roots=roots, environ={"HOME": str(tmp_path)}
    )
    old_registry = config.PROFILE_REGISTRY
    config.PROFILE_REGISTRY = registry
    config.refresh_profile(create=False)
    manager = SetupManager(
        home=tmp_path / "home",
        environ={"HOME": str(tmp_path / "home")},
        launcher=Launcher("/opt/storybook/bin/storybook"),
        roots=roots,
    )
    monkeypatch.setattr(
        manager,
        "_ensure_models",
        lambda **kwargs: (
            [
                {"name": config.EMBED_MODEL, "status": "cached"},
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        manager,
        "_smoke_tests",
        lambda selected: [
            {"name": "schema", "ok": True, "detail": "ready"},
            {"name": "embedding", "ok": True, "detail": "dimension=1024"},
            {"name": "recall", "ok": True, "detail": "matches=0"},
        ],
    )
    try:
        yield manager, roots
    finally:
        config.PROFILE_REGISTRY = old_registry
        config.refresh_profile(create=False)
        config.refresh_model_config()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _setup_state(adapters) -> dict:
    return {
        "schema_version": 1,
        "installed_at": "2026-08-01T00:00:00+00:00",
        "updated_at": "2026-08-01T00:00:00+00:00",
        "profile_id": "profile-test",
        "launcher": {"command": "/opt/storybook/bin/storybook", "args": []},
        "adapters": adapters,
    }


def _json_adapter_state(name: str) -> dict:
    state = {
        "adapter": name,
        "changed": True,
        "files": [],
        "previous": {"present": False, "value": None},
        "managed": {
            "command": "/opt/storybook/bin/storybook",
            "args": ["mcp"],
            "env": {},
        },
    }
    if name == "claude":
        state.update(
            {
                "managed_hook": {"matcher": "startup", "hooks": []},
                "hook_added": True,
            }
        )
    return state


def _tree_snapshot(root: Path) -> list[tuple[str, str, bytes | None]]:
    if not root.exists():
        return []
    return [
        (
            str(path.relative_to(root)),
            "dir" if path.is_dir() else "file",
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"), key=str)
    ]


def test_dry_run_fresh_home_performs_zero_writes(tmp_path):
    storybook_home = tmp_path / "never-created"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "storybook.cli", "setup", "--dry-run", "--json"],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run"
    assert payload["writes_performed"] == 0
    assert not storybook_home.exists()
    assert list(user_home.iterdir()) == []


def test_auto_detection_empty_home_and_path_selects_no_agents(tmp_path):
    storybook_home = tmp_path / "storybook-home"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PATH": "",
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "storybook.cli", "setup", "--dry-run", "--json"],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    adapters = payload["plan"]["adapters"]
    assert all(not item["detected"] for item in adapters)
    assert all(not item["selected"] for item in adapters)
    assert payload["plan"]["writes"] == []
    assert payload["writes_performed"] == 0
    assert not storybook_home.exists()
    assert list(user_home.iterdir()) == []


@pytest.mark.parametrize("signal", ["config", "directory", "executable"])
def test_auto_detection_selects_claude_for_each_install_signal(tmp_path, signal):
    storybook_home = tmp_path / "storybook-home"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    executable_dir = tmp_path / "bin"
    path_value = ""
    if signal == "config":
        (user_home / ".claude.json").write_text("{}\n", encoding="utf-8")
    elif signal == "directory":
        (user_home / ".claude").mkdir()
    else:
        executable_dir.mkdir()
        executable = executable_dir / "claude"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        path_value = str(executable_dir)
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PATH": path_value,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "storybook.cli", "setup", "--dry-run", "--json"],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    claude = next(
        item for item in payload["plan"]["adapters"] if item["adapter"] == "claude"
    )
    assert claude["detected"] is True
    assert claude["selected"] is True
    assert payload["writes_performed"] == 0
    assert not storybook_home.exists()


def test_explicit_claude_selection_overrides_missing_detection_signals(tmp_path):
    storybook_home = tmp_path / "storybook-home"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PATH": "",
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "storybook.cli",
            "setup",
            "--dry-run",
            "--json",
            "--agent",
            "claude",
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    claude = next(
        item for item in payload["plan"]["adapters"] if item["adapter"] == "claude"
    )
    assert claude["detected"] is False
    assert claude["selected"] is True
    assert payload["plan"]["writes"] == claude["targets"]
    assert payload["writes_performed"] == 0
    assert not storybook_home.exists()
    assert list(user_home.iterdir()) == []


def test_setup_merges_all_agents_is_idempotent_and_uninstall_keeps_data(
    isolated_setup,
):
    manager, roots = isolated_setup
    home = manager.home
    _write_json(
        home / ".claude.json",
        {"theme": "dark", "mcpServers": {"existing": {"command": "keep"}}},
    )
    existing_hook = {"matcher": "startup", "hooks": [{"type": "command", "command": "keep"}]}
    _write_json(
        home / ".claude" / "settings.json",
        {"permissions": {"allow": ["Read"]}, "hooks": {"SessionStart": [existing_hook]}},
    )
    _write_json(
        home / ".cursor" / "mcp.json",
        {"mcpServers": {"existing": {"command": "keep"}}},
    )
    codex = home / ".codex" / "config.toml"
    codex.parent.mkdir(parents=True)
    codex.write_text('model = "keep"\n\n[mcp_servers.existing]\ncommand = "keep"\n', encoding="utf-8")

    first = manager.execute(
        requested_agents=("claude", "cursor", "codex"), download_models=False
    )
    second = manager.execute(
        requested_agents=("claude", "cursor", "codex"), download_models=False
    )

    assert first["status"] == "ready"
    assert all(item["changed"] for item in first["adapters"])
    assert not any(item["changed"] for item in second["adapters"])
    claude = json.loads((home / ".claude.json").read_text(encoding="utf-8"))
    assert claude["theme"] == "dark"
    assert claude["mcpServers"]["existing"] == {"command": "keep"}
    assert claude["mcpServers"]["storybook"]["args"] == ["mcp"]
    settings = json.loads(
        (home / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert existing_hook in settings["hooks"]["SessionStart"]
    assert len(settings["hooks"]["SessionStart"]) == 2
    cursor = json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert cursor["mcpServers"]["existing"] == {"command": "keep"}
    parsed_codex = tomllib.loads(codex.read_text(encoding="utf-8"))
    assert parsed_codex["model"] == "keep"
    assert parsed_codex["mcp_servers"]["existing"]["command"] == "keep"
    assert parsed_codex["mcp_servers"]["storybook"]["args"] == ["mcp"]
    assert manager.state_path.is_file()
    assert list((roots.state / "setup-backups").rglob("*.bak"))

    database = Path(first["profile"]["database"])
    assert database.is_file()
    result = manager.uninstall()

    assert result["status"] == "uninstalled"
    assert result["data"] == "kept"
    assert database.is_file()
    assert "storybook" not in json.loads(
        (home / ".claude.json").read_text(encoding="utf-8")
    )["mcpServers"]
    restored_settings = json.loads(
        (home / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert restored_settings["hooks"]["SessionStart"] == [existing_hook]
    assert "storybook" not in json.loads(
        (home / ".cursor" / "mcp.json").read_text(encoding="utf-8")
    )["mcpServers"]
    assert "storybook" not in tomllib.loads(codex.read_text(encoding="utf-8"))[
        "mcp_servers"
    ]


def test_uninstall_restores_preexisting_storybook_nodes(isolated_setup):
    manager, _ = isolated_setup
    home = manager.home
    previous = {"command": "/custom/storybook", "args": ["serve"]}
    _write_json(home / ".cursor" / "mcp.json", {"mcpServers": {"storybook": previous}})
    codex = home / ".codex" / "config.toml"
    codex.parent.mkdir(parents=True)
    codex.write_text(
        '[mcp_servers.storybook]\ncommand = "/custom/storybook"\nargs = ["serve"]\n',
        encoding="utf-8",
    )

    manager.execute(requested_agents=("cursor", "codex"), download_models=False)
    result = manager.uninstall()

    assert result["status"] == "uninstalled"
    cursor = json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert cursor["mcpServers"]["storybook"] == previous
    codex_payload = tomllib.loads(codex.read_text(encoding="utf-8"))
    assert codex_payload["mcp_servers"]["storybook"]["command"] == "/custom/storybook"
    assert codex_payload["mcp_servers"]["storybook"]["args"] == ["serve"]


def test_setup_updates_changed_launcher_without_duplicating_hook(
    isolated_setup, monkeypatch
):
    manager, roots = isolated_setup
    manager.execute(requested_agents=("claude",), download_models=False)
    updated = SetupManager(
        home=manager.home,
        environ=manager.environ,
        launcher=Launcher("/new/location/storybook"),
        roots=roots,
    )
    monkeypatch.setattr(
        updated,
        "_ensure_models",
        lambda **kwargs: ([], []),
    )
    monkeypatch.setattr(
        updated,
        "_smoke_tests",
        lambda selected: [{"name": "schema", "ok": True, "detail": "ready"}],
    )

    result = updated.execute(requested_agents=("claude",), download_models=False)

    assert result["status"] == "ready"
    claude = json.loads((updated.home / ".claude.json").read_text(encoding="utf-8"))
    assert claude["mcpServers"]["storybook"]["command"] == "/new/location/storybook"
    settings = json.loads(
        (updated.home / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert len(settings["hooks"]["SessionStart"]) == 1
    assert "/new/location/storybook" in settings["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_state_write_failure_rolls_back_first_install_and_returns_json_error(
    isolated_setup, monkeypatch
):
    manager, roots = isolated_setup
    path = manager.home / ".cursor" / "mcp.json"
    _write_json(path, {"mcpServers": {"existing": {"command": "keep"}}})
    fixed_mtime_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed_mtime_ns, fixed_mtime_ns))
    before = path.read_bytes()

    def fail_state_write(state):
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_write_state", fail_state_write)
    monkeypatch.setattr("storybook.cli.SetupManager", lambda: manager)

    result = CliRunner().invoke(
        cli, ["setup", "--json", "--agent", "cursor", "--skip-models"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "SB_SETUP_STATE_WRITE_FAILED"
    assert payload["error"]["message"]
    assert payload["error"]["hint"]
    assert "Traceback" not in result.output
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == fixed_mtime_ns
    assert "storybook" not in json.loads(path.read_text(encoding="utf-8"))["mcpServers"]
    assert not manager.state_path.exists()
    assert not (roots.state / "setup-backups").exists()


def test_state_write_failure_rolls_back_launcher_upgrade_and_preserves_old_state(
    isolated_setup, monkeypatch
):
    manager, roots = isolated_setup
    path = manager.home / ".cursor" / "mcp.json"
    _write_json(path, {"mcpServers": {"existing": {"command": "keep"}}})
    manager.execute(requested_agents=("cursor",), download_models=False)

    updated = SetupManager(
        home=manager.home,
        environ=manager.environ,
        launcher=Launcher("/new/location/storybook"),
        roots=roots,
    )
    monkeypatch.setattr(updated, "_ensure_models", lambda **kwargs: ([], []))
    monkeypatch.setattr(
        updated,
        "_smoke_tests",
        lambda selected: [{"name": "schema", "ok": True, "detail": "ready"}],
    )
    config_mtime_ns = 1_700_000_000_100_000_000
    state_mtime_ns = 1_700_000_000_200_000_000
    os.utime(path, ns=(config_mtime_ns, config_mtime_ns))
    os.utime(manager.state_path, ns=(state_mtime_ns, state_mtime_ns))
    config_before = path.read_bytes()
    state_before = manager.state_path.read_bytes()
    backup_root = roots.state / "setup-backups"
    backups_before = sorted(
        str(item.relative_to(backup_root)) for item in backup_root.rglob("*")
    )

    def fail_state_write(state):
        raise OSError("disk full")

    monkeypatch.setattr(updated, "_write_state", fail_state_write)

    with pytest.raises(SetupError) as raised:
        updated.execute(requested_agents=("cursor",), download_models=False)

    assert raised.value.code == "SB_SETUP_STATE_WRITE_FAILED"
    assert raised.value.hint
    assert path.read_bytes() == config_before
    assert path.stat().st_mtime_ns == config_mtime_ns
    assert manager.state_path.read_bytes() == state_before
    assert manager.state_path.stat().st_mtime_ns == state_mtime_ns
    assert "/opt/storybook/bin/storybook" in path.read_text(encoding="utf-8")
    assert "/new/location/storybook" not in path.read_text(encoding="utf-8")
    assert sorted(
        str(item.relative_to(backup_root)) for item in backup_root.rglob("*")
    ) == backups_before

    uninstall = manager.uninstall()

    assert uninstall["status"] == "uninstalled"
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored == {"mcpServers": {"existing": {"command": "keep"}}}


def test_purge_removes_profile_roots_after_explicit_manager_call(isolated_setup):
    manager, roots = isolated_setup
    result = manager.execute(requested_agents=(), download_models=False)
    database = Path(result["profile"]["database"])
    assert database.is_file()

    removed = manager.uninstall(purge_data=True)

    assert removed["status"] == "uninstalled"
    assert removed["data"] == "purged"
    assert not database.exists()
    assert not roots.data.exists()
    assert not roots.config.exists()


def test_malformed_agent_config_has_stable_error_code(isolated_setup):
    manager, _ = isolated_setup
    path = manager.home / ".cursor" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(SetupError) as raised:
        manager.plan(("cursor",))

    assert raised.value.code == "SB_SETUP_CONFIG_INVALID"
    assert "修复配置语法" in (raised.value.hint or "")


def test_codex_semantically_configured_without_marker_is_not_rewritten(
    isolated_setup,
):
    manager, roots = isolated_setup
    path = manager.home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "[mcp_servers.storybook]\n"
        'command = "/opt/storybook/bin/storybook"\n'
        'args = ["mcp"]\n'
        "startup_timeout_sec = 20\n"
        "tool_timeout_sec = 60\n",
        encoding="utf-8",
    )
    fixed_mtime_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed_mtime_ns, fixed_mtime_ns))
    before = path.read_bytes()

    plan = manager.plan(("codex",)).as_dict()
    codex_plan = next(item for item in plan["adapters"] if item["adapter"] == "codex")
    result = manager.execute(requested_agents=("codex",), download_models=False)

    assert codex_plan["changed"] is False
    assert result["adapters"] == [{"name": "codex", "changed": False}]
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == fixed_mtime_ns
    assert not (roots.state / "setup-backups").exists()
    state = json.loads(manager.state_path.read_text(encoding="utf-8"))
    assert state["selected_adapters"] == ["codex"]

    manager.execute(requested_agents=("cursor",), download_models=False)
    state = json.loads(manager.state_path.read_text(encoding="utf-8"))
    assert state["selected_adapters"] == ["codex", "cursor"]


@pytest.mark.parametrize(
    ("agent", "relative_path", "content", "message_fragment"),
    [
        (
            "codex",
            ".codex/config.toml",
            'mcp_servers = "valid-toml-but-wrong-shape"\n',
            "mcp_servers 必须是 table",
        ),
        (
            "claude",
            ".claude/settings.json",
            '{"hooks": []}\n',
            "hooks 必须是 object",
        ),
    ],
)
def test_wrong_shape_agent_config_returns_json_error_without_traceback(
    tmp_path, agent, relative_path, content, message_fragment
):
    storybook_home = tmp_path / "storybook-home"
    user_home = tmp_path / "user-home"
    config_path = user_home / relative_path
    config_path.parent.mkdir(parents=True)
    config_path.write_text(content, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "storybook.cli",
            "setup",
            "--dry-run",
            "--json",
            "--agent",
            agent,
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "SB_SETUP_CONFIG_INVALID"
    assert message_fragment in payload["error"]["message"]
    assert "修复配置语法" in payload["error"]["hint"]
    assert "Traceback" not in completed.stderr
    assert "AttributeError" not in completed.stderr
    assert not storybook_home.exists()


def test_explicit_agent_skips_invalid_unselected_adapter_config(tmp_path):
    storybook_home = tmp_path / "storybook-home"
    user_home = tmp_path / "user-home"
    claude_settings = user_home / ".claude" / "settings.json"
    claude_settings.parent.mkdir(parents=True)
    claude_settings.write_text('{"hooks": []}\n', encoding="utf-8")
    before = claude_settings.read_bytes()
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "storybook.cli",
            "setup",
            "--dry-run",
            "--json",
            "--agent",
            "cursor",
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run"
    assert payload["writes_performed"] == 0
    selected = [
        item["adapter"] for item in payload["plan"]["adapters"] if item["selected"]
    ]
    assert selected == ["cursor"]
    assert claude_settings.read_bytes() == before
    assert not storybook_home.exists()


def test_invalid_profile_registry_returns_json_error_without_writes(tmp_path):
    storybook_home = tmp_path / "storybook-home"
    user_home = tmp_path / "user-home"
    registry = storybook_home / "config" / "profiles.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{broken", encoding="utf-8")
    before = registry.read_bytes()
    before_tree = sorted(
        str(path.relative_to(storybook_home)) for path in storybook_home.rglob("*")
    )
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "storybook.cli",
            "setup",
            "--dry-run",
            "--json",
            "--agent",
            "cursor",
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "SB_SETUP_PROFILE_INVALID"
    assert payload["error"]["message"]
    assert payload["error"]["hint"]
    assert "Traceback" not in completed.stderr
    assert "ProfileError" not in completed.stderr
    assert registry.read_bytes() == before
    after_tree = sorted(
        str(path.relative_to(storybook_home)) for path in storybook_home.rglob("*")
    )
    assert after_tree == before_tree
    assert not user_home.exists()


@pytest.mark.parametrize(
    ("adapters", "case"),
    [
        (1, "adapters-number"),
        ("invalid", "adapters-string"),
        ([], "adapters-list"),
        ({"cursor": []}, "adapter-record-list"),
        (
            {
                "cursor": {
                    **_json_adapter_state("cursor"),
                    "previous": [],
                }
            },
            "json-previous-list",
        ),
        (
            {
                "claude": {
                    **_json_adapter_state("claude"),
                    "hook_added": "yes",
                }
            },
            "claude-hook-added-string",
        ),
        (
            {
                "codex": {
                    "adapter": "codex",
                    "changed": True,
                    "files": [],
                    "previous_blocks": "",
                    "managed_block": 1,
                }
            },
            "codex-managed-block-number",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize(
    "command",
    [
        ["setup", "--yes", "--json", "--skip-models", "--agent", "cursor"],
        ["uninstall", "--yes", "--json"],
    ],
    ids=["setup", "uninstall"],
)
def test_invalid_setup_state_returns_json_error_without_writes(
    tmp_path, adapters, case, command
):
    storybook_home = tmp_path / "storybook-home"
    user_home = tmp_path / "user-home"
    state_path = storybook_home / "state" / "setup-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(_setup_state(adapters), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fixed_mtime_ns = 1_700_000_000_000_000_000
    os.utime(state_path, ns=(fixed_mtime_ns, fixed_mtime_ns))
    before = _tree_snapshot(storybook_home)
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
            "CODEX_HOME": str(user_home / ".codex"),
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "storybook.cli", *command],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1, case
    payload = json.loads(completed.stdout)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "SB_SETUP_STATE_INVALID"
    assert payload["error"]["message"]
    assert payload["error"]["hint"]
    assert "Traceback" not in completed.stderr
    assert "TypeError" not in completed.stderr
    assert "AttributeError" not in completed.stderr
    assert _tree_snapshot(storybook_home) == before
    assert state_path.stat().st_mtime_ns == fixed_mtime_ns
    assert not user_home.exists()


def test_model_unavailable_returns_degraded_not_failed(isolated_setup, monkeypatch):
    manager, _ = isolated_setup
    monkeypatch.setattr(
        manager,
        "_ensure_models",
        lambda **kwargs: (
            [{"name": config.EMBED_MODEL, "status": "unavailable"}],
            ["Ollama unavailable"],
        ),
    )

    result = manager.execute(requested_agents=(), download_models=True)

    assert result["status"] == "degraded"
    assert result["degraded_reasons"] == ["Ollama unavailable"]


def test_legacy_storybook_init_and_admin_init_db_share_low_level_behavior(monkeypatch):
    calls = []
    monkeypatch.setattr("storybook.cli.store.init_db", lambda: calls.append("init"))

    legacy = CliRunner().invoke(cli, ["init"], prog_name="storybook")
    admin = CliRunner().invoke(cli, ["admin", "init-db"], prog_name="book")

    assert legacy.exit_code == 0, legacy.output
    assert admin.exit_code == 0, admin.output
    assert calls == ["init", "init"]


def test_book_init_json_reuses_setup_contract(isolated_setup, monkeypatch):
    manager, _ = isolated_setup
    monkeypatch.setattr("storybook.cli.SetupManager", lambda: manager)

    result = CliRunner().invoke(
        cli,
        ["init", "--yes", "--json", "--skip-models", "--agent", "codex"],
        prog_name="book",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["next_command"].startswith("book search")
    assert payload["profile"]["id"]
    assert payload["model_config"]["embedding"]["provider"] == "ollama"


def test_book_help_hides_setup_compatibility_alias():
    result = CliRunner().invoke(cli, ["--help"], prog_name="book")

    assert result.exit_code == 0, result.output
    assert "  init " in result.output
    assert "  setup " not in result.output


def test_book_init_interactive_api_secret_is_ephemeral_and_hidden(
    isolated_setup, monkeypatch
):
    manager, roots = isolated_setup
    sentinel = "SECRET-SENTINEL-NEVER-PERSIST"
    monkeypatch.setattr("storybook.cli.SetupManager", lambda: manager)

    def probe(value):
        assert manager.environ["TEST_BOOK_KEY"] == sentinel
        return [
            {"name": "generation", "ok": True, "detail": "ready"},
            {"name": "embedding-provider", "ok": True, "detail": "ready"},
        ]

    monkeypatch.setattr(manager, "_probe_provider", probe)
    original_stream = click.get_text_stream

    class TtyInput:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def isatty(self):
            return True

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    monkeypatch.setattr(
        click,
        "get_text_stream",
        lambda name: TtyInput(original_stream(name)) if name == "stdin" else original_stream(name),
    )
    user_input = "\n".join([
        "api",
        "https://models.example.test",
        "generation-v1",
        "embedding-v1",
        "TEST_BOOK_KEY",
        "auto",
        "skip",
        "y",
        sentinel,
        "",
    ])

    result = CliRunner().invoke(cli, ["init"], input=user_input, prog_name="book")

    assert result.exit_code == 0, result.output
    assert sentinel not in result.output
    assert "TEST_BOOK_KEY" not in manager.environ
    written = b"".join(
        path.read_bytes() for path in tmp_files(roots.config, roots.state)
    )
    assert sentinel.encode() not in written


def tmp_files(*roots: Path) -> list[Path]:
    return [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]


def test_enable_schedule_is_user_owned_and_idempotent(isolated_setup):
    manager, _ = isolated_setup

    first = manager.execute(
        requested_agents=(), download_models=False, enable_schedule=True
    )
    before = manager.schedule_path.read_bytes()
    second = manager.execute(
        requested_agents=(), download_models=False, enable_schedule=True
    )

    assert first["schedule"]["status"] == "configured"
    assert second["schedule"]["status"] == "configured"
    assert manager.schedule_path.read_bytes() == before
    assert str(manager.schedule_path).startswith(str(manager.home))
    assert "sudo" not in before.decode("utf-8", errors="ignore")
    removed = manager.uninstall()
    assert removed["schedule"]["status"] == "removed"
    assert not manager.schedule_path.exists()


def _invoke_runtime_status(
    manager, monkeypatch, *, reachable=True, models=None, credentials=True
):
    tags = {"models": [{"name": name} for name in (models or ())]}
    monkeypatch.setattr(
        "storybook.setup_manager.health._check_ollama_reachable",
        lambda: (reachable, tags if reachable else None, "offline"),
    )
    monkeypatch.setattr("storybook.cli.SetupManager", lambda: manager)
    monkeypatch.setattr(config, "LLM_API_KEY", "configured" if credentials else None)
    result = CliRunner().invoke(cli, ["status", "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_status_reports_normal_ready_components(isolated_setup, monkeypatch):
    manager, _ = isolated_setup
    manager.execute(requested_agents=("codex",), download_models=False)

    payload = _invoke_runtime_status(
        manager,
        monkeypatch,
        models=(config.EMBED_MODEL,),
    )

    assert payload["status"] == "ready"
    assert payload["profile"]["status"] == "ready"
    assert payload["model"]["status"] == "ready"
    assert payload["adapter"] == {
        "status": "ready",
        "checks": [{"name": "codex", "status": "ready"}],
    }
    assert payload["sync"] == {"state": "local_only", "enabled": False}
    assert payload["degraded_reasons"] == []


def test_status_reports_ollama_unavailable(isolated_setup, monkeypatch):
    manager, _ = isolated_setup
    manager.execute(requested_agents=("codex",), download_models=False)

    payload = _invoke_runtime_status(manager, monkeypatch, reachable=False)

    assert payload["status"] == "ready_degraded"
    assert payload["model"]["status"] == "degraded"
    assert payload["degraded_reasons"] == ["ollama_unavailable"]


def test_status_reports_missing_embedding_model(isolated_setup, monkeypatch):
    manager, _ = isolated_setup
    manager.execute(requested_agents=("codex",), download_models=False)

    payload = _invoke_runtime_status(
        manager, monkeypatch, models=()
    )

    assert payload["status"] == "ready_degraded"
    assert payload["model"]["embedding"]["status"] == "missing"
    assert payload["degraded_reasons"] == ["model_missing:embedding"]


def test_status_reports_managed_adapter_missing(isolated_setup, monkeypatch):
    manager, _ = isolated_setup
    manager.execute(requested_agents=("codex",), download_models=False)
    (manager.home / ".codex" / "config.toml").unlink()

    payload = _invoke_runtime_status(
        manager,
        monkeypatch,
        models=(config.EMBED_MODEL,),
    )

    assert payload["status"] == "ready_degraded"
    assert payload["adapter"] == {
        "status": "degraded",
        "checks": [{"name": "codex", "status": "missing"}],
    }
    assert payload["degraded_reasons"] == ["adapter_unavailable:codex"]


def test_status_reports_invalid_setup_state_without_crashing(
    isolated_setup, monkeypatch
):
    manager, _ = isolated_setup
    manager.state_path.parent.mkdir(parents=True, exist_ok=True)
    manager.state_path.write_text("{broken", encoding="utf-8")

    payload = _invoke_runtime_status(
        manager,
        monkeypatch,
        models=(config.EMBED_MODEL,),
    )

    assert payload["status"] == "ready_degraded"
    assert payload["adapter"] == {"status": "not_configured", "checks": []}
    assert payload["degraded_reasons"] == ["setup_state_invalid"]


def test_status_reports_mixed_providers_and_missing_llm_credentials(
    isolated_setup, monkeypatch
):
    manager, _ = isolated_setup
    payload = _invoke_runtime_status(
        manager,
        monkeypatch,
        models=(config.EMBED_MODEL,),
        credentials=False,
    )

    assert payload["model"]["provider"] == "hybrid"
    assert payload["model"]["llm"] == {
        "provider": "deepseek_anthropic",
        "name": config.LLM_MODEL,
        "status": "credentials_missing",
    }
    assert payload["model"]["embedding"]["provider"] == "ollama"
    assert payload["degraded_reasons"] == ["llm_credentials_missing"]


def test_ensure_models_only_checks_and_pulls_embedding(isolated_setup, monkeypatch):
    manager, _ = isolated_setup
    monkeypatch.delattr(manager, "_ensure_models")
    checked = []
    pulled = []
    monkeypatch.setattr(
        "storybook.setup_manager._ollama_tags",
        lambda: checked.append(True) or {},
    )
    monkeypatch.setattr(
        "storybook.setup_manager._pull_model",
        lambda model, progress=None: pulled.append(model),
    )

    models, degraded = manager._ensure_models(download=True, progress=None)

    assert checked == [True]
    assert pulled == [config.EMBED_MODEL]
    assert models == [
        {"name": config.EMBED_MODEL, "status": "downloaded", "size": None}
    ]
    assert degraded == []


def test_noninteractive_purge_requires_second_confirmation():
    result = CliRunner().invoke(
        cli, ["uninstall", "--yes", "--purge-data", "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "SB_UNINSTALL_PURGE_CONFIRM_REQUIRED"
