"""一键 setup / uninstall 的隔离功能测试。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
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
    old_profile_id = config.PROFILE_ID
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
                {"name": config.LLM_MODEL, "status": "cached"},
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
        config.refresh_profile(old_profile_id)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_dry_run_fresh_home_performs_zero_writes(tmp_path):
    storybook_home = tmp_path / "never-created"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "STORYBOOK_HOME": str(storybook_home),
            "HOME": str(user_home),
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


def test_noninteractive_purge_requires_second_confirmation():
    result = CliRunner().invoke(
        cli, ["uninstall", "--yes", "--purge-data", "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "SB_UNINSTALL_PURGE_CONFIRM_REQUIRED"
