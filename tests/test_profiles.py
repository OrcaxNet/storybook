"""用户级 Profile registry、平台目录与 CLI 状态测试。"""
from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from storybook import config, performance
from storybook.cli import cli
from storybook.profiles import (
    DEFAULT_SYNC_STATE,
    PlatformRoots,
    ProfileError,
    ProfileRegistry,
    platform_roots,
)


def _roots(tmp_path: Path) -> PlatformRoots:
    return PlatformRoots(
        config=tmp_path / "config",
        data=tmp_path / "data",
        cache=tmp_path / "cache",
        state=tmp_path / "state",
        logs=tmp_path / "logs",
    )


class TestPlatformRoots:
    def test_macos_uses_application_support_caches_and_logs(self):
        roots = platform_roots("darwin", environ={}, home=Path("/Users/alice"))
        assert roots.data == Path(
            "/Users/alice/Library/Application Support/Storybook"
        )
        assert roots.config == roots.data
        assert roots.cache == Path("/Users/alice/Library/Caches/Storybook")
        assert roots.logs == Path("/Users/alice/Library/Logs/Storybook")

    def test_linux_honors_xdg_directories(self):
        env = {
            "XDG_CONFIG_HOME": "/xdg/config",
            "XDG_DATA_HOME": "/xdg/data",
            "XDG_CACHE_HOME": "/xdg/cache",
            "XDG_STATE_HOME": "/xdg/state",
        }
        roots = platform_roots("linux", environ=env, home=Path("/home/alice"))
        assert roots.config == Path("/xdg/config/storybook")
        assert roots.data == Path("/xdg/data/storybook")
        assert roots.cache == Path("/xdg/cache/storybook")
        assert roots.state == Path("/xdg/state/storybook")
        assert roots.logs == Path("/xdg/state/storybook/logs")

    def test_linux_falls_back_to_home_conventions(self):
        roots = platform_roots("linux", environ={}, home=Path("/home/alice"))
        assert roots.config == Path("/home/alice/.config/storybook")
        assert roots.data == Path("/home/alice/.local/share/storybook")
        assert roots.cache == Path("/home/alice/.cache/storybook")
        assert roots.state == Path("/home/alice/.local/state/storybook")

    def test_windows_uses_local_app_data(self):
        roots = platform_roots(
            "win32",
            environ={"LOCALAPPDATA": "/Users/Alice/AppData/Local"},
            home=Path("/Users/Alice"),
        )
        assert roots.data == Path("/Users/Alice/AppData/Local/Storybook")
        assert roots.config == Path(
            "/Users/Alice/AppData/Local/Storybook/config"
        )
        assert roots.cache == Path("/Users/Alice/AppData/Local/Storybook/cache")
        assert roots.logs == Path("/Users/Alice/AppData/Local/Storybook/logs")

    def test_storybook_home_is_an_explicit_isolation_boundary(self):
        roots = platform_roots(
            "linux",
            environ={"STORYBOOK_HOME": "/portable/storybook"},
            home=Path("/ignored"),
        )
        assert roots.config == Path("/portable/storybook/config")
        assert roots.data == Path("/portable/storybook/data")
        assert roots.cache == Path("/portable/storybook/cache")
        assert roots.state == Path("/portable/storybook/state")
        assert roots.logs == Path("/portable/storybook/logs")


class TestProfileRegistry:
    def test_default_profile_uses_random_uuid_and_local_only(self, tmp_path):
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        profile = registry.active_profile()
        uuid.UUID(profile.id)
        assert uuid.UUID(profile.id).version == 7
        assert profile.display_name == "default"
        assert profile.mode == "local"
        assert profile.sync_state == DEFAULT_SYNC_STATE

        payload = json.loads(registry.path.read_text(encoding="utf-8"))
        serialized = json.dumps(payload)
        assert str(tmp_path) not in serialized
        assert "hostname" not in serialized
        assert "email" not in serialized

    def test_profile_directories_are_separated(self, tmp_path):
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        profile = registry.active_profile()
        paths = registry.active_paths()
        assert paths.root == tmp_path / "data" / "profiles" / profile.id
        assert paths.database == paths.root / "db" / "memory.db"
        assert paths.index_dir == paths.root / "indexes"
        assert paths.cache_dir == tmp_path / "cache" / "profiles" / profile.id
        assert paths.log_dir == tmp_path / "logs" / "profiles" / profile.id
        for directory in (
            paths.root,
            paths.database_dir,
            paths.index_dir,
            paths.cache_dir,
            paths.log_dir,
        ):
            assert directory.is_dir()

    def test_create_isolated_and_switch_by_name(self, tmp_path):
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        original = registry.active_profile()
        isolated = registry.create_profile("client-a", mode="isolated")
        assert isolated.id != original.id
        assert registry.active_profile().id == original.id

        selected = registry.switch_profile("client-a")
        assert selected.id == isolated.id
        assert registry.active_profile().id == isolated.id
        assert registry.active_paths().database != registry.paths_for(original).database

    def test_duplicate_names_and_invalid_modes_are_rejected(self, tmp_path):
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        registry.ensure()
        with pytest.raises(ProfileError, match="已存在"):
            registry.create_profile("DEFAULT")
        with pytest.raises(ProfileError, match="mode"):
            registry.create_profile("bad", mode="account")

    def test_corrupt_registry_is_not_silently_replaced(self, tmp_path):
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        registry.path.parent.mkdir(parents=True)
        registry.path.write_text("{broken", encoding="utf-8")
        with pytest.raises(ProfileError, match="无法读取"):
            registry.ensure()
        assert registry.path.read_text(encoding="utf-8") == "{broken"

    def test_same_profile_path_from_any_cwd(self, tmp_path, monkeypatch):
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        first = tmp_path / "repo-a"
        second = tmp_path / "renamed-repo-b"
        first.mkdir()
        second.mkdir()

        monkeypatch.chdir(first)
        path_a = registry.active_paths().database
        monkeypatch.chdir(second)
        path_b = registry.active_paths().database

        assert path_a == path_b
        assert first not in path_a.parents
        assert second not in path_b.parents

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_registry_and_profile_directories_are_private(self, tmp_path):
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        paths = registry.active_paths()
        assert stat.S_IMODE(registry.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(registry.path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.database_dir.stat().st_mode) == 0o700


class TestProfileCLI:
    @pytest.fixture
    def isolated_config(self, tmp_path):
        old_registry = config.PROFILE_REGISTRY
        old_profile_id = config.PROFILE_ID
        registry = ProfileRegistry(
            tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
        )
        config.PROFILE_REGISTRY = registry
        config.refresh_profile()
        try:
            yield registry
        finally:
            config.PROFILE_REGISTRY = old_registry
            config.refresh_profile(old_profile_id)

    def test_show_and_sync_status_are_explicitly_local_only(self, isolated_config):
        runner = CliRunner()
        shown = runner.invoke(cli, ["profile", "show", "--json"])
        assert shown.exit_code == 0, shown.output
        profile = json.loads(shown.output)
        assert profile["mode"] == "local"
        assert profile["sync_state"] == "local_only"

        synced = runner.invoke(cli, ["sync", "status", "--json"])
        assert synced.exit_code == 0, synced.output
        status_payload = json.loads(synced.output)
        assert status_payload["sync_state"] == "local_only"
        assert status_payload["enabled"] is False

    def test_create_and_switch_isolated_profile(self, isolated_config):
        runner = CliRunner()
        created = runner.invoke(
            cli, ["profile", "create", "sensitive", "--mode", "isolated", "--switch"]
        )
        assert created.exit_code == 0, created.output
        assert config.PROFILE_MODE == "isolated"
        assert config.ACTIVE_PROFILE.display_name == "sensitive"
        assert config.DB_PATH.parent.parent.name == config.PROFILE_ID

    def test_switch_moves_query_diagnostics_to_active_profile(self, isolated_config):
        original_log = config.PERFORMANCE_LOG_PATH
        isolated = isolated_config.create_profile("diagnostics", mode="isolated")

        switched = CliRunner().invoke(cli, ["profile", "switch", isolated.id])

        assert switched.exit_code == 0, switched.output
        assert config.PERFORMANCE_LOG_PATH == (
            isolated_config.paths_for(isolated).log_dir / "query_performance.jsonl"
        )
        assert config.PERFORMANCE_LOG_PATH != original_log

        written = performance.record_query_diagnostic(
            request_id="profile-switch",
            mode="vector",
            latency_ms=performance.empty_latency(),
            result_count=0,
        )

        assert written is True
        assert config.PERFORMANCE_LOG_PATH.is_file()
        assert not original_log.exists()
