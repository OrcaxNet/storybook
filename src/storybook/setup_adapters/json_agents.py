"""Claude Code 与 Cursor 的 JSON 配置 adapters。"""
from __future__ import annotations

import copy
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .base import (
    AdapterContext,
    AdapterError,
    AdapterPlan,
    AgentAdapter,
    atomic_write,
    backup_file,
    json_bytes,
    read_json_object,
    sha256_bytes,
)


_MISSING = object()


class JsonMCPAdapter(AgentAdapter):
    executable_names: tuple[str, ...]

    def config_path(self, context: AdapterContext) -> Path:
        raise NotImplementedError

    def detected(self, context: AdapterContext) -> bool:
        path = self.config_path(context)
        return (
            path.exists()
            or path.parent.exists()
            or any(shutil.which(name) for name in self.executable_names)
        )

    def managed_node(self, context: AdapterContext) -> dict[str, Any]:
        return context.launcher.mcp_node()

    def _current(self, context: AdapterContext) -> tuple[Path, dict[str, Any], Any]:
        path = self.config_path(context)
        payload = read_json_object(path)
        servers = payload.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise AdapterError(
                "SB_SETUP_CONFIG_INVALID", f"{path} 的 mcpServers 必须是 object"
            )
        return path, payload, servers.get("storybook", _MISSING)

    def plan(self, context: AdapterContext, *, selected: bool) -> AdapterPlan:
        path, _, current = self._current(context)
        changed = selected and current != self.managed_node(context)
        return AdapterPlan(
            adapter=self.name,
            display_name=self.display_name,
            detected=self.detected(context),
            selected=selected,
            changed=changed,
            targets=(str(path),),
            summary=(
                "merge mcpServers.storybook"
                if changed
                else ("already configured" if selected else "not selected")
            ),
        )

    def apply(self, context: AdapterContext, backup_dir: Path) -> dict[str, Any]:
        path, payload, previous = self._current(context)
        managed = self.managed_node(context)
        if previous == managed:
            return {
                "adapter": self.name,
                "changed": False,
                "files": [],
                "previous": {"present": previous is not _MISSING, "value": previous},
                "managed": managed,
            }
        record = backup_file(path, backup_dir, self.name)
        updated = copy.deepcopy(payload)
        updated.setdefault("mcpServers", {})["storybook"] = managed
        data = json_bytes(updated)
        atomic_write(path, data)
        record["after_sha256"] = sha256_bytes(data)
        return {
            "adapter": self.name,
            "changed": True,
            "files": [record],
            "previous": {
                "present": previous is not _MISSING,
                "value": None if previous is _MISSING else previous,
            },
            "managed": managed,
        }

    def uninstall(self, context: AdapterContext, state: Mapping[str, Any]) -> dict[str, Any]:
        path, payload, current = self._current(context)
        managed = state.get("managed")
        if current is _MISSING:
            return {"adapter": self.name, "changed": False, "status": "absent"}
        if current != managed:
            return {"adapter": self.name, "changed": False, "status": "drifted"}
        servers = payload["mcpServers"]
        previous = state.get("previous", {})
        if previous.get("present"):
            servers["storybook"] = previous.get("value")
        else:
            servers.pop("storybook", None)
            if not servers:
                payload.pop("mcpServers", None)
        atomic_write(path, json_bytes(payload))
        return {"adapter": self.name, "changed": True, "status": "restored"}

    def verify(self, context: AdapterContext) -> tuple[bool, str]:
        path, _, current = self._current(context)
        if current == self.managed_node(context):
            return True, f"{path}: storybook MCP ready"
        return False, f"{path}: storybook MCP node missing or changed"


class CursorAdapter(JsonMCPAdapter):
    name = "cursor"
    display_name = "Cursor"
    executable_names = ("cursor", "cursor-agent")

    def config_path(self, context: AdapterContext) -> Path:
        return context.home / ".cursor" / "mcp.json"


class ClaudeCodeAdapter(JsonMCPAdapter):
    name = "claude"
    display_name = "Claude Code"
    executable_names = ("claude",)

    def config_path(self, context: AdapterContext) -> Path:
        return context.home / ".claude.json"

    def settings_path(self, context: AdapterContext) -> Path:
        return context.home / ".claude" / "settings.json"

    def managed_hook(self, context: AdapterContext) -> dict[str, Any]:
        parts = [context.launcher.command, *context.launcher.args, "prime"]
        prefix = subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)
        return {
            "matcher": "startup|resume|clear|compact",
            "hooks": [
                {
                    "type": "command",
                    "command": f'{prefix} --cwd "$CLAUDE_PROJECT_DIR"',
                    "timeout": 5,
                }
            ],
        }

    def _session_hooks(
        self, settings: dict[str, Any], settings_path: Path
    ) -> list[Any]:
        hooks = settings.get("hooks", {})
        if not isinstance(hooks, dict):
            raise AdapterError(
                "SB_SETUP_CONFIG_INVALID", f"{settings_path} 的 hooks 必须是 object"
            )
        session_hooks = hooks.get("SessionStart", [])
        if not isinstance(session_hooks, list):
            raise AdapterError(
                "SB_SETUP_CONFIG_INVALID",
                f"{settings_path} 的 hooks.SessionStart 必须是 array",
            )
        return session_hooks

    def plan(self, context: AdapterContext, *, selected: bool) -> AdapterPlan:
        base = super().plan(context, selected=selected)
        settings_path = self.settings_path(context)
        settings = read_json_object(settings_path)
        session_hooks = self._session_hooks(settings, settings_path)
        hook_changed = selected and self.managed_hook(context) not in session_hooks
        return AdapterPlan(
            adapter=base.adapter,
            display_name=base.display_name,
            detected=base.detected,
            selected=selected,
            changed=base.changed or hook_changed,
            targets=(str(self.config_path(context)), str(settings_path)),
            summary=(
                "merge user MCP + SessionStart recall hook"
                if base.changed or hook_changed
                else "already configured"
            ),
        )

    def apply(self, context: AdapterContext, backup_dir: Path) -> dict[str, Any]:
        settings_path = self.settings_path(context)
        settings = read_json_object(settings_path)
        session_hooks = self._session_hooks(settings, settings_path)
        # 两个配置文件先全部校验，再开始任何写入；第二步失败时回滚第一步。
        state = super().apply(context, backup_dir)
        managed_hook = self.managed_hook(context)
        state["managed_hook"] = managed_hook
        state["hook_added"] = managed_hook not in session_hooks
        try:
            if managed_hook not in session_hooks:
                record = backup_file(settings_path, backup_dir, f"{self.name}-hook")
                updated = copy.deepcopy(settings)
                updated.setdefault("hooks", {}).setdefault("SessionStart", []).append(
                    managed_hook
                )
                data = json_bytes(updated)
                atomic_write(settings_path, data)
                record["after_sha256"] = sha256_bytes(data)
                state.setdefault("files", []).append(record)
                state["changed"] = True
        except Exception:
            super().uninstall(context, state)
            raise
        return state

    def uninstall(self, context: AdapterContext, state: Mapping[str, Any]) -> dict[str, Any]:
        result = super().uninstall(context, state)
        if not state.get("hook_added"):
            return result
        path = self.settings_path(context)
        settings = read_json_object(path)
        hooks = settings.get("hooks", {})
        session_hooks = hooks.get("SessionStart", []) if isinstance(hooks, dict) else []
        managed = state.get("managed_hook")
        if isinstance(session_hooks, list) and managed in session_hooks:
            session_hooks.remove(managed)
            if not session_hooks:
                hooks.pop("SessionStart", None)
            if not hooks:
                settings.pop("hooks", None)
            atomic_write(path, json_bytes(settings))
            result["changed"] = True
            result["hook_status"] = "restored"
        else:
            result["hook_status"] = "absent_or_drifted"
            result["status"] = "drifted"
        return result

    def verify(self, context: AdapterContext) -> tuple[bool, str]:
        mcp_ok, message = super().verify(context)
        settings = read_json_object(self.settings_path(context))
        hooks = settings.get("hooks", {})
        session_hooks = hooks.get("SessionStart", []) if isinstance(hooks, dict) else []
        hook_ok = isinstance(session_hooks, list) and self.managed_hook(context) in session_hooks
        return mcp_ok and hook_ok, f"{message}; SessionStart hook={'ready' if hook_ok else 'missing'}"
