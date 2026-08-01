"""Codex user ``config.toml`` adapter。"""
from __future__ import annotations

import re
import shutil
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .base import (
    AdapterContext,
    AdapterError,
    AdapterPlan,
    AgentAdapter,
    atomic_write,
    backup_file,
    sha256_bytes,
)


_BEGIN = "# >>> storybook setup managed; do not edit this block >>>"
_END = "# <<< storybook setup managed <<<"
_SECTION = re.compile(r"(?m)^\s*\[([^\]]+)]\s*(?:#.*)?$")
_MISSING = object()


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_block(context: AdapterContext) -> str:
    args = [*context.launcher.args, "mcp"]
    args_text = ", ".join(_quote(item) for item in args)
    return (
        f"{_BEGIN}\n"
        "[mcp_servers.storybook]\n"
        f"command = {_quote(context.launcher.command)}\n"
        f"args = [{args_text}]\n"
        "startup_timeout_sec = 20\n"
        "tool_timeout_sec = 60\n"
        f"{_END}\n"
    )


def _target_node(context: AdapterContext) -> dict[str, Any]:
    return {
        "command": context.launcher.command,
        "args": [*context.launcher.args, "mcp"],
        "startup_timeout_sec": 20,
        "tool_timeout_sec": 60,
    }


def _managed_range(text: str) -> tuple[int, int] | None:
    start = text.find(_BEGIN)
    if start < 0:
        return None
    end_marker = text.find(_END, start)
    if end_marker < 0:
        raise AdapterError(
            "SB_SETUP_CONFIG_INVALID", "Codex config 含不完整的 Storybook managed block"
        )
    end = end_marker + len(_END)
    if end < len(text) and text[end] == "\n":
        end += 1
    return start, end


def _storybook_section_ranges(text: str) -> list[tuple[int, int]]:
    matches = list(_SECTION.finditer(text))
    ranges: list[tuple[int, int]] = []
    for index, match in enumerate(matches):
        normalized = match.group(1).replace('"', "").replace("'", "").replace(" ", "")
        if normalized == "mcp_servers.storybook" or normalized.startswith(
            "mcp_servers.storybook."
        ):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            ranges.append((match.start(), end))
    return ranges


def _without_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    for start, end in reversed(ranges):
        text = text[:start] + text[end:]
    return text.rstrip() + ("\n" if text.strip() else "")


def _parse(text: str, path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(text) if text.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        raise AdapterError("SB_SETUP_CONFIG_INVALID", f"无法解析 {path}: {exc}") from exc


def _current_node(parsed: Mapping[str, Any], path: Path) -> Any:
    servers = parsed.get("mcp_servers", {})
    if not isinstance(servers, dict):
        raise AdapterError(
            "SB_SETUP_CONFIG_INVALID", f"{path} 的 mcp_servers 必须是 table"
        )
    current = servers.get("storybook", _MISSING)
    if current is not _MISSING and not isinstance(current, dict):
        raise AdapterError(
            "SB_SETUP_CONFIG_INVALID",
            f"{path} 的 mcp_servers.storybook 必须是 table",
        )
    return current


class CodexAdapter(AgentAdapter):
    name = "codex"
    display_name = "Codex"

    def config_path(self, context: AdapterContext) -> Path:
        codex_home = context.environ.get("CODEX_HOME", "").strip()
        root = Path(codex_home).expanduser() if codex_home else context.home / ".codex"
        return root / "config.toml"

    def detected(self, context: AdapterContext) -> bool:
        path = self.config_path(context)
        return path.exists() or path.parent.exists() or shutil.which("codex") is not None

    def _read(self, context: AdapterContext) -> tuple[Path, str, dict[str, Any]]:
        path = self.config_path(context)
        try:
            text = path.read_text(encoding="utf-8") if path.exists() else ""
        except (OSError, UnicodeError) as exc:
            raise AdapterError("SB_SETUP_CONFIG_INVALID", f"无法读取 {path}: {exc}") from exc
        return path, text, _parse(text, path)

    def plan(self, context: AdapterContext, *, selected: bool) -> AdapterPlan:
        path, text, parsed = self._read(context)
        current = _current_node(parsed, path)
        target = _target_node(context)
        managed_range = _managed_range(text)
        needs_change = current != target
        if managed_range is not None:
            current_block = text[managed_range[0]:managed_range[1]]
            needs_change = needs_change or current_block != _render_block(context)
        changed = selected and needs_change
        return AdapterPlan(
            adapter=self.name,
            display_name=self.display_name,
            detected=self.detected(context),
            selected=selected,
            changed=changed,
            targets=(str(path),),
            summary=(
                "merge mcp_servers.storybook"
                if changed
                else ("already configured" if selected else "not selected")
            ),
        )

    def apply(self, context: AdapterContext, backup_dir: Path) -> dict[str, Any]:
        path, text, parsed = self._read(context)
        current = _current_node(parsed, path)
        target = _target_node(context)
        block = _render_block(context)
        managed_range = _managed_range(text)
        previous_blocks = ""
        if managed_range:
            current_block = text[managed_range[0]:managed_range[1]]
            if current == target and current_block == block:
                return {
                    "adapter": self.name,
                    "changed": False,
                    "files": [],
                    "previous_blocks": "",
                    "managed_block": block,
                }
            previous_blocks = ""
            base = text[:managed_range[0]] + text[managed_range[1]:]
        else:
            # 用户已有语义相同的配置时不夺取 ownership，也不为了 marker 改写文件。
            if current == target:
                return {
                    "adapter": self.name,
                    "changed": False,
                    "files": [],
                    "previous_blocks": "",
                    "managed_block": block,
                }
            ranges = _storybook_section_ranges(text)
            previous_blocks = "\n".join(text[start:end].rstrip() for start, end in ranges)
            base = _without_ranges(text, ranges)

        record = backup_file(path, backup_dir, self.name)
        updated = base.rstrip()
        if updated:
            updated += "\n\n"
        updated += block
        # 在写入前再次解析，保证不会生成损坏的 TOML。
        _parse(updated, path)
        data = updated.encode("utf-8")
        atomic_write(path, data)
        record["after_sha256"] = sha256_bytes(data)
        return {
            "adapter": self.name,
            "changed": True,
            "files": [record],
            "previous_blocks": previous_blocks,
            "managed_block": block,
        }

    def uninstall(self, context: AdapterContext, state: Mapping[str, Any]) -> dict[str, Any]:
        path, text, _ = self._read(context)
        managed_range = _managed_range(text)
        if managed_range is None:
            return {"adapter": self.name, "changed": False, "status": "absent"}
        current = text[managed_range[0]:managed_range[1]]
        if current != state.get("managed_block"):
            return {"adapter": self.name, "changed": False, "status": "drifted"}
        restored = text[:managed_range[0]] + text[managed_range[1]:]
        previous = str(state.get("previous_blocks") or "").strip()
        restored = restored.rstrip()
        if previous:
            restored += "\n\n" + previous
        if restored:
            restored += "\n"
        _parse(restored, path)
        atomic_write(path, restored.encode("utf-8"))
        return {"adapter": self.name, "changed": True, "status": "restored"}

    def verify(self, context: AdapterContext) -> tuple[bool, str]:
        path, _, parsed = self._read(context)
        current = _current_node(parsed, path)
        expected = _target_node(context)
        return current == expected, f"{path}: storybook MCP {'ready' if current == expected else 'missing or changed'}"
