"""Gemini CLI auto-session adapter (0.38+ chat JSON)."""
from __future__ import annotations

import json
from pathlib import Path

from .base import HistorySession, ParseResult
from .common import (
    bounded_transcript, content_text, ensure_readable_dir, file_fingerprint
)


class GeminiAdapter:
    name = "gemini"
    display_name = "Gemini CLI"
    version = "1.0/gemini-cli-0.38+"

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else Path.home() / ".gemini"

    def detect(self) -> dict:
        ensure_readable_dir(self.root)
        ensure_readable_dir(self.root / "tmp")
        available = self.root.is_dir() and any(self.root.glob("tmp/*/chats/session-*.json"))
        return {"available": available, "status": "ready" if available else "missing"}

    def discover(self) -> list[Path]:
        return sorted(self.root.glob("tmp/*/chats/session-*.json"))

    def parse(self, path: Path) -> ParseResult:
        raw = path.read_bytes()
        fingerprint = file_fingerprint(path)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ParseResult((), 0, fingerprint, 1, ("SB_SOURCE_JSON_INVALID",))
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            return ParseResult((), len(raw), fingerprint, 1, ("SB_SOURCE_SCHEMA_UNKNOWN",))
        lines: list[str] = []
        first_user = ""
        last_model = ""
        for message in data["messages"]:
            if not isinstance(message, dict):
                continue
            role = message.get("type") or message.get("role")
            if role == "model":
                role = "assistant"
            if role not in {"user", "assistant", "gemini"}:
                continue
            if role == "gemini":
                role = "assistant"
            text = content_text(message.get("content"))
            if not text:
                continue
            lines.append(f"[{role}] {text}")
            if role == "user" and not first_user:
                first_user = text
            if role == "assistant":
                last_model = text
        external_id = data.get("sessionId")
        if not isinstance(external_id, str) or not first_user:
            return ParseResult((), len(raw), fingerprint, 1, ("SB_SOURCE_SCHEMA_UNKNOWN",))
        project_root_file = path.parent.parent / ".project_root"
        try:
            workspace_path = project_root_file.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError, UnicodeError):
            workspace_path = None
        metadata = dict(data)
        metadata["workspace_path"] = workspace_path
        metadata["project_label"] = data.get("projectHash") or path.parent.parent.name
        session = HistorySession(
            external_id=external_id,
            raw_content=bounded_transcript(lines),
            problem_desc=(data.get("summary") or first_user)[:200],
            conclusion=last_model[:300],
            context=self.context(metadata),
        )
        return ParseResult((session,), len(raw), fingerprint)

    def cursor(self, path: Path) -> int:
        return path.stat().st_size

    def fingerprint(self, path: Path) -> str:
        return file_fingerprint(path)

    def context(self, metadata: dict) -> dict:
        return {
            "tool": {
                "type": "gemini_cli", "adapter": self.name,
                "adapter_version": self.version,
                "integration_mode": "log_import",
            },
            "session": {"started_at": metadata.get("startTime")},
            "workspace": {
                "path": metadata.get("workspace_path"),
                "project_label": metadata.get("project_label"),
            },
            "runtime": {"kind": "unknown"},
            "provenance": {
                "tool.type": "detected",
                "tool.adapter": "detected",
                "tool.adapter_version": "detected",
                "tool.integration_mode": "detected",
                "session.started_at": "reported",
                "workspace.path": "reported",
                "workspace.project_label": "inferred",
                "runtime.kind": "unknown",
            },
        }

    def diagnostics(self) -> list[dict]:
        return [{"code": "SB_SOURCE_GEMINI_CHAT_JSON", "adapter_version": self.version}]
