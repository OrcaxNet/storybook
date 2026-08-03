"""Cline VS Code task history adapter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import sys
from pathlib import Path

from .base import HistorySession, ParseResult
from .common import (
    bounded_transcript, content_text, ensure_readable_dir, file_fingerprint
)


def _roots(home: Path) -> list[Path]:
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = home / ".config"
    return [
        base / name / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "tasks"
        for name in ("Code", "Code - Insiders", "VSCodium")
    ]


class ClineAdapter:
    name = "cline"
    display_name = "Cline"
    version = "1.0/cline-messageparam"

    def __init__(self, roots: list[Path] | None = None):
        self.roots = list(roots) if roots is not None else _roots(Path.home())

    def detect(self) -> dict:
        for root in self.roots:
            ensure_readable_dir(root)
        available = any(root.is_dir() and any(root.glob("*/api_conversation_history.json")) for root in self.roots)
        return {"available": available, "status": "ready" if available else "missing"}

    def discover(self) -> list[Path]:
        return sorted(path for root in self.roots for path in root.glob("*/api_conversation_history.json"))

    def parse(self, path: Path) -> ParseResult:
        raw = path.read_bytes()
        fingerprint = file_fingerprint(path)
        try:
            messages = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ParseResult((), 0, fingerprint, 1, ("SB_SOURCE_JSON_INVALID",))
        if not isinstance(messages, list):
            return ParseResult((), len(raw), fingerprint, 1, ("SB_SOURCE_SCHEMA_UNKNOWN",))
        lines: list[str] = []
        first_user = ""
        last_assistant = ""
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                continue
            role = message["role"]
            text = content_text(message.get("content"))
            if not text:
                continue
            # Cline appends machine-generated environment blocks to user turns.
            text = text.split("<environment_details>", 1)[0].strip()
            if not text:
                continue
            lines.append(f"[{role}] {text}")
            if role == "user" and not first_user:
                first_user = text
            if role == "assistant":
                last_assistant = text
        if not first_user:
            return ParseResult((), len(raw), fingerprint, 1, ("SB_SOURCE_SCHEMA_UNKNOWN",))
        metadata_path = path.with_name("task_metadata.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = metadata if isinstance(metadata, dict) else {}
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            metadata = {}
        metadata["captured_at"] = datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat().replace("+00:00", "Z")
        session = HistorySession(
            external_id=path.parent.name,
            raw_content=bounded_transcript(lines),
            problem_desc=first_user[:200],
            conclusion=last_assistant[:300],
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
                "type": "cline", "adapter": self.name,
                "adapter_version": self.version,
                "integration_mode": "log_import",
            },
            "workspace": {
                "path": metadata.get("workspace") or metadata.get("cwd"),
                "project_label": metadata.get("projectName"),
            },
            "runtime": {"kind": "unknown"},
            "captured_at": metadata.get("captured_at"),
            "provenance": {
                "tool.type": "detected",
                "tool.adapter": "detected",
                "tool.adapter_version": "detected",
                "tool.integration_mode": "detected",
                "workspace.path": "reported",
                "workspace.project_label": "reported",
                "runtime.kind": "unknown",
                "captured_at": "detected",
            },
        }

    def diagnostics(self) -> list[dict]:
        return [{"code": "SB_SOURCE_CLINE_HISTORY", "adapter_version": self.version}]
