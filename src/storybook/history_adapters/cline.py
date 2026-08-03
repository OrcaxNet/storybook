"""Cline VS Code task history adapter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .base import HistorySession, ParseResult
from .common import bounded_transcript, content_text, file_fingerprint


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
        session = HistorySession(
            external_id=path.parent.name,
            raw_content=bounded_transcript(lines),
            problem_desc=first_user[:200],
            conclusion=last_assistant[:300],
            context=self.context({}),
        )
        return ParseResult((session,), len(raw), fingerprint)

    def cursor(self, path: Path) -> int:
        return path.stat().st_size

    def fingerprint(self, path: Path) -> str:
        return file_fingerprint(path)

    def context(self, metadata: dict) -> dict:
        return {
            "tool": {"type": "cline", "integration_mode": "log_import"},
            "provenance": {
                "tool.type": "detected",
                "tool.integration_mode": "detected",
            },
        }

    def diagnostics(self) -> list[dict]:
        return [{"code": "SB_SOURCE_CLINE_HISTORY", "adapter_version": self.version}]

