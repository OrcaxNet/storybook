"""Codex CLI/app rollout adapter for active and archived local sessions."""
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

from .base import HistorySession, IncrementalParseResult, ParseResult
from .common import (
    bounded_transcript,
    complete_jsonl,
    content_text,
    ensure_readable_dir,
    incremental_jsonl,
)


class CodexAdapter:
    name = "codex"
    display_name = "Codex"
    version = "1.1/codex-rollout"

    def __init__(self, root: Path | None = None):
        configured = os.getenv("CODEX_HOME", "").strip()
        self.root = Path(root or configured or (Path.home() / ".codex")).expanduser()

    def detect(self) -> dict:
        ensure_readable_dir(self.root)
        for directory in self._directories():
            ensure_readable_dir(directory)
        available = next(self._paths(), None) is not None
        return {"available": available, "status": "ready" if available else "missing"}

    def discover(self) -> list[Path]:
        return sorted(self._paths())

    def _directories(self) -> tuple[Path, Path]:
        return self.root / "sessions", self.root / "archived_sessions"

    def _paths(self) -> Iterator[Path]:
        for directory in self._directories():
            yield from (path for path in directory.glob("**/*.jsonl") if path.is_file())

    def parse(self, path: Path) -> ParseResult:
        records, cursor, invalid, fingerprint = complete_jsonl(path)
        meta: dict = {}
        for record in records:
            if record.get("type") == "session_meta" and isinstance(record.get("payload"), dict):
                meta = record["payload"]
                break
        lines, first_user, last_assistant, record_invalid = self._parse_records(records)
        invalid += record_invalid
        external_id = meta.get("id") or meta.get("session_id")
        if not isinstance(external_id, str) or not external_id:
            return ParseResult((), cursor, fingerprint, invalid + 1, ("SB_SOURCE_SCHEMA_UNKNOWN",))
        diagnostics = ("SB_SOURCE_JSONL_INVALID",) if invalid else ()
        if not first_user:
            # An initialized/aborted rollout may contain only injected context.
            # Keep its cursor eligible for a later real request, without importing noise.
            return ParseResult((), cursor, fingerprint, invalid, diagnostics)
        session = HistorySession(
            external_id=external_id,
            raw_content=bounded_transcript(lines),
            problem_desc=first_user[:200],
            conclusion=last_assistant[:300],
            context=self.context(meta),
        )
        return ParseResult((session,), cursor, fingerprint, invalid, diagnostics)

    def parse_incremental(
        self, path: Path, offset: int, previous_fingerprint: str
    ) -> IncrementalParseResult:
        records, cursor, invalid, fingerprint = incremental_jsonl(
            path, offset, previous_fingerprint
        )
        lines, _first_user, last_assistant, record_invalid = self._parse_records(records)
        invalid += record_invalid
        diagnostics = ("SB_SOURCE_JSONL_INVALID",) if invalid else ()
        return IncrementalParseResult(
            tuple(lines), last_assistant[:300], cursor, fingerprint,
            invalid, diagnostics,
        )

    @staticmethod
    def _parse_records(records: list[dict]) -> tuple[list[str], str, str, int]:
        lines: list[str] = []
        first_user = ""
        first_agent = ""
        last_assistant = ""
        invalid = 0
        for record in records:
            kind = record.get("type")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                invalid += 1
                continue
            if kind == "session_meta":
                continue
            if kind == "response_item":
                item_type = payload.get("type")
                if item_type == "message" and payload.get("role") in {"user", "assistant"}:
                    text = content_text(payload.get("content"))
                    if not text:
                        continue
                    role = payload["role"]
                    if role == "user":
                        text = _user_text(text)
                    elif payload.get("channel") == "analysis":
                        continue
                    if not text:
                        continue
                    lines.append(f"[{role}] {text}")
                    if role == "user" and not first_user:
                        first_user = text
                    if role == "assistant" and (
                        payload.get("phase") in {None, "final_answer"}
                        and payload.get("channel") in {None, "final"}
                    ):
                        last_assistant = text
                elif item_type in {"function_call", "custom_tool_call"}:
                    name = payload.get("name")
                    if isinstance(name, str) and name:
                        lines.append(f"[tool] {name}")
                elif item_type == "agent_message":
                    text = content_text(payload.get("content"))
                    if text:
                        lines.append(f"[agent] {text}")
                        if not first_agent:
                            first_agent = text
            elif kind == "event_msg" and payload.get("type") in {
                "turn_aborted", "stream_error", "task_complete"
            }:
                lines.append(f"[event] {payload['type']}")
        return lines, first_user or first_agent, last_assistant, invalid

    def cursor(self, path: Path) -> int:
        return complete_jsonl(path)[1]

    def fingerprint(self, path: Path) -> str:
        return complete_jsonl(path)[3]

    def context(self, metadata: dict) -> dict:
        return {
            "tool": {
                "type": "codex",
                "version": metadata.get("cli_version"),
                "adapter": self.name,
                "adapter_version": self.version,
                "integration_mode": "log_import",
            },
            "session": {"started_at": metadata.get("timestamp")},
            "workspace": {"path": metadata.get("cwd")},
            "provenance": {
                "tool.type": "detected",
                "tool.version": "reported",
                "tool.adapter": "detected",
                "tool.adapter_version": "detected",
                "tool.integration_mode": "detected",
                "session.started_at": "reported",
                "workspace.path": "reported",
            },
        }

    def diagnostics(self) -> list[dict]:
        return [{"code": "SB_SOURCE_CODEX_ROLLOUT", "adapter_version": self.version}]


_INJECTED_PREAMBLE = re.compile(
    r"\A(?:<(environment_context|environment_info|recommended_plugins)>.*?</\1>"
    r"|# AGENTS\.md instructions for [^\n]*\n.*?</INSTRUCTIONS>)\s*",
    re.DOTALL,
)


def _user_text(text: str) -> str:
    """Remove known injected preambles, preserving a request after the block."""
    while match := _INJECTED_PREAMBLE.match(text):
        text = text[match.end():].strip()
    return text
