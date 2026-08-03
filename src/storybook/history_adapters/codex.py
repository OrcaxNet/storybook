"""Codex CLI/app rollout adapter (verified against codex-cli 0.145)."""
from __future__ import annotations

import os
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
    version = "1.0/codex-rollout-0.145"

    def __init__(self, root: Path | None = None):
        configured = os.getenv("CODEX_HOME")
        self.root = Path(root) if root else Path(configured or (Path.home() / ".codex"))

    def detect(self) -> dict:
        ensure_readable_dir(self.root)
        ensure_readable_dir(self.root / "sessions")
        available = self.root.is_dir() and any(self.root.glob("sessions/**/*.jsonl"))
        return {"available": available, "status": "ready" if available else "missing"}

    def discover(self) -> list[Path]:
        try:
            return sorted(self.root.glob("sessions/**/*.jsonl"))
        except OSError:
            return []

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
        if not isinstance(external_id, str) or not external_id or not first_user:
            return ParseResult((), cursor, fingerprint, invalid + 1, ("SB_SOURCE_SCHEMA_UNKNOWN",))
        diagnostics = ("SB_SOURCE_JSONL_INVALID",) if invalid else ()
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
                    lines.append(f"[{role}] {text}")
                    if role == "user" and not first_user:
                        first_user = text
                    if role == "assistant":
                        last_assistant = text
                elif item_type in {"function_call", "custom_tool_call"}:
                    name = payload.get("name")
                    if isinstance(name, str) and name:
                        lines.append(f"[tool] {name}")
            elif kind == "event_msg" and payload.get("type") in {
                "turn_aborted", "stream_error", "task_complete"
            }:
                lines.append(f"[event] {payload['type']}")
        return lines, first_user, last_assistant, invalid

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
