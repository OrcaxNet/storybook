"""Cursor workspaceStorage adapter on the unified history contract."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .. import collector, config, context as context_module
from .base import HistorySession, ParseResult
from .common import ensure_readable_dir, file_fingerprint


class CursorAdapter:
    name = "cursor"
    display_name = "Cursor"
    version = "1.0/cursor-state-vscdb"

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else config.CURSOR_STORAGE_PATH

    def detect(self) -> dict:
        ensure_readable_dir(self.root)
        available = self.root.is_dir() and any(self.root.glob("*/state.vscdb"))
        return {"available": available, "status": "ready" if available else "missing"}

    def discover(self) -> list[Path]:
        return sorted(self.root.glob("*/state.vscdb"))

    def parse(self, path: Path) -> ParseResult:
        fingerprint = file_fingerprint(path)
        workspace_path = collector._cursor_workspace_path(path)
        adapter_context = context_module.normalize_envelope({
            "tool": {
                "type": "cursor", "adapter": self.name,
                "adapter_version": self.version,
                "integration_mode": "log_import",
            },
            "workspace": {"path": workspace_path} if workspace_path else {},
            "provenance": {
                "tool.type": "detected",
                "tool.adapter": "detected",
                "tool.adapter_version": "detected",
                "tool.integration_mode": "detected",
                "workspace.path": "detected",
            },
        })
        sessions: list[HistorySession] = []
        invalid = 0
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            rows = db.execute(
                """SELECT key, value FROM ItemTable
                   WHERE key LIKE '%aiService%' OR key LIKE '%aichat%'
                      OR key LIKE '%cursor%chat%'"""
            ).fetchall()
            for row in rows:
                try:
                    value = row["value"]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    data = json.loads(value)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    invalid += 1
                    continue
                parsed = collector._parse_cursor_conversation(
                    data, row["key"], adapter_context=adapter_context
                )
                for index, item in enumerate(parsed):
                    sessions.append(HistorySession(
                        external_id=f"{path.parent.name}:{row['key']}:{index}",
                        raw_content=item["raw_content"],
                        problem_desc=item["problem_desc"],
                        conclusion=item["conclusion"],
                        code_snippets=item["code_snippets"],
                        context=item["context"],
                    ))
        finally:
            db.close()
        return ParseResult(tuple(sessions), path.stat().st_size, fingerprint, invalid)

    def cursor(self, path: Path) -> int:
        return path.stat().st_size

    def fingerprint(self, path: Path) -> str:
        return file_fingerprint(path)

    def context(self, metadata: dict) -> dict:
        return metadata

    def diagnostics(self) -> list[dict]:
        return [{"code": "SB_SOURCE_CURSOR_VSCDB", "adapter_version": self.version}]
