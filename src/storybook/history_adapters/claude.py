"""Claude Code adapter on the unified history contract."""
from __future__ import annotations

from pathlib import Path

from .. import collector, config
from .base import HistorySession, ParseResult
from .common import complete_jsonl, ensure_readable_dir


class ClaudeAdapter:
    name = "claude"
    session_source = collector.CLAUDE_SOURCE
    display_name = "Claude Code"
    version = "1.0/claude-jsonl"

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else config.CLAUDE_PROJECTS_PATH

    def detect(self) -> dict:
        ensure_readable_dir(self.root)
        available = self.root.is_dir() and any(self.root.glob("*/*.jsonl"))
        return {"available": available, "status": "ready" if available else "missing"}

    def discover(self) -> list[Path]:
        return sorted(self.root.glob("*/*.jsonl"))

    def parse(self, path: Path) -> ParseResult:
        _records, cursor, invalid, fingerprint = complete_jsonl(path)
        parsed = collector._parse_claude_jsonl(path, path.stem)
        if not parsed:
            return ParseResult((), cursor, fingerprint, invalid + 1, ("SB_SOURCE_SCHEMA_UNKNOWN",))
        parsed["context"]["tool"]["adapter"] = self.name
        parsed["context"]["tool"]["adapter_version"] = self.version
        parsed["context"]["provenance"]["tool.adapter"] = "detected"
        parsed["context"]["provenance"]["tool.adapter_version"] = "detected"
        return ParseResult(
            (HistorySession(
                external_id=path.stem,
                raw_content=parsed["raw_content"],
                problem_desc=parsed["problem_desc"],
                conclusion=parsed["conclusion"],
                code_snippets=parsed["code_snippets"],
                context=parsed["context"],
            ),),
            cursor,
            fingerprint,
            invalid,
        )

    def cursor(self, path: Path) -> int:
        return complete_jsonl(path)[1]

    def fingerprint(self, path: Path) -> str:
        return complete_jsonl(path)[3]

    def context(self, metadata: dict) -> dict:
        return metadata

    def diagnostics(self) -> list[dict]:
        return [{"code": "SB_SOURCE_CLAUDE_JSONL", "adapter_version": self.version}]
