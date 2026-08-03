"""Stable contract shared by local Agent history adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class HistorySession:
    external_id: str
    raw_content: str
    problem_desc: str
    conclusion: str = ""
    code_snippets: str = "[]"
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ParseResult:
    sessions: tuple[HistorySession, ...]
    cursor: int
    fingerprint: str
    invalid_records: int = 0
    diagnostics: tuple[str, ...] = ()


class HistoryAdapter(Protocol):
    name: str
    display_name: str
    version: str

    def detect(self) -> dict: ...
    def discover(self) -> list[Path]: ...
    def parse(self, path: Path) -> ParseResult: ...
    def cursor(self, path: Path) -> int: ...
    def fingerprint(self, path: Path) -> str: ...
    def context(self, metadata: dict) -> dict: ...
    def diagnostics(self) -> list[dict]: ...

