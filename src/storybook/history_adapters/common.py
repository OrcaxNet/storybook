"""Privacy-safe parsing helpers for history adapters."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .. import context as context_module

_ASSIGNED_SECRET = re.compile(
    r"(?i)['\"]?[A-Za-z0-9_.-]*"
    r"(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token)"
    r"['\"]?"
    r"\s*[:=]\s*(?:bearer\s+)?(?:['\"][^'\"\r\n]+['\"]|[^\s,;}\]\r\n]+)"
)
_AUTHORIZATION_HEADER = re.compile(
    r"(?im)(?P<prefix>\b(?:proxy-)?authorization\s*:\s*)[^\r\n]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


class PrefixMismatchError(ValueError):
    """The bytes committed by a checkpoint no longer match the source file."""


def redact_text(value: Any) -> str:
    text = value if isinstance(value, str) else ""
    text = _AUTHORIZATION_HEADER.sub(r"\g<prefix>[REDACTED]", text)
    text = _ASSIGNED_SECRET.sub("[REDACTED]", text)
    text = _BEARER_SECRET.sub("[REDACTED]", text)
    return _OPENAI_SECRET.sub("[REDACTED]", text)


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return redact_text(content).strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"text", "input_text", "output_text"}:
            text = redact_text(item.get("text", "")).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def bounded_transcript(lines: list[str], cap: int = 6000) -> str:
    text = "\n".join(line for line in lines if line)
    if len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + "\n...\n" + text[-half:]


def complete_jsonl(path: Path) -> tuple[list[dict], int, int, str]:
    """Read complete JSONL records only; tolerate a concurrently-written tail."""

    raw = path.read_bytes()
    complete_len = len(raw) if raw.endswith(b"\n") else raw.rfind(b"\n") + 1
    complete = raw[:complete_len]
    records: list[dict] = []
    invalid = 0
    for line in complete.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid += 1
            continue
        if isinstance(item, dict):
            records.append(item)
        else:
            invalid += 1
    return records, complete_len, invalid, hashlib.sha256(complete).hexdigest()


def incremental_jsonl(
    path: Path, offset: int, previous_fingerprint: str
) -> tuple[list[dict], int, int, str]:
    """Verify the committed prefix, then parse only complete appended records."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = offset
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise PrefixMismatchError("checkpoint cursor exceeds file length")
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != previous_fingerprint:
            raise PrefixMismatchError("committed source prefix changed")
        raw = handle.read()
    complete_len = len(raw) if raw.endswith(b"\n") else raw.rfind(b"\n") + 1
    complete = raw[:complete_len]
    records: list[dict] = []
    invalid = 0
    for line in complete.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid += 1
            continue
        if isinstance(item, dict):
            records.append(item)
        else:
            invalid += 1
    digest.update(complete)
    return records, offset + complete_len, invalid, digest.hexdigest()


def ensure_readable_dir(path: Path) -> None:
    """Raise a stable permission error even when the current user is privileged."""

    if not path.exists():
        return
    mode = path.stat().st_mode
    if (
        mode & 0o444 == 0
        or mode & 0o111 == 0
        or not os.access(path, os.R_OK | os.X_OK)
    ):
        raise PermissionError("history source directory is not readable")


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_file_key(source: str, path: Path) -> str:
    return context_module.external_session_hash(f"{source}:{path}") or "unknown"
