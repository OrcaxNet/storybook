"""Registry, source settings and failure-isolated incremental ingestion."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .. import config, store
from .base import HistoryAdapter
from .claude import ClaudeAdapter
from .cline import ClineAdapter
from .codex import CodexAdapter
from .common import private_file_key
from .cursor import CursorAdapter
from .gemini import GeminiAdapter

SOURCE_NAMES = ("claude", "codex", "cursor", "gemini", "cline")


def adapters() -> dict[str, HistoryAdapter]:
    return {
        "claude": ClaudeAdapter(),
        "codex": CodexAdapter(),
        "cursor": CursorAdapter(),
        "gemini": GeminiAdapter(),
        "cline": ClineAdapter(),
    }


def _settings_path() -> Path:
    return config.DATA_DIR / "sources.json"


def load_settings() -> dict[str, bool]:
    defaults = {name: True for name in SOURCE_NAMES}
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return defaults
    enabled = data.get("enabled") if isinstance(data, dict) else None
    if not isinstance(enabled, dict):
        return defaults
    for name in SOURCE_NAMES:
        if isinstance(enabled.get(name), bool):
            defaults[name] = enabled[name]
    return defaults


def set_enabled(name: str, enabled: bool) -> None:
    if name not in SOURCE_NAMES:
        raise ValueError(f"unknown source: {name}")
    config.ensure_profile()
    settings = load_settings()
    settings[name] = enabled
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps({"version": 1, "enabled": settings}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def import_source(name: str, *, adapter: HistoryAdapter | None = None) -> dict:
    item = adapter or adapters()[name]
    summary = {
        "source": name,
        "adapter_version": item.version,
        "detected": False,
        "files": 0,
        "scanned": 0,
        "imported": 0,
        "updated": 0,
        "skipped": 0,
        "invalid": 0,
        "errors": [],
        "status": "unavailable",
    }
    try:
        detection = item.detect()
        summary["detected"] = bool(detection.get("available"))
        if not summary["detected"]:
            return summary
        files = item.discover()
    except (OSError, PermissionError) as exc:
        summary["status"] = "degraded"
        summary["errors"].append({"code": "SB_SOURCE_DISCOVERY_FAILED", "hint": type(exc).__name__})
        return summary

    summary["files"] = len(files)
    session_source = getattr(item, "session_source", item.name)
    for path in files:
        file_key = private_file_key(name, path)
        try:
            fingerprint = item.fingerprint(path)
            checkpoint = store.get_source_checkpoint(name, file_key)
            if checkpoint and checkpoint["fingerprint"] == fingerprint:
                summary["skipped"] += 1
                continue
            parsed = item.parse(path)
            summary["scanned"] += 1
            summary["invalid"] += parsed.invalid_records
            if not parsed.sessions:
                summary["invalid"] += 1 if not parsed.invalid_records else 0
            for session in parsed.sessions:
                _, action = store.upsert_external_session(
                    session_source,
                    session.external_id,
                    session.raw_content,
                    session.problem_desc,
                    session.code_snippets,
                    session.conclusion,
                    session.context,
                )
                if action == "created":
                    summary["imported"] += 1
                elif action == "updated":
                    summary["updated"] += 1
                else:
                    summary["skipped"] += 1
            store.set_source_checkpoint(
                name,
                file_key,
                cursor=parsed.cursor,
                fingerprint=parsed.fingerprint,
                adapter_version=item.version,
                status="degraded" if parsed.invalid_records else "ok",
                error_code=parsed.diagnostics[0] if parsed.diagnostics else None,
            )
        except (OSError, PermissionError, sqlite3.Error) as exc:
            summary["errors"].append({
                "code": "SB_SOURCE_FILE_FAILED",
                "file": file_key,
                "hint": type(exc).__name__,
            })
        except Exception as exc:  # one unknown schema/file must not stop other sources
            summary["errors"].append({
                "code": "SB_SOURCE_PARSE_FAILED",
                "file": file_key,
                "hint": type(exc).__name__,
            })
    summary["status"] = "degraded" if summary["errors"] or summary["invalid"] else "ok"
    return summary


def import_enabled(selected: tuple[str, ...] | list[str] | None = None) -> dict:
    enabled = load_settings()
    names = list(selected) if selected else [name for name in SOURCE_NAMES if enabled[name]]
    results: list[dict] = []
    for name in names:
        if name not in SOURCE_NAMES:
            results.append({"source": name, "status": "invalid", "errors": [{"code": "SB_SOURCE_UNKNOWN"}]})
            continue
        try:
            results.append(import_source(name))
        except Exception as exc:  # failure isolation across sources
            results.append({
                "source": name,
                "status": "degraded",
                "errors": [{"code": "SB_SOURCE_FAILED", "hint": type(exc).__name__}],
            })
    imported = sum(item.get("imported", 0) for item in results)
    updated = sum(item.get("updated", 0) for item in results)
    degraded = any(item.get("status") in {"degraded", "invalid"} for item in results)
    return {
        "status": "degraded" if degraded else "ok",
        "imported": imported,
        "updated": updated,
        "sources": results,
    }


def list_sources() -> list[dict]:
    enabled = load_settings()
    checkpoints = store.list_source_checkpoints()
    latest: dict[str, str] = {}
    errors: dict[str, str] = {}
    for row in checkpoints:
        latest.setdefault(row["source"], row["updated_at"])
        if row["error_code"]:
            errors.setdefault(row["source"], row["error_code"])
    items = []
    for name, adapter in adapters().items():
        try:
            detection = adapter.detect()
        except (OSError, PermissionError):
            detection = {"available": False, "status": "permission_denied"}
        items.append({
            "name": name,
            "display_name": adapter.display_name,
            "available": bool(detection.get("available")),
            "enabled": enabled[name],
            "status": errors.get(name) or detection.get("status", "unknown"),
            "adapter_version": adapter.version,
            "last_imported_at": latest.get(name),
        })
    return items
