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
from .common import (
    PrefixMismatchError,
    incremental_checkpoint_fingerprint,
    private_file_key,
)
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
            if detection.get("status") not in {None, "missing"}:
                summary["status"] = "degraded"
                summary["errors"].append({
                    "code": detection.get("code", "SB_SOURCE_UNAVAILABLE"),
                    "hint": detection.get("hint", detection.get("status")),
                })
            return summary
        files = item.discover()
    except PermissionError as exc:
        summary["status"] = "degraded"
        summary["errors"].append({
            "code": "SB_SOURCE_PERMISSION_DENIED", "hint": type(exc).__name__
        })
        return summary
    except OSError as exc:
        summary["status"] = "degraded"
        summary["errors"].append({"code": "SB_SOURCE_DISCOVERY_FAILED", "hint": type(exc).__name__})
        return summary

    summary["files"] = len(files)
    session_source = getattr(item, "session_source", item.name)
    for path in files:
        file_key = private_file_key(name, path)
        try:
            checkpoint = store.get_source_checkpoint(name, file_key)
            stat = path.stat()
            incremental = getattr(item, "parse_incremental", None)
            if (
                checkpoint
                and callable(incremental)
                and checkpoint["adapter_version"] == item.version
                and checkpoint["session_row_id"] is not None
                and stat.st_size >= checkpoint["cursor"]
            ):
                if (
                    stat.st_size == checkpoint["cursor"]
                    and stat.st_mtime_ns == checkpoint["mtime_ns"]
                    and incremental_checkpoint_fingerprint(
                        path, checkpoint["cursor"]
                    ) == checkpoint["fingerprint"]
                ):
                    summary["skipped"] += 1
                    _restore_checkpoint_diagnostic(summary, checkpoint, file_key)
                    continue
                if stat.st_size > checkpoint["cursor"]:
                    try:
                        parsed_append = incremental(
                            path,
                            checkpoint["cursor"],
                            checkpoint["fingerprint"],
                        )
                    except PrefixMismatchError:
                        # A rewrite or replacement can grow the file too. Reparse
                        # safely instead of seeking into the middle of a record.
                        pass
                    else:
                        summary["scanned"] += 1
                        changed = store.append_session_transcript(
                            checkpoint["session_row_id"],
                            parsed_append.lines,
                            parsed_append.conclusion,
                        )
                        summary["updated" if changed else "skipped"] += 1
                        invalid_records = (
                            checkpoint["invalid_records"]
                            + parsed_append.invalid_records
                        )
                        error_code = (
                            parsed_append.diagnostics[0]
                            if parsed_append.diagnostics
                            else checkpoint["error_code"]
                        )
                        if invalid_records and not error_code:
                            error_code = "SB_SOURCE_JSONL_INVALID"
                        summary["invalid"] += invalid_records
                        if error_code:
                            summary["errors"].append({
                                "code": error_code, "file": file_key,
                                "hint": "invalid_records_persist_until_file_rewrite",
                            })
                        store.set_source_checkpoint(
                            name,
                            file_key,
                            cursor=parsed_append.cursor,
                            fingerprint=parsed_append.fingerprint,
                            adapter_version=item.version,
                            session_row_id=checkpoint["session_row_id"],
                            invalid_records=invalid_records,
                            mtime_ns=stat.st_mtime_ns,
                            status="degraded" if invalid_records else "ok",
                            error_code=error_code,
                        )
                        continue

            if checkpoint and not callable(incremental):
                fingerprint = item.fingerprint(path)
                if checkpoint["fingerprint"] == fingerprint:
                    summary["skipped"] += 1
                    _restore_checkpoint_diagnostic(summary, checkpoint, file_key)
                    continue
            parsed = item.parse(path)
            summary["scanned"] += 1
            summary["invalid"] += parsed.invalid_records
            if not parsed.sessions:
                summary["invalid"] += 1 if not parsed.invalid_records else 0
            session_row_id = None
            for session in parsed.sessions:
                session_row_id, action = store.upsert_external_session(
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
            error_code = parsed.diagnostics[0] if parsed.diagnostics else None
            if parsed.invalid_records and not error_code:
                error_code = "SB_SOURCE_INVALID_RECORD"
            if error_code:
                summary["errors"].append({
                    "code": error_code, "file": file_key,
                    "hint": "source_file_requires_repair",
                })
            store.set_source_checkpoint(
                name,
                file_key,
                cursor=parsed.cursor,
                fingerprint=(
                    incremental_checkpoint_fingerprint(path, parsed.cursor)
                    if callable(incremental)
                    else parsed.fingerprint
                ),
                adapter_version=item.version,
                session_row_id=session_row_id,
                invalid_records=parsed.invalid_records,
                mtime_ns=stat.st_mtime_ns,
                status="degraded" if parsed.invalid_records else "ok",
                error_code=error_code,
            )
        except PermissionError as exc:
            summary["errors"].append({
                "code": "SB_SOURCE_PERMISSION_DENIED",
                "file": file_key,
                "hint": type(exc).__name__,
            })
        except (OSError, sqlite3.Error) as exc:
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


def _restore_checkpoint_diagnostic(
    summary: dict, checkpoint: sqlite3.Row, file_key: str
) -> None:
    if checkpoint["status"] != "degraded" and not checkpoint["error_code"]:
        return
    invalid = max(1, int(checkpoint["invalid_records"] or 0))
    summary["invalid"] += invalid
    summary["errors"].append({
        "code": checkpoint["error_code"] or "SB_SOURCE_CHECKPOINT_DEGRADED",
        "file": file_key,
        "hint": "persisted_source_diagnostic",
    })


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
