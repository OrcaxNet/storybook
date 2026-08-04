"""Profile-local persistent cache for deterministic LLM and embedding calls."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from . import config

_SCHEMA_LOCK = threading.Lock()


def _path() -> Path:
    path = config.CACHE_DIR / "inference-cache.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    path = _path()
    db = sqlite3.connect(str(path), timeout=5.0)
    db.execute("PRAGMA busy_timeout=5000")
    with _SCHEMA_LOCK:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """CREATE TABLE IF NOT EXISTS inference_cache (
                   namespace TEXT NOT NULL,
                   input_hash TEXT NOT NULL,
                   value_json TEXT NOT NULL,
                   created_at TEXT NOT NULL DEFAULT (datetime('now')),
                   last_hit_at TEXT,
                   hit_count INTEGER NOT NULL DEFAULT 0,
                   PRIMARY KEY(namespace, input_hash)
               )"""
        )
        if os.name != "nt":
            path.chmod(0o600)
    return db


def input_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get(namespace: str, payload: Any) -> Any | None:
    if not config.INFERENCE_CACHE_ENABLED:
        return None
    key = input_hash(payload)
    db = _connect()
    try:
        row = db.execute(
            "SELECT value_json FROM inference_cache WHERE namespace = ? AND input_hash = ?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        db.execute(
            """UPDATE inference_cache
               SET hit_count = hit_count + 1, last_hit_at = datetime('now')
               WHERE namespace = ? AND input_hash = ?""",
            (namespace, key),
        )
        db.commit()
        return json.loads(row[0])
    finally:
        db.close()


def set(namespace: str, payload: Any, value: Any) -> None:
    if not config.INFERENCE_CACHE_ENABLED or value is None:
        return
    key = input_hash(payload)
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    db = _connect()
    try:
        db.execute(
            """INSERT INTO inference_cache(namespace, input_hash, value_json)
               VALUES (?, ?, ?)
               ON CONFLICT(namespace, input_hash) DO UPDATE SET
                   value_json = excluded.value_json,
                   created_at = datetime('now')""",
            (namespace, key, encoded),
        )
        db.commit()
    finally:
        db.close()


def stats() -> dict:
    db = _connect()
    try:
        row = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM inference_cache"
        ).fetchone()
        return {"entries": int(row[0]), "hits": int(row[1])}
    finally:
        db.close()
