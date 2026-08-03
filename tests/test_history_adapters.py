from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from storybook import config, store
from storybook.cli import cli
from storybook.history_adapters import manager
from storybook.history_adapters.cline import ClineAdapter
from storybook.history_adapters.codex import CodexAdapter
from storybook.history_adapters.gemini import GeminiAdapter


def _jsonl(path: Path, rows: list[dict], *, complete: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row) for row in rows)
    path.write_text(text + ("\n" if complete else ""), encoding="utf-8")


def _codex_rows(session_id: str, cwd: str, prompt: str = "Fix the database race") -> list[dict]:
    return [
        {
            "timestamp": "2026-08-01T00:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": "2026-08-01T00:00:00Z",
                "cwd": cwd,
                "cli_version": "0.145.0",
            },
        },
        {
            "timestamp": "2026-08-01T00:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
        {
            "timestamp": "2026-08-01T00:00:02Z",
            "type": "response_item",
            "payload": {"type": "custom_tool_call", "name": "exec", "arguments": "Authorization: Bearer secret"},
        },
        {
            "timestamp": "2026-08-01T00:00:03Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Added a transaction and tests."}],
            },
        },
    ]


def test_codex_incremental_import_is_idempotent_and_updates_only_changed_session(tmp_path):
    root = tmp_path / ".codex"
    files = [
        root / "sessions/2026/08/01/rollout-a.jsonl",
        root / "sessions/2026/08/01/rollout-b.jsonl",
        root / "sessions/2026/08/02/rollout-c.jsonl",
    ]
    _jsonl(files[0], _codex_rows("session-a", "/private/work-a"))
    _jsonl(files[1], _codex_rows("session-b", "/private/work-a", "Add API validation"))
    _jsonl(files[2], _codex_rows("session-c", "/private/work-b", "Repair retry logic"))
    adapter = CodexAdapter(root)

    first = manager.import_source("codex", adapter=adapter)
    second = manager.import_source("codex", adapter=adapter)
    with files[0].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": "2026-08-01T00:00:04Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Verified under concurrency."}],
            },
        }) + "\n")
    third = manager.import_source("codex", adapter=adapter)

    assert (first["imported"], first["files"], first["invalid"]) == (3, 3, 0)
    assert (second["imported"], second["updated"], second["skipped"]) == (0, 0, 3)
    assert (third["imported"], third["updated"], third["skipped"]) == (0, 1, 2)
    db = store.get_db()
    try:
        rows = db.execute("SELECT source, raw_content, context_json FROM sessions ORDER BY id").fetchall()
    finally:
        db.close()
    assert len(rows) == 3
    assert all(row["source"] == "codex" for row in rows)
    persisted = "\n".join(row["raw_content"] + row["context_json"] for row in rows)
    assert "Authorization" not in persisted
    assert "Bearer secret" not in persisted
    assert "/private/work-" not in persisted
    assert "session-a" not in persisted


def test_codex_redacts_complete_secret_values_from_content_and_conclusion(tmp_path):
    root = tmp_path / ".codex"
    path = root / "sessions/2026/08/01/secret.jsonl"
    rows = _codex_rows("session-secret", "/private/work")
    rows[-1]["payload"]["content"][0]["text"] = (
        "Authorization: topsecret-value\n"
        "Authorization: Basic dXNlcjpwYXNz\n"
        "Proxy-Authorization: Digest username=admin,response=deadbeef\n"
        "Authorization: ApiKey live-secret-123\n"
        "token=another-secret\nBearer third-secret\n"
        "OPENAI_API_KEY=fourth-secret\n{\"Authorization\":\"fifth-secret\"}"
    )
    _jsonl(path, rows)

    out = manager.import_source("codex", adapter=CodexAdapter(root))

    assert out["status"] == "ok"
    db = store.get_db()
    try:
        row = db.execute(
            "SELECT raw_content, conclusion FROM sessions WHERE source = 'codex'"
        ).fetchone()
    finally:
        db.close()
    persisted = row["raw_content"] + row["conclusion"]
    assert "topsecret-value" not in persisted
    assert "dXNlcjpwYXNz" not in persisted
    assert "username=admin" not in persisted
    assert "deadbeef" not in persisted
    assert "live-secret-123" not in persisted
    assert "another-secret" not in persisted
    assert "third-secret" not in persisted
    assert "fourth-secret" not in persisted
    assert "fifth-secret" not in persisted
    assert "[REDACTED]" in persisted


def test_codex_truncated_tail_and_bad_line_are_isolated_and_resumable(tmp_path):
    root = tmp_path / ".codex"
    path = root / "sessions/2026/08/01/rollout.jsonl"
    _jsonl(path, _codex_rows("session-safe", "/workspace"))
    with path.open("ab") as handle:
        handle.write(b"not-json\n")
        handle.write(b'{"type":"response_item","payload":')

    first = manager.import_source("codex", adapter=CodexAdapter(root))
    assert first["imported"] == 1
    assert first["invalid"] == 1
    assert first["status"] == "degraded"

    with path.open("ab") as handle:
        handle.write(b'{"type":"message"}}\n')
    second = manager.import_source("codex", adapter=CodexAdapter(root))
    assert second["imported"] == 0
    assert second["updated"] == 0
    assert second["scanned"] == 1

    third = manager.import_source("codex", adapter=CodexAdapter(root))
    assert third["status"] == "degraded"
    assert third["invalid"] == 1
    assert third["errors"][0]["code"] == "SB_SOURCE_JSONL_INVALID"

    monkeypatch_adapter = CodexAdapter(root)
    original = manager.adapters
    try:
        manager.adapters = lambda: {"codex": monkeypatch_adapter}
        listed = manager.list_sources()
    finally:
        manager.adapters = original
    assert listed[0]["status"] == "SB_SOURCE_JSONL_INVALID"

    _jsonl(path, _codex_rows("session-safe", "/workspace"))
    repaired = manager.import_source("codex", adapter=CodexAdapter(root))
    assert repaired["status"] == "ok"
    assert repaired["invalid"] == 0
    assert store.list_source_checkpoints("codex")[0]["error_code"] is None


def test_codex_permission_denied_is_stable_and_degraded(tmp_path):
    root = tmp_path / ".codex"
    sessions = root / "sessions"
    _jsonl(sessions / "2026/08/01/one.jsonl", _codex_rows("one", "/work"))
    sessions.chmod(0)
    try:
        first = manager.import_source("codex", adapter=CodexAdapter(root))
        second = manager.import_source("codex", adapter=CodexAdapter(root))
    finally:
        sessions.chmod(0o755)
    for result in (first, second):
        assert result["status"] == "degraded"
        assert result["errors"] == [{
            "code": "SB_SOURCE_PERMISSION_DENIED", "hint": "PermissionError"
        }]


def test_codex_append_uses_checkpoint_offset_without_full_parse(tmp_path, monkeypatch):
    root = tmp_path / ".codex"
    path = root / "sessions/2026/08/01/offset.jsonl"
    adapter = CodexAdapter(root)
    _jsonl(path, _codex_rows("session-offset", "/work"))
    assert manager.import_source("codex", adapter=adapter)["imported"] == 1

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": "2026-08-01T00:00:04Z",
            "type": "response_item",
            "payload": {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "offset-only append"}],
            },
        }) + "\n")
    restarted_adapter = CodexAdapter(root)
    monkeypatch.setattr(restarted_adapter, "parse", lambda _path: (_ for _ in ()).throw(
        AssertionError("full parse must not run for append")
    ))

    out = manager.import_source("codex", adapter=restarted_adapter)
    assert out["updated"] == 1
    checkpoint = store.list_source_checkpoints("codex")[0]
    assert checkpoint["cursor"] == path.stat().st_size
    assert checkpoint["session_row_id"] is not None


@pytest.mark.parametrize("replace_file", [False, True])
def test_codex_longer_rewrite_invalidates_prefix_and_reparses(
    tmp_path, monkeypatch, replace_file
):
    root = tmp_path / ".codex"
    path = root / "sessions/2026/08/01/rewrite.jsonl"
    adapter = CodexAdapter(root)
    rows = _codex_rows("session-rewrite", "/work")
    _jsonl(path, rows)
    assert manager.import_source("codex", adapter=adapter)["imported"] == 1

    with path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
    degraded = manager.import_source("codex", adapter=CodexAdapter(root))
    assert degraded["status"] == "degraded"
    assert degraded["invalid"] == 1

    rows.append({
        "timestamp": "2026-08-01T00:00:04Z",
        "type": "response_item",
        "payload": {
            "type": "message", "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "healthy replacement record that is longer than the bad line",
            }],
        },
    })
    if replace_file:
        replacement = path.with_name("replacement.jsonl")
        _jsonl(replacement, rows)
        replacement.replace(path)
    else:
        _jsonl(path, rows)
    reparsing_adapter = CodexAdapter(root)
    parse_calls = 0
    original_parse = reparsing_adapter.parse

    def tracked_parse(candidate):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(candidate)

    monkeypatch.setattr(reparsing_adapter, "parse", tracked_parse)
    repaired = manager.import_source("codex", adapter=reparsing_adapter)

    assert repaired["status"] == "ok"
    assert repaired["invalid"] == 0
    assert repaired["updated"] == 1
    assert parse_calls == 1
    checkpoint = store.list_source_checkpoints("codex")[0]
    assert checkpoint["cursor"] == path.stat().st_size
    assert checkpoint["error_code"] is None
    db = store.get_db()
    try:
        conclusion = db.execute(
            "SELECT conclusion FROM sessions WHERE source = 'codex'"
        ).fetchone()["conclusion"]
    finally:
        db.close()
    assert conclusion == "healthy replacement record that is longer than the bad line"


def test_gemini_and_cline_supported_fixtures_import(tmp_path):
    gemini_root = tmp_path / ".gemini"
    gemini_path = gemini_root / "tmp/project/chats/session-gemini.json"
    gemini_path.parent.mkdir(parents=True)
    gemini_path.write_text(json.dumps({
        "sessionId": "gemini-1",
        "startTime": "2026-08-01T00:00:00Z",
        "summary": "Fix the cache", "projectHash": "project-safe-hash",
        "messages": [
            {"type": "user", "content": "Fix the cache invalidation"},
            {"type": "model", "content": "Added versioned keys"},
        ],
    }), encoding="utf-8")
    (gemini_path.parent.parent / ".project_root").write_text(
        "/private/gemini-workspace", encoding="utf-8"
    )

    cline_root = tmp_path / "cline-tasks"
    cline_path = cline_root / "task-1/api_conversation_history.json"
    cline_path.parent.mkdir(parents=True)
    cline_path.write_text(json.dumps([
        {"role": "user", "content": [{"type": "text", "text": "Repair the queue\n<environment_details>/secret/path</environment_details>"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Added bounded retries"}]},
    ]), encoding="utf-8")
    (cline_path.parent / "task_metadata.json").write_text(json.dumps({
        "workspace": "/private/cline-workspace", "projectName": "queue-service"
    }), encoding="utf-8")

    gemini = manager.import_source("gemini", adapter=GeminiAdapter(gemini_root))
    cline = manager.import_source("cline", adapter=ClineAdapter([cline_root]))
    assert gemini["status"] == cline["status"] == "ok"
    assert gemini["imported"] == cline["imported"] == 1
    db = store.get_db()
    try:
        rows = db.execute(
            "SELECT source, raw_content, context_json FROM sessions ORDER BY id"
        ).fetchall()
    finally:
        db.close()
    assert [row["source"] for row in rows] == ["gemini", "cline"]
    assert "/secret/path" not in rows[1]["raw_content"]
    contexts = [json.loads(row["context_json"]) for row in rows]
    assert contexts[0]["tool"]["type"] == "gemini_cli"
    assert contexts[1]["tool"]["type"] == "cline"
    assert contexts[0]["tool"]["adapter_version"] == GeminiAdapter.version
    assert contexts[1]["tool"]["adapter_version"] == ClineAdapter.version
    assert contexts[0]["workspace"]["cwd_alias"] == "gemini-workspace"
    assert contexts[1]["workspace"]["cwd_alias"] == "cline-workspace"
    persisted_context = json.dumps(contexts)
    assert "/private/gemini-workspace" not in persisted_context
    assert "/private/cline-workspace" not in persisted_context


def test_import_data_codex_json_has_stable_summary(monkeypatch):
    expected = {
        "status": "ok",
        "imported": 3,
        "updated": 0,
        "sources": [{"source": "codex", "files": 3, "scanned": 3, "imported": 3,
                     "updated": 0, "skipped": 0, "invalid": 0, "status": "ok"}],
    }
    monkeypatch.setattr(manager, "import_enabled", lambda selected: expected)
    result = CliRunner().invoke(cli, ["import-data", "--codex", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected


def test_database_initialization_log_does_not_expose_absolute_path(caplog):
    caplog.set_level(logging.INFO, logger="storybook.store")
    store.init_db()
    assert str(config.DB_PATH) not in caplog.text
    assert "数据库初始化完成" in caplog.text


def test_existing_checkpoint_table_is_upgraded_in_place():
    db = store.get_db(load_vector_extension=False)
    try:
        db.execute("DROP TABLE source_checkpoints")
        db.execute("""CREATE TABLE source_checkpoints (
            source TEXT NOT NULL,
            file_key TEXT NOT NULL,
            cursor INTEGER NOT NULL DEFAULT 0,
            fingerprint TEXT NOT NULL,
            adapter_version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ok',
            error_code TEXT,
            updated_at TEXT,
            PRIMARY KEY (source, file_key)
        )""")
        db.commit()
    finally:
        db.close()

    store.init_db()

    db = store.get_db(load_vector_extension=False)
    try:
        columns = {
            row["name"] for row in db.execute(
                "PRAGMA table_info(source_checkpoints)"
            ).fetchall()
        }
    finally:
        db.close()
    assert {"session_row_id", "invalid_records", "mtime_ns"} <= columns


def test_source_failure_does_not_block_other_sources(monkeypatch):
    class Broken:
        name = "codex"
        version = "broken"
        def detect(self):
            raise RuntimeError("boom")

    class Missing:
        name = "gemini"
        version = "missing"
        def detect(self):
            return {"available": False}

    monkeypatch.setattr(manager, "adapters", lambda: {"codex": Broken(), "gemini": Missing()})
    out = manager.import_enabled(["codex", "gemini"])
    assert out["status"] == "degraded"
    assert [item["source"] for item in out["sources"]] == ["codex", "gemini"]
    assert out["sources"][1]["status"] == "unavailable"
