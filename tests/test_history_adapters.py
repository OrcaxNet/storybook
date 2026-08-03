from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from storybook import store
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


def test_gemini_and_cline_supported_fixtures_import(tmp_path):
    gemini_root = tmp_path / ".gemini"
    gemini_path = gemini_root / "tmp/project/chats/session-gemini.json"
    gemini_path.parent.mkdir(parents=True)
    gemini_path.write_text(json.dumps({
        "sessionId": "gemini-1",
        "startTime": "2026-08-01T00:00:00Z",
        "summary": "Fix the cache",
        "messages": [
            {"type": "user", "content": "Fix the cache invalidation"},
            {"type": "model", "content": "Added versioned keys"},
        ],
    }), encoding="utf-8")

    cline_root = tmp_path / "cline-tasks"
    cline_path = cline_root / "task-1/api_conversation_history.json"
    cline_path.parent.mkdir(parents=True)
    cline_path.write_text(json.dumps([
        {"role": "user", "content": [{"type": "text", "text": "Repair the queue\n<environment_details>/secret/path</environment_details>"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Added bounded retries"}]},
    ]), encoding="utf-8")

    gemini = manager.import_source("gemini", adapter=GeminiAdapter(gemini_root))
    cline = manager.import_source("cline", adapter=ClineAdapter([cline_root]))
    assert gemini["status"] == cline["status"] == "ok"
    assert gemini["imported"] == cline["imported"] == 1
    db = store.get_db()
    try:
        rows = db.execute("SELECT source, raw_content FROM sessions ORDER BY id").fetchall()
    finally:
        db.close()
    assert [row["source"] for row in rows] == ["gemini", "cline"]
    assert "/secret/path" not in rows[1]["raw_content"]


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

