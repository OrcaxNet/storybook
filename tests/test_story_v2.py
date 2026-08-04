"""Story v2 acceptance tests: lossless detail, semantic chunks and backfill."""
from __future__ import annotations

import json
import sqlite3
import uuid

import numpy as np
import pytest

from storybook import config, embeddings, mcp_server, processor, store, story_v2

from ._helpers import basis, vector_in_index


def _payload(title: str, marker: str) -> dict:
    return {
        "title": title,
        "abstract": f"{marker} 的关键结论",
        "detail": {
            "problem": f"{marker} 问题",
            "actions": [f"执行 {marker} 修复"],
            "outcome": f"{marker} 已恢复",
            "pitfalls": [f"避免 {marker} 旧配置"],
            "evidence": [f"日志包含 {marker}-ok"],
            "applicability": {
                "applies_when": [{"runtime.kind": "local"}],
                "excludes_when": [],
            },
        },
        "sources": [{"evidence": [f"原会话 {marker} 段落"]}],
        "keywords": [marker],
    }


def test_atomic_long_story_persists_full_detail_and_bounded_abstract():
    long_evidence = "原子证据 " * 1000
    sid = store.add_session("test", long_evidence, "一个不可拆问题")
    story_id = store.add_story(
        "长原子经历",
        long_evidence,
        ["atomic"],
        basis(0),
        source_session_ids=[sid],
        abstract="结论 " * 400,
        detail={
            "problem": long_evidence,
            "actions": ["保持为单一 Story"],
            "outcome": "完整保存",
            "pitfalls": [],
            "evidence": [long_evidence],
            "applicability": {"applies_when": [], "excludes_when": []},
        },
    )

    story = store.get_story(story_id)
    assert story["detail"]["problem"] == long_evidence.strip()
    assert story["detail"]["evidence"] == [long_evidence.strip()]
    assert len(story["abstract"]) <= config.STORY_ABSTRACT_MAX_CHARS
    assert store.count_stories() == 1


def test_short_session_with_two_conclusions_forms_two_stories(
    fake_llm, fake_embedder
):
    fake_llm.keywords = []
    fake_llm.stories = [_payload("问题 A", "alpha"), _payload("问题 B", "beta")]
    sid = store.add_session(
        "test", "先修 alpha 并成功；再处理 beta 并成功。", "两个独立问题"
    )
    normalized = [
        story_v2.normalize_story_payload(payload, source_session_ids=[sid])
        for payload in fake_llm.stories
    ]
    fake_embedder.register("两个独立问题 两个独立问题", basis(9))
    fake_embedder.register(story_v2.embedding_input(normalized[0]), basis(0))
    fake_embedder.register(story_v2.embedding_input(normalized[1]), basis(1))

    first_id = processor.process_session(sid)

    stories = store.get_all_stories()
    assert first_id in {story["id"] for story in stories}
    assert {story["title"] for story in stories} == {"问题 A", "问题 B"}
    assert all(story["source_session_ids"] == [sid] for story in stories)
    for story in stories:
        detail = store.get_story(story["id"])
        assert detail["sources"][0]["session_id"] == sid
        assert detail["sources"][0]["session_global_id"]


def test_default_embedding_uses_title_abstract_and_applicability_only():
    payload = story_v2.normalize_story_payload(_payload("标题", "marker"))
    text = story_v2.embedding_input(payload)
    assert "标题" in text
    assert "marker 的关键结论" in text
    assert "runtime.kind" in text
    assert "执行 marker 修复" not in text
    assert "日志包含 marker-ok" not in text


def test_recall_is_summary_first_and_detail_expands_with_compat_fields(fake_embedder):
    payload = story_v2.normalize_story_payload(_payload("标题", "marker"))
    story_id = store.add_story(
        payload["title"], payload["content"], payload["keywords"], basis(0),
        abstract=payload["abstract"], detail=payload["detail"],
        sources=payload["sources"], applicability=payload["applicability"],
    )
    fake_embedder.register("q", basis(0))

    recalled = mcp_server.recall_memories("q")["matches"][0]
    assert recalled["content"] == payload["abstract"]  # legacy field retained
    assert recalled["abstract"] == payload["abstract"]
    assert recalled["truncated"] is True
    assert "日志包含 marker-ok" not in recalled["content"]

    detail = mcp_server.get_story_detail(story_id)
    assert detail["content"] == payload["content"]
    assert detail["detail"]["evidence"] == ["日志包含 marker-ok"]
    assert detail["sources"][0]["evidence"] == ["原会话 marker 段落"]
    assert {"title", "content", "version"} <= detail.keys()


def test_revision_chain_records_create_merge_and_split():
    story_id = store.add_story("t", "c", [], basis(0))
    store.update_story(
        story_id, title="t2", content="c2", embedding=basis(1),
        event_type="merge",
    )
    store.delete_story_vector(story_id)
    revisions = store.get_story_revisions(story_id)
    assert [item["event_type"] for item in revisions] == [
        "create", "merge", "split_parent"
    ]
    assert [item["version"] for item in revisions] == [1, 2, 3]


def test_embedding_backfill_failure_resumes_and_switches_atomically(monkeypatch):
    first = store.add_story("first", "c1", [], basis(0))
    second = store.add_story("second", "c2", [], basis(1))
    active_before = {
        first: vector_in_index(first), second: vector_in_index(second)
    }
    calls = []

    def fail_second(text, model=None, **kwargs):
        calls.append(text)
        return basis(2) if "first" in text else None

    monkeypatch.setattr(embeddings, "embed", fail_second)
    partial = embeddings.backfill(
        model="new-model", version="v2-test", batch_size=10
    )
    assert partial["failed"] == 1
    assert partial["activation"] is None
    assert vector_in_index(first) == active_before[first]
    assert vector_in_index(second) == active_before[second]

    retry_calls = []

    def succeed(text, model=None, **kwargs):
        retry_calls.append(text)
        return basis(3)

    monkeypatch.setattr(embeddings, "embed", succeed)
    resumed = embeddings.backfill(
        model="new-model", version="v2-test", batch_size=10
    )
    assert resumed["attempted"] == 1  # ready first Story was content-hash skipped
    assert resumed["activation"]["activated"] == 2
    assert vector_in_index(first) == basis(2)
    assert vector_in_index(second) == basis(3)
    state = store.get_embedding_index_state()
    assert state["active_model"] == "new-model"
    assert state["active_version"] == "v2-test"
    assert state["active_endpoint"] == config.EMBED_BASE_URL
    assert state["active_adapter"] == config.EMBED_ADAPTER
    assert state["active_dimension"] == config.EMBED_DIM


def test_endpoint_switch_requires_shadow_and_activates_full_identity(monkeypatch):
    from storybook import health

    store.add_story("endpoint identity", "content", [], basis(0))
    active_before = store.get_embedding_index_state()
    monkeypatch.setattr(config, "EMBED_BASE_URL", "https://endpoint-b.example/v1")
    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")

    compatibility = health.serving_index_compatibility(config.EMBED_DIM)
    assert compatibility["ok"] is False
    assert "endpoint active=" in compatibility["detail"]
    assert "adapter active=ollama target=openai_compatible" in compatibility["detail"]
    assert store.get_embedding_index_state()["active_endpoint"] == active_before[
        "active_endpoint"
    ]

    monkeypatch.setattr(embeddings, "embed", lambda *args, **kwargs: basis(1))
    result = embeddings.backfill(
        model=active_before["active_model"],
        version=active_before["active_version"],
        batch_size=10,
    )

    assert result["activation"]["activated"] == 1
    active_after = store.get_embedding_index_state()
    assert active_after["active_endpoint"] == "https://endpoint-b.example/v1"
    assert active_after["active_adapter"] == "openai_compatible"
    assert active_after["active_dimension"] == config.EMBED_DIM
    assert health.serving_index_compatibility(config.EMBED_DIM)["ok"] is True


@pytest.mark.parametrize("extra_updates", [1, 3])
def test_embedding_activation_uses_each_story_pre_switch_version(
    monkeypatch, extra_updates
):
    first = store.add_story("first", "c1", [], basis(0))
    second = store.add_story("second", "c2", [], basis(1))
    for version in range(2, extra_updates + 2):
        store.update_story(second, title=f"second-v{version}")

    monkeypatch.setattr(embeddings, "embed", lambda *args, **kwargs: basis(3))
    result = embeddings.backfill(
        model="version-aware-model",
        version="version-aware-v1",
        batch_size=10,
    )

    assert result["activation"]["activated"] == 2
    expected = {
        first: (1, 2),
        second: (1 + extra_updates, 2 + extra_updates),
    }
    for story_id, (base_version, version) in expected.items():
        event = store.get_memory_events(story_id)[-1]
        assert event["operation"] == "update"
        assert event["payload"]["revision_type"] == "embedding_switch"
        assert event["base_version"] == base_version
        assert event["version"] == version
        assert store.get_story(story_id)["version"] == version
        assert store.replay_memory_events(
            store.get_memory_events(story_id)
        )["conflicts"] == []


def test_embedding_activation_event_failure_rolls_back_serving_state(
    monkeypatch,
):
    first = store.add_story("first", "c1", [], basis(0))
    second = store.add_story("second", "c2", [], basis(1))
    story_before = {
        story_id: store.get_story(story_id) for story_id in (first, second)
    }
    vectors_before = {
        story_id: vector_in_index(story_id) for story_id in (first, second)
    }
    revisions_before = {
        story_id: store.get_story_revisions(story_id)
        for story_id in (first, second)
    }
    events_before = {
        story_id: store.get_memory_events(story_id)
        for story_id in (first, second)
    }
    active_before = store.get_embedding_index_state()

    db = store.get_db(load_vector_extension=False)
    try:
        db.execute(
            """CREATE TRIGGER reject_embedding_switch_event
               BEFORE INSERT ON memory_events
               WHEN new.operation = 'update' BEGIN
                   SELECT RAISE(ABORT, 'embedding event rejected');
               END"""
        )
        db.commit()
    finally:
        db.close()

    monkeypatch.setattr(embeddings, "embed", lambda *args, **kwargs: basis(4))
    with pytest.raises(sqlite3.DatabaseError, match="embedding event rejected"):
        embeddings.backfill(
            model="rollback-model",
            version="rollback-v1",
            batch_size=10,
        )

    active_after = store.get_embedding_index_state()
    assert active_after["active_model"] == active_before["active_model"]
    assert active_after["active_version"] == active_before["active_version"]
    assert active_after["active_representation"] == active_before[
        "active_representation"
    ]
    for story_id in (first, second):
        assert vector_in_index(story_id) == vectors_before[story_id]
        assert store.get_story(story_id) == story_before[story_id]
        assert store.get_story_revisions(story_id) == revisions_before[story_id]
        assert store.get_memory_events(story_id) == events_before[story_id]


def test_story_metadata_hash_is_stable_and_auditable():
    payload = story_v2.normalize_story_payload(_payload("标题", "marker"))
    story_id = store.add_story(
        payload["title"], payload["content"], payload["keywords"], basis(0),
        abstract=payload["abstract"], detail=payload["detail"],
        sources=payload["sources"], applicability=payload["applicability"],
    )
    story = store.get_story(story_id)
    assert story["embedding_model"] == config.EMBED_MODEL
    assert story["embedding_version"] == config.EMBED_VERSION
    assert story["embedding_content_hash"] == story_v2.content_hash(story)
    json.dumps(store.get_story_revisions(story_id), ensure_ascii=False)


@pytest.mark.parametrize("representation", ["default", "full", "legacy"])
def test_update_hash_matches_final_persisted_story_for_active_representation(
    representation,
):
    story_id = store.add_story("old", "old content", ["old"], basis(0))
    db = store.get_db(load_vector_extension=False)
    try:
        db.execute(
            "UPDATE embedding_index_state SET active_representation = ? WHERE id = 1",
            (representation,),
        )
        db.commit()
    finally:
        db.close()

    detail = {
        "problem": "new problem",
        "actions": ["new action"],
        "outcome": "new outcome",
        "evidence": ["new evidence"],
        "applicability": {
            "applies_when": [{"runtime.kind": "local"}],
            "excludes_when": [],
        },
    }
    store.update_story(
        story_id,
        title="new title",
        abstract="new abstract",
        detail=detail,
        keywords=["new"],
        embedding=basis(1),
    )

    story = store.get_story(story_id)
    assert story["content"] == story_v2.render_detail(story["detail"])
    assert story["applicability"] == story["detail"]["applicability"]
    assert story["embedding_content_hash"] == story_v2.content_hash(
        story, representation
    )


def test_legacy_vector_requires_shadow_backfill_before_v2_activation(monkeypatch):
    config.DB_PATH.unlink()
    legacy = sqlite3.connect(config.DB_PATH)
    legacy.executescript(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            raw_content TEXT NOT NULL,
            problem_desc TEXT,
            code_snippets TEXT,
            conclusion TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            processed_at TEXT
        );
        CREATE TABLE stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            keywords TEXT NOT NULL DEFAULT '[]',
            embedding BLOB,
            parent_id INTEGER,
            source_session_ids TEXT DEFAULT '[]',
            access_count INTEGER DEFAULT 0,
            version INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            weight REAL DEFAULT 0.0,
            edge_type TEXT DEFAULT 'semantic',
            UNIQUE(source_id, target_id)
        );
        """
    )
    legacy.execute(
        "INSERT INTO stories (title, content, embedding) VALUES (?, ?, ?)",
        (
            "legacy",
            "legacy content",
            np.asarray(basis(0), dtype=np.float32).tobytes(),
        ),
    )
    legacy.commit()
    legacy.close()

    store.init_db()
    store.init_db()

    migrated = store.get_story(1)
    migration_events = store.get_memory_events(1)
    assert len(migration_events) == 1
    assert migration_events[0]["operation"] == "create"
    assert migration_events[0]["base_version"] == 0
    assert migration_events[0]["version"] == migrated["version"]
    assert uuid.UUID(migration_events[0]["event_id"]).version == 7
    assert uuid.UUID(migrated["global_id"]).version == 7
    initial_state = store.get_embedding_index_state()
    assert migrated["embedding_status"] == "stale"
    assert migrated["embedding_version"] is None
    assert migrated["embedding_content_hash"] is None
    assert initial_state["active_version"] != config.EMBED_VERSION
    assert initial_state["active_representation"] == "legacy"

    monkeypatch.setattr(embeddings, "embed", lambda *args, **kwargs: basis(2))
    result = embeddings.backfill(
        model=config.EMBED_MODEL,
        version=config.EMBED_VERSION,
        representation="default",
    )
    switched = store.get_story(1)
    assert result["activation"]["activated"] == 1
    assert switched["embedding_status"] == "active"
    assert switched["embedding_version"] == config.EMBED_VERSION
    assert switched["embedding_content_hash"] == story_v2.content_hash(switched)
    assert [
        event["operation"] for event in store.get_memory_events(1)
    ] == ["create", "update"]


def test_legacy_ollama_index_identity_migrates_without_vector_rewrite(monkeypatch):
    story_id = store.add_story("legacy identity", "content", [], basis(0))
    story_before = store.get_story(story_id)
    vector_before = vector_in_index(story_id)
    active = store.get_embedding_index_state()
    db = store.get_db(load_vector_extension=False)
    try:
        db.execute("DROP TABLE embedding_index_state")
        db.execute(
            """CREATE TABLE embedding_index_state (
                   id INTEGER PRIMARY KEY CHECK(id = 1),
                   active_model TEXT NOT NULL,
                   active_version TEXT NOT NULL,
                   active_representation TEXT NOT NULL,
                   target_model TEXT,
                   target_version TEXT,
                   target_representation TEXT,
                   backfill_status TEXT NOT NULL DEFAULT 'idle',
                   updated_at TEXT DEFAULT (datetime('now'))
               )"""
        )
        db.execute(
            """INSERT INTO embedding_index_state (
                   id, active_model, active_version, active_representation,
                   backfill_status
               ) VALUES (1, ?, ?, ?, 'idle')""",
            (
                active["active_model"], active["active_version"],
                active["active_representation"],
            ),
        )
        db.commit()
    finally:
        db.close()

    store.init_db()

    migrated = store.get_embedding_index_state()
    assert migrated["active_endpoint"] == config.EMBED_BASE_URL
    assert migrated["active_adapter"] == "ollama"
    assert migrated["active_api_key_env"] == ""
    assert store.get_story(story_id) == story_before
    assert vector_in_index(story_id) == vector_before

    requested = {}

    def fake_post(url, **kwargs):
        requested.update({"url": url, **kwargs})

        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": basis(0)}

        return Response()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    assert embeddings.embed("legacy recall") == basis(0)
    assert requested["url"] == f"{config.EMBED_BASE_URL}/api/embeddings"


def test_backfill_atomically_switches_serving_vector_dimension(monkeypatch):
    story_id = store.add_story("old", "old content", ["old"], basis(0))
    assert store.serving_embedding_dimension() == config.EMBED_DIM
    assert vector_in_index(story_id) == basis(0)

    monkeypatch.setattr(config, "EMBED_DIM", 2)
    monkeypatch.setattr(config, "EMBED_MODEL", "two-dimensional-model")
    monkeypatch.setattr(config, "EMBED_VERSION", "two-dimensional-v1")
    assert store.serving_embedding_dimension() != config.EMBED_DIM
    assert vector_in_index(story_id) == basis(0)
    monkeypatch.setattr(
        embeddings, "embed", lambda *args, **kwargs: [1.0, 0.0]
    )

    result = embeddings.backfill(
        model=config.EMBED_MODEL,
        version=config.EMBED_VERSION,
        representation="default",
    )

    assert result["activation"]["activated"] == 1
    assert store.serving_embedding_dimension() == 2
    assert vector_in_index(story_id) == [1.0, 0.0]
    assert store.get_embedding_index_state()["active_version"] == config.EMBED_VERSION
