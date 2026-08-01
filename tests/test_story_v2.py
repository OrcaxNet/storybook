"""Story v2 acceptance tests: lossless detail, semantic chunks and backfill."""
from __future__ import annotations

import json

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
