"""v0.3 召回反馈飞轮：埋点 → 点击 → 离线采纳判定 → 排名先验 / 共同采纳边。"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from click.testing import CliRunner

from storybook import (
    config, feedback, llm, mcp_server, processor, search as search_module, store,
    story_v2,
)
from storybook.cli import cli
from ._helpers import basis, with_cos


def _seed(title, vector, keywords=None):
    return store.add_story(title, f"{title} 内容", keywords or [title], vector)


def _events():
    db = store.get_db(load_vector_extension=False)
    try:
        return [dict(r) for r in db.execute("SELECT * FROM recall_events")]
    finally:
        db.close()


def _exposures():
    db = store.get_db(load_vector_extension=False)
    try:
        return [dict(r) for r in db.execute(
            "SELECT * FROM recall_exposures ORDER BY event_id, rank"
        )]
    finally:
        db.close()


def _co_recall(a, b):
    return next(
        (e for e in store.get_edges(a)
         if e["related_id"] == b and e["edge_type"] == "co_recall"),
        None,
    )


def test_clean_keywords_drops_instruction_noise_and_keeps_real_terms():
    cleaned = story_v2.clean_keywords([
        "AGENTS.md", "<INSTRUCTIONS>", "#", "storybook.recall",
        "mcp__storybook__get_story", "记忆检索", "SpiceDB", "Storybook", "SpiceDB",
    ])
    assert cleaned == ["SpiceDB", "Storybook"]


def test_mcp_recall_logs_exposure_without_edge_reinforcement(fake_embedder):
    a = _seed("A", with_cos(0, 0.9))
    b = _seed("B", with_cos(0, 0.85))
    fake_embedder.register("q", basis(0))

    out = mcp_server.recall_memories("q", telemetry_origin="mcp_recall")
    assert feedback.flush_feedback(timeout=1.0)

    events = _events()
    assert [(e["origin"], e["query"], e["result_count"]) for e in events] == [
        ("mcp_recall", "q", 2)
    ]
    assert events[0]["request_id"] == out["request_id"]
    assert [x["story_id"] for x in _exposures()] == [a, b]
    # v0.3 默认：曝光只计访问，不再凭共同曝光建 co_recall 边。
    assert _co_recall(a, b) is None
    assert store.get_story(a)["access_count"] == 1


def test_plain_search_does_not_log_telemetry(fake_embedder):
    _seed("A", basis(0))
    fake_embedder.register("q", basis(0))
    search_module.search("q")
    mcp_server.recall_memories("q")  # 无 origin：CLI / eval / benchmark 路径
    assert feedback.flush_feedback(timeout=1.0)
    assert _events() == []


def test_recall_payload_is_compact_by_default(fake_embedder):
    _seed("A", basis(0))
    fake_embedder.register("q", basis(0))
    compact = mcp_server.recall_memories("q")
    verbose = mcp_server.recall_memories("q", verbose=True)
    for key in ("rerank_trace", "query_plan", "latency_ms", "graph_trace"):
        assert key not in compact
        assert key in verbose
    assert "score_components" not in compact["matches"][0]
    assert "score_components" in verbose["matches"][0]
    assert len(json.dumps(compact)) < len(json.dumps(verbose))


def test_get_story_click_marks_latest_exposure(fake_embedder):
    a = _seed("A", basis(0))
    fake_embedder.register("q", basis(0))
    mcp_server.recall_memories("q", telemetry_origin="mcp_recall")
    assert feedback.flush_feedback(timeout=1.0)

    mcp_server.get_story_detail(a, record_click=True)
    assert feedback.flush_feedback(timeout=1.0)
    assert _exposures()[0]["clicked_at"] is not None


def test_click_without_prior_exposure_is_ignored(fake_embedder):
    a = _seed("A", basis(0))
    assert store.record_story_click(a) is False


def _session_with_recall(raw, started, captured, workspace=None):
    envelope = {
        "session": {"started_at": started.isoformat()},
        "captured_at": captured.isoformat(),
        "workspace": {"path_fingerprint": workspace},
    }
    return {"id": 0, "raw_content": raw, "context_json": json.dumps(envelope)}


def test_dream_cycle_judges_adoption_and_learns_from_co_adoption(
    fake_embedder, fake_llm, monkeypatch
):
    a = _seed("A", with_cos(0, 0.9))
    b = _seed("B", with_cos(0, 0.85))
    c = _seed("C", with_cos(0, 0.8))
    fake_embedder.register("q", basis(0))
    mcp_server.recall_memories("q", telemetry_origin="mcp_recall")
    assert feedback.flush_feedback(timeout=1.0)

    calls = []

    def judge(content, recalls):
        calls.append(recalls)
        return {"verdicts": {a: ("used", "沿用了命令"), b: ("used", "避坑"),
                             c: ("misleading", "过时")},
                "missing": "缺少部署回滚经验"}

    monkeypatch.setattr(llm, "judge_recall_adoption", judge)
    now = datetime.now(UTC)
    raw = "[user] 修复\n[tool] recall\n[assistant] done"
    sid = store.add_session("codex", raw, "修复", context={
        "session": {"started_at": (now - timedelta(minutes=5)).isoformat()},
    })
    fake_embedder.set_default(basis(9))
    assert processor.process_session(sid) is not None

    assert len(calls) == 1 and {s["story_id"] for s in calls[0][0]["stories"]} == {a, b, c}
    verdicts = {x["story_id"]: x["verdict"] for x in _exposures()}
    assert verdicts == {a: "used", b: "used", c: "misleading"}
    event = _events()[0]
    assert event["judged_session_id"] == sid
    assert event["missing_note"] == "缺少部署回滚经验"
    assert _co_recall(a, b) is not None
    assert _co_recall(a, c) is None

    # 已判定的事件不会被再次判定
    session = store.get_session(sid)
    assert processor._prepare_adoption(session) is None


def test_session_without_marker_skips_non_hook_events(fake_embedder, monkeypatch):
    _seed("A", basis(0))
    fake_embedder.register("q", basis(0))
    mcp_server.recall_memories("q", telemetry_origin="mcp_recall")
    assert feedback.flush_feedback(timeout=1.0)
    monkeypatch.setattr(
        llm, "judge_recall_adoption",
        lambda *_: pytest.fail("无工具标记的会话不应调用模型"),
    )
    now = datetime.now(UTC)
    session = _session_with_recall("[user] hi", now - timedelta(minutes=5), now)
    from storybook import adoption
    assert adoption.prepare_judgment(session) is None


def test_adoption_prior_reorders_close_candidates(fake_embedder):
    good = _seed("good", with_cos(0, 0.86))
    bad = _seed("bad", with_cos(0, 0.88))
    fake_embedder.register("q", basis(0))
    assert [m["story_id"] for m in search_module.search("q")["top_matches"]][:2] == [bad, good]

    for _ in range(4):
        event_id = store.record_recall_event(
            request_id=f"r{_}", origin="mcp_recall", query="q",
            exposures=[(bad, 1, 0.88), (good, 2, 0.86)],
        )
        store.apply_recall_verdicts(
            event_id, 1, {bad: ("misleading", ""), good: ("used", "")}
        )
    from storybook import query_cache
    query_cache.clear()
    result = search_module.search("q")
    assert [m["story_id"] for m in result["top_matches"]][:2] == [good, bad]
    assert result["top_matches"][0]["score_components"]["adoption_prior"] > 0.5


def test_feedback_report_and_keyword_cleanup_cli(fake_embedder):
    a = _seed("A", basis(0), keywords=["AGENTS.md", "SpiceDB", "storybook.recall"])
    event_id = store.record_recall_event(
        request_id="r1", origin="mcp_prime", query="q", exposures=[(a, 1, 0.9)],
    )
    store.record_recall_event(
        request_id="r2", origin="mcp_recall", query="nothing", exposures=[],
    )
    store.apply_recall_verdicts(event_id, 1, {a: ("used", "ok")})

    runner = CliRunner()
    report = runner.invoke(cli, ["memory", "feedback", "--json"])
    assert report.exit_code == 0, report.output
    data = json.loads(report.output)
    assert data["events"] == 2 and data["empty_events"] == 1
    assert data["funnel"]["adoption_rate"] == 1.0
    assert data["empty_queries"][0]["query"] == "nothing"

    preview = runner.invoke(cli, ["memory", "clean-keywords", "--json"])
    assert json.loads(preview.output)["stories"] == 1
    assert store.get_story(a)["keywords"] == ["AGENTS.md", "SpiceDB", "storybook.recall"]

    fake_embedder.set_default(basis(0))
    applied = runner.invoke(cli, ["memory", "clean-keywords", "--apply", "--json"])
    assert json.loads(applied.output)["applied"] == 1
    assert store.get_story(a)["keywords"] == ["SpiceDB"]


def test_judge_recall_adoption_filters_unknown_ids(monkeypatch):
    captured = {}

    def fake_call(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        result = {"verdicts": [
            {"story_id": 7, "verdict": "used", "reason": " 用了 "},
            {"story_id": 99, "verdict": "used", "reason": "幻觉 id"},
        ], "missing": ""}
        assert llm._matches_schema(result, schema)
        return result

    monkeypatch.setattr(llm, "_structured_call", fake_call)
    out = llm.judge_recall_adoption("[user] x", [
        {"query": "q", "origin": "mcp_recall",
         "stories": [{"story_id": 7, "title": "T", "abstract": "A"}]},
    ])
    assert out == {"verdicts": {7: ("used", "用了")}, "missing": ""}
    assert "story_id=7" in captured["prompt"]
    assert llm.judge_recall_adoption("x", []) is None
