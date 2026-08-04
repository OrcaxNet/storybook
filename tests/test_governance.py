from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

from storybook import config, context, embeddings, llm, processor, search, store

from ._helpers import basis, with_cos


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_decay_archives_low_value_and_protects_frequent_story():
    low = store.add_story("low", "old low value", [], basis(0))
    high = store.add_story("high", "old frequent value", [], basis(1))
    now = datetime(2026, 8, 4, tzinfo=UTC)
    old = (now - timedelta(days=60)).isoformat()
    db = store.get_db()
    try:
        db.execute(
            "UPDATE stories SET access_count = 8, updated_at = ?, access_decay_at = ? WHERE id = ?",
            (old, old, low),
        )
        db.execute(
            "UPDATE stories SET access_count = 128, updated_at = ?, access_decay_at = ? WHERE id = ?",
            (old, old, high),
        )
        db.commit()
    finally:
        db.close()

    decay = store.decay_story_access_counts(half_life_days=30, now=now)
    result = store.archive_low_value_stories(
        max_access_count=2,
        max_edge_weight=0.25,
        min_age_days=30,
        now=now,
        dry_run=False,
    )

    assert decay == {"examined": 2, "decayed": 2}
    assert result["archived"] == 1
    assert result["candidates"][0]["story_id"] == low
    assert store.get_story(low) is None
    assert store.get_story(low, include_deleted=True)["embedding_status"] == "archived"
    assert store.get_story(high)["access_count"] == 32
    assert [item["story_id"] for item in store.search_by_vector(basis(0), top_k=5)] == [high]


def test_project_scope_removes_more_relevant_cross_project_noise(fake_embedder):
    project_a = context.capture_context(workspace_path="/work/payments")
    project_b = context.capture_context(workspace_path="/work/catalog")
    session_a = store.add_session("codex", "a", context=project_a)
    session_b = store.add_session("codex", "b", context=project_b)
    local_story = store.add_story(
        "payments", "local project", ["q"], with_cos(0, 0.9),
        source_session_ids=[session_a],
    )
    cross_story = store.add_story(
        "catalog", "cross project noise", ["q"], basis(0),
        source_session_ids=[session_b],
    )
    fake_embedder.register("q", basis(0))

    profile_result = search.search("q", top_k=2, context=project_a, graph_enabled=False)
    project_result = search.search(
        "q", top_k=2, context=project_a, scope="project", graph_enabled=False
    )

    assert profile_result["top_matches"][0]["story_id"] == cross_story
    assert [item["story_id"] for item in project_result["top_matches"]] == [local_story]
    assert project_result["strict_filtered"] >= 1


def test_llm_and_embedding_cache_skip_duplicate_provider_calls(monkeypatch):
    calls = {"llm": 0, "embedding": 0}
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")

    def fake_post(url, **kwargs):
        time.sleep(0.03)
        if url.endswith("/v1/messages"):
            calls["llm"] += 1
            return _Response({"content": [{"type": "text", "text": "cached answer"}]})
        calls["embedding"] += 1
        return _Response({"embedding": basis(0)})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(embeddings.requests, "post", fake_post)

    cold_started = time.perf_counter()
    assert llm._chat("same prompt") == "cached answer"
    assert embeddings.embed("same text") == basis(0)
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    assert llm._chat("same prompt") == "cached answer"
    assert embeddings.embed("same text") == basis(0)
    warm_seconds = time.perf_counter() - warm_started
    assert calls == {"llm": 1, "embedding": 1}
    assert warm_seconds < cold_seconds * 0.25


def test_parallel_preparation_uses_workers_and_serial_writes_remain_valid(
    monkeypatch, fake_llm, fake_embedder
):
    barrier = threading.Barrier(3)
    lock = threading.Lock()
    active = 0
    peak = 0

    def concurrent_keywords(text):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=2)
            return [text]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(config, "PROCESS_WORKERS", 3)
    monkeypatch.setattr(processor.llm, "extract_keywords", concurrent_keywords)
    for index in range(3):
        store.add_session("test", f"raw-{index}", f"problem-{index}")
        fake_embedder.register(f"raw-{index} problem-{index}", basis(index))

    result = processor.process_all_pending(verbose=False)

    assert peak == 3
    assert result == {"total": 3, "success": 3, "failed": 0}
    db = store.get_db()
    try:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'processed'"
        ).fetchone()[0] == 3
    finally:
        db.close()


def test_parallel_batch_is_significantly_faster_than_single_worker(
    monkeypatch, fake_llm, fake_embedder
):
    def delayed_keywords(_text):
        time.sleep(0.025)
        return ["kw"]

    original_form = fake_llm.form_stories

    def delayed_form(text):
        time.sleep(0.025)
        return original_form(text)

    monkeypatch.setattr(processor.llm, "extract_keywords", delayed_keywords)
    monkeypatch.setattr(processor.llm, "form_stories", delayed_form)

    def run_batch(workers: int, offset: int) -> float:
        monkeypatch.setattr(config, "PROCESS_WORKERS", workers)
        for index in range(4):
            problem = f"batch-{offset}-{index}"
            store.add_session("test", f"raw-{offset}-{index}", problem)
            fake_embedder.register(f"kw {problem}", basis(offset + index))
        started = time.perf_counter()
        result = processor.process_all_pending(verbose=False)
        elapsed = time.perf_counter() - started
        assert result == {"total": 4, "success": 4, "failed": 0}
        return elapsed

    serial_seconds = run_batch(1, 10)
    parallel_seconds = run_batch(4, 20)

    assert parallel_seconds < serial_seconds * 0.65
