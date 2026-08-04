from __future__ import annotations

import uuid
import threading
import time
import subprocess
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


def _git_repository(path, remote):
    path.mkdir(parents=True)
    subprocess.run(
        ["git", "init", str(path)], check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", remote],
        check=True,
        capture_output=True,
        text=True,
    )


def _set_git_remote(path, remote):
    subprocess.run(
        ["git", "-C", str(path), "remote", "set-url", "origin", remote],
        check=True,
        capture_output=True,
        text=True,
    )


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


def test_decay_is_frequency_invariant_across_one_half_life():
    once = store.add_story("once", "single decay", [], basis(0))
    daily = store.add_story("daily", "daily decay", [], basis(1))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    db = store.get_db()
    try:
        db.execute(
            """UPDATE stories
               SET access_count = 8, access_decay_at = ?
               WHERE id = ?""",
            (start.isoformat(), once),
        )
        db.commit()
    finally:
        db.close()

    store.decay_story_access_counts(
        half_life_days=30, now=start + timedelta(days=30)
    )
    once_count = store.get_story(once)["access_count"]
    db = store.get_db(load_vector_extension=False)
    try:
        db.execute(
            "UPDATE stories SET access_count = 0, access_score = 0 WHERE id = ?",
            (once,),
        )
        db.execute(
            """UPDATE stories
               SET access_count = 8, access_score = 0, access_decay_at = ?
               WHERE id = ?""",
            (start.isoformat(), daily),
        )
        db.commit()
    finally:
        db.close()
    for day in range(1, 31):
        store.decay_story_access_counts(
            half_life_days=30, now=start + timedelta(days=day)
        )

    assert once_count == 4
    assert store.get_story(daily)["access_count"] == 4


def test_project_scope_cannot_be_starved_by_global_top_n(fake_embedder):
    local_context = context.capture_context(workspace_path="/work/local")
    noise_context = context.capture_context(workspace_path="/work/noise")
    local_session = store.add_session("codex", "local", context=local_context)
    noise_session = store.add_session("codex", "noise", context=noise_context)
    for index in range(25):
        store.add_story(
            f"noise-{index}", "q cross project", ["q"], basis(0),
            source_session_ids=[noise_session],
        )
    local_story = store.add_story(
        "local", "q local project", ["q"], with_cos(0, 0.9),
        source_session_ids=[local_session],
    )
    fake_embedder.register("q", basis(0))

    result = search.search(
        "q", top_k=1, context=local_context, scope="project", graph_enabled=False
    )

    assert [item["story_id"] for item in result["top_matches"]] == [local_story]


def test_project_scope_lexical_fallback_is_filtered_before_limit(fake_embedder):
    local_context = context.capture_context(workspace_path="/work/local-lexical")
    noise_context = context.capture_context(workspace_path="/work/noise-lexical")
    local_session = store.add_session("codex", "local", context=local_context)
    noise_session = store.add_session("codex", "noise", context=noise_context)
    for index in range(25):
        store.add_story(
            f"needle-noise-{index}", "needle noise", ["needle"], basis(index),
            source_session_ids=[noise_session],
        )
    local_story = store.add_story(
        "needle-local", "needle local", ["needle"], basis(100),
        source_session_ids=[local_session],
    )
    fake_embedder.register("needle", None)

    result = search.search(
        "needle",
        top_k=1,
        context=local_context,
        scope="project",
        graph_enabled=False,
        rerank_enabled=False,
    )

    assert result["mode"] == "lexical_fallback"
    assert [item["story_id"] for item in result["top_matches"]] == [local_story]


def test_git_root_and_subdirectory_share_project_identity(tmp_path):
    repository = tmp_path / "repo"
    child = repository / "src" / "package"
    child.mkdir(parents=True)
    subprocess.run(
        ["git", "init", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )

    root_context = context.capture_context(workspace_path=repository)
    child_context = context.capture_context(workspace_path=child)

    assert context.project_identity(root_context) == context.project_identity(child_context)
    assert root_context["workspace"]["project_label"] == repository.name


def test_imported_path_and_realtime_subdirectory_use_remote_project_identity(tmp_path):
    repository = tmp_path / "payments"
    child = repository / "src" / "package"
    _git_repository(repository, "https://github.com/acme/payments.git")
    child.mkdir(parents=True)

    imported = context.normalize_envelope({"workspace": {"path": str(repository)}})
    realtime = context.capture_context(workspace_path=child)

    assert context.project_identity(imported) == context.project_identity(realtime)
    assert imported["workspace"]["repo_fingerprint"].startswith("sha256:")
    assert imported["workspace"]["path_fingerprint"].startswith("hmac-sha256:")
    assert str(repository) not in str(imported)


def test_imported_story_is_recalled_from_realtime_project_context(
    tmp_path, fake_embedder
):
    repository = tmp_path / "payments"
    child = repository / "services" / "api"
    _git_repository(repository, "https://github.com/acme/payments.git")
    child.mkdir(parents=True)
    imported = context.normalize_envelope({"workspace": {"path": str(repository)}})
    realtime = context.capture_context(workspace_path=child)
    session_id = store.add_session("codex", "imported", context=imported)
    story_id = store.add_story(
        "payment retry", "imported retry policy", ["retry"], basis(0),
        source_session_ids=[session_id],
    )
    fake_embedder.register("retry", basis(0))

    result = search.search(
        "retry", top_k=1, context=realtime, scope="project", graph_enabled=False
    )

    assert [item["story_id"] for item in result["top_matches"]] == [story_id]


def test_realtime_project_context_recalls_legacy_path_fingerprint_story(
    tmp_path, fake_embedder
):
    repository = tmp_path / "payments"
    _git_repository(repository, "https://github.com/acme/payments.git")
    realtime = context.capture_context(workspace_path=repository)
    legacy_path = context.workspace_path_hash(repository)
    legacy = context.normalize_envelope({
        "workspace": {
            "id": str(uuid.uuid5(uuid.UUID(config.PROFILE_ID), legacy_path)),
            "repo_fingerprint": legacy_path,
        }
    })
    session_id = store.add_session("codex", "legacy", context=legacy)
    story_id = store.add_story(
        "legacy retry", "legacy path-only memory", ["legacy"], basis(0),
        source_session_ids=[session_id],
    )
    fake_embedder.register("legacy", basis(0))

    assert context.project_identity(legacy) != context.project_identity(realtime)
    assert context.story_matches_project(realtime, [legacy])
    result = search.search(
        "legacy", top_k=1, context=realtime, scope="project", graph_enabled=False
    )
    assert [item["story_id"] for item in result["top_matches"]] == [story_id]


def test_remote_project_identity_keeps_different_repositories_isolated(tmp_path):
    payments = tmp_path / "payments"
    catalog = tmp_path / "catalog"
    _git_repository(payments, "https://github.com/acme/payments.git")
    _git_repository(catalog, "https://github.com/acme/catalog.git")

    payments_context = context.capture_context(workspace_path=payments)
    catalog_import = context.normalize_envelope({"workspace": {"path": str(catalog)}})

    assert not context.story_matches_project(payments_context, [catalog_import])


def test_different_remotes_at_same_path_have_authoritative_conflict(
    tmp_path, fake_embedder
):
    repository = tmp_path / "reused"
    _git_repository(repository, "https://github.com/acme/payments.git")
    payments = context.capture_context(workspace_path=repository)
    payments_session = store.add_session("codex", "payments", context=payments)
    legacy_path = context.workspace_path_hash(repository)
    legacy = context.normalize_envelope({
        "workspace": {
            "id": str(uuid.uuid5(uuid.UUID(config.PROFILE_ID), legacy_path)),
            "repo_fingerprint": legacy_path,
        }
    })
    legacy_session = store.add_session("codex", "legacy", context=legacy)
    payments_story = store.add_story(
        "needle payments", "old repository memory", ["needle"], basis(0),
        source_session_ids=[payments_session, legacy_session],
    )
    _set_git_remote(repository, "https://github.com/acme/catalog.git")
    catalog = context.capture_context(workspace_path=repository)
    fake_embedder.register("needle", basis(0))

    assert context.project_identity(payments) != context.project_identity(catalog)
    assert not context.story_matches_project(catalog, [payments, legacy])
    result = search.search(
        "needle", top_k=1, context=catalog, scope="project", graph_enabled=False
    )
    assert payments_story not in [item["story_id"] for item in result["top_matches"]]


def test_different_remotes_at_same_path_are_excluded_from_lexical_fallback(
    tmp_path, fake_embedder
):
    repository = tmp_path / "reused"
    _git_repository(repository, "https://github.com/acme/payments.git")
    payments = context.capture_context(workspace_path=repository)
    payments_session = store.add_session("codex", "payments", context=payments)
    store.add_story(
        "needle payments", "old repository memory", ["needle"], basis(0),
        source_session_ids=[payments_session],
    )
    _set_git_remote(repository, "https://github.com/acme/catalog.git")
    catalog = context.capture_context(workspace_path=repository)
    fake_embedder.register("needle", None)

    result = search.search(
        "needle", top_k=1, context=catalog, scope="project", graph_enabled=False
    )

    assert result["mode"] == "lexical_fallback"
    assert result["top_matches"] == []


def test_graph_expansion_respects_remote_conflict_at_same_path(
    tmp_path, fake_embedder
):
    repository = tmp_path / "reused"
    _git_repository(repository, "https://github.com/acme/payments.git")
    payments = context.capture_context(workspace_path=repository)
    payments_session = store.add_session("codex", "payments", context=payments)
    payments_story = store.add_story(
        "payments", "cross-project graph memory", [], basis(5),
        source_session_ids=[payments_session],
    )
    _set_git_remote(repository, "https://github.com/acme/catalog.git")
    catalog = context.capture_context(workspace_path=repository)
    catalog_session = store.add_session("codex", "catalog", context=catalog)
    catalog_story = store.add_story(
        "catalog", "needle current project", ["needle"], basis(0),
        source_session_ids=[catalog_session],
    )
    store.add_or_update_edge(catalog_story, payments_story, 1.0, "causal")
    fake_embedder.register("needle", basis(0))

    result = search.search(
        "needle", top_k=3, context=catalog, scope="project", graph_enabled=True
    )

    assert [item["story_id"] for item in result["top_matches"]] == [catalog_story]


def test_same_remote_matches_across_different_clones(tmp_path):
    first = tmp_path / "first" / "payments"
    second = tmp_path / "second" / "payments"
    remote = "https://github.com/acme/payments.git"
    _git_repository(first, remote)
    _git_repository(second, remote)

    first_context = context.capture_context(workspace_path=first)
    second_context = context.capture_context(workspace_path=second)

    assert context.story_matches_project(second_context, [first_context])


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
