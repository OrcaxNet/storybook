"""pytest 共享夹具：隔离的临时数据库 + 可注入的 Ollama mock。

工具函数与桩类位于 ``tests/_helpers.py``；本文件只定义夹具。
"""
from __future__ import annotations

import pytest

from ._helpers import FakeEmbedder, FakeLLM, config, embeddings_mod, llm_mod, store
from storybook import adaptive, feedback, query_cache
from storybook.profiles import PlatformRoots, ProfileRegistry


def _profile_roots(tmp_path):
    return PlatformRoots(
        config=tmp_path / "profile-config",
        data=tmp_path / "profile-data",
        cache=tmp_path / "profile-cache",
        state=tmp_path / "profile-state",
        logs=tmp_path / "profile-logs",
    )


@pytest.fixture(autouse=True)
def tmp_db(tmp_path):
    """隔离每个测试的 Profile registry、数据库和性能日志。"""
    assert feedback.flush_feedback(timeout=2.0)
    query_cache.clear()
    adaptive.reset_reranker_circuit()
    embeddings_mod.mark_model_cold()
    isolation = pytest.MonkeyPatch()
    registry = ProfileRegistry(
        tmp_path / "profile-config" / "profiles.json",
        roots=_profile_roots(tmp_path),
    )
    isolation.setattr(config, "PROFILE_REGISTRY", registry)
    config.refresh_profile(create=False)
    db_path = tmp_path / "test_memory.db"
    isolation.setattr(config, "DB_PATH", db_path)
    isolation.setattr(
        config, "PERFORMANCE_LOG_PATH", tmp_path / "query_performance.jsonl"
    )
    try:
        store.init_db()
        yield db_path
    finally:
        try:
            flushed = feedback.flush_feedback(timeout=2.0)
            query_cache.clear()
            adaptive.reset_reranker_circuit()
            embeddings_mod.mark_model_cold()
        finally:
            isolation.undo()
            # ``refresh_profile`` restores every derived path and the persisted
            # flag from the host registry without creating or modifying it.
            config.refresh_profile(create=False)
        assert flushed


@pytest.fixture
def fake_embedder(monkeypatch):
    """替换 ``embeddings.embed`` 为可控桩。processor / search 均通过模块属性访问，替换即生效。"""
    fe = FakeEmbedder()
    monkeypatch.setattr(embeddings_mod, "embed", fe)
    return fe


@pytest.fixture
def fake_llm(monkeypatch):
    """替换 ``llm`` 全部对外函数为可控桩。"""
    fl = FakeLLM()
    monkeypatch.setattr(llm_mod, "extract_keywords", fl.extract_keywords)
    monkeypatch.setattr(llm_mod, "summarize_session", fl.summarize_session)
    monkeypatch.setattr(llm_mod, "form_stories", fl.form_stories)
    monkeypatch.setattr(llm_mod, "merge_stories", fl.merge_stories)
    monkeypatch.setattr(llm_mod, "judge_split", fl.judge_split)
    monkeypatch.setattr(llm_mod, "split_story", fl.split_story)
    monkeypatch.setattr(
        llm_mod, "transform_search_query", fl.transform_search_query
    )
    return fl
