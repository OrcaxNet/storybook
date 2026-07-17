"""pytest 共享夹具：隔离的临时数据库 + 可注入的 Ollama mock。

工具函数与桩类位于 ``tests/_helpers.py``；本文件只定义夹具。
"""
from __future__ import annotations

import pytest

from ._helpers import config, store, embeddings_mod, llm_mod, FakeEmbedder, FakeLLM


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """每个测试一个隔离的临时数据库（autouse）：重定向 ``config.DB_PATH`` 到 tmp_path。"""
    db_path = tmp_path / "test_memory.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    store.init_db()
    return db_path


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
    monkeypatch.setattr(llm_mod, "merge_stories", fl.merge_stories)
    monkeypatch.setattr(llm_mod, "judge_split", fl.judge_split)
    monkeypatch.setattr(llm_mod, "split_story", fl.split_story)
    return fl
