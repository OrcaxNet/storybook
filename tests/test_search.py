"""检索激活层测试：阈值过滤、关联激活、共同召回提权（每对仅一次）。

全程 mock ``embeddings.embed``（返回精确构造的查询向量），
``store.search_by_vector`` / 关联边读写走真实 SQLite，验证检索语义。
"""
from __future__ import annotations

import threading
import time

import pytest

from storybook import embeddings, feedback, store, search as search_module, config
from ._helpers import basis, with_cos


def _seed(title: str, vec: list[float]) -> int:
    return store.add_story(title, "c", ["k"], vec)


# ═══════════════════════════════════════════════
#  阈值过滤
# ═══════════════════════════════════════════════

class TestThresholdFilter:
    def test_filters_below_search_threshold(self, fake_embedder):
        """sim < SIM_THRESHOLD_SEARCH(0.50) 的结果被过滤掉。"""
        low = _seed("low", with_cos(0, 0.4))    # sim 0.4 -> 过滤
        high = _seed("high", with_cos(0, 0.8))  # sim 0.8 -> 保留
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=3)
        ids = [m["story_id"] for m in result["top_matches"]]
        assert ids == [high]
        assert low not in ids

    def test_all_below_threshold_returns_empty(self, fake_embedder):
        _seed("low", with_cos(0, 0.3))
        fake_embedder.register("q", basis(0))
        result = search_module.search("q", top_k=3)
        assert result["top_matches"] == []
        assert "error" not in result

    def test_empty_db_returns_empty(self, fake_embedder):
        fake_embedder.register("q", basis(0))
        result = search_module.search("q", top_k=3)
        assert result["top_matches"] == []
        assert result["query"] == "q"

    def test_top_k_limit(self, fake_embedder):
        """命中数超过 top_k 时只返回前 top_k 条。"""
        for i in range(5):
            _seed(f"s{i}", with_cos(0, 0.9 - i * 0.02))   # sim 0.90/0.88/... 全 >= 0.5
        fake_embedder.register("q", basis(0))
        result = search_module.search("q", top_k=3)
        assert len(result["top_matches"]) == 3
        # 按相似度降序
        sims = [m["similarity"] for m in result["top_matches"]]
        assert sims == sorted(sims, reverse=True)


# ═══════════════════════════════════════════════
#  关联激活
# ═══════════════════════════════════════════════

class TestAssociationActivation:
    def test_related_stories_attached_to_match(self, fake_embedder):
        """命中 story 沿 edges 浮现关联 story；关联 story 自身未必被召回。"""
        a = _seed("A", basis(0))           # sim 1.0 -> 命中
        b = _seed("B", basis(5))           # 与 query 正交 sim 0 -> 不命中
        store.add_or_update_edge(a, b, 0.6, "semantic")
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=3)
        assert len(result["top_matches"]) == 1
        match = result["top_matches"][0]
        assert match["story_id"] == a
        related_ids = [r["story_id"] for r in match["related"]]
        assert b in related_ids
        # 命中 story 的 access_count 自增
        assert feedback.flush_feedback(timeout=1.0)
        assert store.get_story(a)["access_count"] == 1

    def test_related_ordered_by_weight_desc(self, fake_embedder):
        a = _seed("A", basis(0))
        b = _seed("B", basis(5))
        c = _seed("C", basis(6))
        store.add_or_update_edge(a, b, 0.3)
        store.add_or_update_edge(a, c, 0.9)
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=3)
        related = result["top_matches"][0]["related"]
        weights = [r["weight"] for r in related]
        assert weights == sorted(weights, reverse=True)
        assert related[0]["story_id"] == c   # 权重 0.9 最高


# ═══════════════════════════════════════════════
#  共同召回提权（每对仅一次）
# ═══════════════════════════════════════════════

class TestCommonRecallBoost:
    def test_each_co_recalled_pair_boosted_once(self, fake_embedder):
        """共同被召回的 story 两两提权，每对每次 search 仅 +WEIGHT_INCREMENT 一次。"""
        a = _seed("A", with_cos(0, 0.90))
        b = _seed("B", with_cos(0, 0.85))
        c = _seed("C", with_cos(0, 0.80))
        # 三对都预置初始边 0.3（increment 只对已存在边生效）
        store.add_or_update_edge(a, b, 0.3)
        store.add_or_update_edge(a, c, 0.3)
        store.add_or_update_edge(b, c, 0.3)
        fake_embedder.register("q", basis(0))

        search_module.search("q", top_k=3)
        assert feedback.flush_feedback(timeout=1.0)

        # 三对边各 +0.1（每对仅一次）
        def weight(x, y):
            for e in store.get_edges(x):
                if e["related_id"] == y:
                    return e["weight"]
            return None

        assert weight(a, b) == pytest.approx(0.4, abs=1e-6)
        assert weight(a, c) == pytest.approx(0.4, abs=1e-6)
        assert weight(b, c) == pytest.approx(0.4, abs=1e-6)

        # 每个 story 的 access_count 各 +1
        for sid in (a, b, c):
            assert store.get_story(sid)["access_count"] == 1

    def test_boost_is_per_call_not_accumulating_within_one_search(self, fake_embedder):
        """同一次 search 内，同一对不重复提权；多次 search 每次 +0.1。"""
        a = _seed("A", with_cos(0, 0.9))
        b = _seed("B", with_cos(0, 0.85))
        store.add_or_update_edge(a, b, 0.3)
        fake_embedder.register("q", basis(0))

        search_module.search("q", top_k=3)
        assert feedback.flush_feedback(timeout=1.0)
        assert _weight(a, b) == pytest.approx(0.4, abs=1e-6)   # 第一次 +0.1

        search_module.search("q", top_k=3)
        assert feedback.flush_feedback(timeout=1.0)
        assert _weight(a, b) == pytest.approx(0.5, abs=1e-6)   # 第二次再 +0.1

    def test_no_boost_for_unrelated_non_co_recalled(self, fake_embedder):
        """未被共同召回的 story 之间不会凭空建边/提权。"""
        a = _seed("A", basis(0))           # 命中
        b = _seed("B", basis(5))           # 不命中（sim 0）
        store.add_or_update_edge(a, b, 0.3)
        fake_embedder.register("q", basis(0))

        search_module.search("q", top_k=3)
        # 只召回 a，b 未被召回 -> a-b 边不提权
        assert _weight(a, b) == pytest.approx(0.3, abs=1e-6)


def _weight(x: int, y: int) -> float | None:
    for e in store.get_edges(x):
        if e["related_id"] == y:
            return e["weight"]
    return None


# ═══════════════════════════════════════════════
#  边界
# ═══════════════════════════════════════════════

class TestSearchBoundaries:
    def test_embedding_failure_returns_explicit_degraded_empty(self, fake_embedder):
        fake_embedder.register("q", None)   # 模拟向量生成失败
        result = search_module.search("q", top_k=3)
        assert result["top_matches"] == []
        assert result["degraded"] is True
        assert result["mode"] == "lexical_fallback"
        assert result["result_state"] == "degraded_empty"
        assert "error" not in result

    def test_result_shape(self, fake_embedder):
        a = _seed("A", basis(0))
        fake_embedder.register("q", basis(0))
        result = search_module.search("q", top_k=3)
        assert result["query"] == "q"
        assert result["keywords"] == []
        m = result["top_matches"][0]
        # 关键字段齐全
        for key in ("story_id", "title", "content", "keywords", "similarity", "related"):
            assert key in m


# ═══════════════════════════════════════════════
#  快路径缓存与 index_version
# ═══════════════════════════════════════════════

class TestFastPathCache:
    def test_result_and_vector_cache_are_invalidated_by_index_version(
        self, fake_embedder
    ):
        first = _seed("first", with_cos(0, 0.8))
        fake_embedder.register("q", basis(0))

        initial = search_module.search("q", top_k=2)
        cached = search_module.search("q", top_k=2)

        assert initial["mode"] == "vector"
        assert cached["mode"] == "cache"
        assert cached["index_version"] == initial["index_version"]
        assert fake_embedder.calls == ["q"]

        second = _seed("second", basis(0))
        refreshed = search_module.search("q", top_k=2)

        assert refreshed["mode"] == "vector"
        assert refreshed["index_version"] > initial["index_version"]
        assert refreshed["top_matches"][0]["story_id"] == second
        assert first in [item["story_id"] for item in refreshed["top_matches"]]
        # index_version 同时使 query vector cache 失效。
        assert fake_embedder.calls == ["q", "q"]

    def test_top_k_result_miss_can_reuse_query_vector_cache(self, fake_embedder):
        _seed("A", basis(0))
        _seed("B", with_cos(0, 0.9))
        fake_embedder.register("q", basis(0))

        search_module.search("q", top_k=1)
        result = search_module.search("q", top_k=2)

        assert result["mode"] == "vector"
        assert result["query_vector_cache_hit"] is True
        assert fake_embedder.calls == ["q"]

    @pytest.mark.parametrize(
        "config_field", ["EMBED_PROVIDER", "EMBED_BASE_URL", "EMBED_MODEL"]
    )
    def test_target_spec_drift_keeps_active_result_and_vector_caches(
        self, fake_embedder, monkeypatch, config_field
    ):
        story_id = store.add_story(
            "cache drift recovery phrase", "lexical fallback", [], basis(0)
        )
        fake_embedder.register("cache drift recovery phrase", basis(0))
        original_value = getattr(config, config_field)
        drifted_value = {
            "EMBED_PROVIDER": "api" if original_value != "api" else "ollama",
            "EMBED_BASE_URL": f"{original_value.rstrip('/')}/different",
            "EMBED_MODEL": f"{original_value}-different",
        }[config_field]

        initial = search_module.search("cache drift recovery phrase", top_k=1)
        assert initial["mode"] == "vector"

        def unexpected_call(*args, **kwargs):
            raise AssertionError(
                "active query vector should remain cached during target drift"
            )

        monkeypatch.setattr(config, config_field, drifted_value)
        monkeypatch.setattr(embeddings, "embed", unexpected_call)

        cached = search_module.search("cache drift recovery phrase", top_k=1)
        assert cached["mode"] == "cache"
        assert cached["top_matches"][0]["story_id"] == story_id

        active_vector = search_module.search(
            "cache drift recovery phrase", top_k=2
        )
        assert active_vector["mode"] == "vector"
        assert active_vector["query_vector_cache_hit"] is True
        assert active_vector["top_matches"][0]["story_id"] == story_id

    def test_feedback_writes_do_not_invalidate_retrieval_cache(self, fake_embedder):
        story_id = _seed("A", basis(0))
        fake_embedder.register("q", basis(0))
        before = store.get_index_version()

        search_module.search("q")
        assert feedback.flush_feedback(timeout=1.0)

        assert store.get_story(story_id)["access_count"] == 1
        assert store.get_index_version() == before
        assert search_module.search("q")["mode"] == "cache"

    def test_fast_path_never_calls_generative_llm(self, fake_embedder, fake_llm):
        _seed("A", basis(0))
        fake_embedder.register("q", basis(0))

        search_module.search("q")

        assert fake_llm.calls == {}

    def test_first_query_uses_cold_budget_then_switches_to_warm_budget(
        self, monkeypatch
    ):
        _seed("A", basis(0))
        monkeypatch.setattr(config, "QUERY_COLD_TIMEOUT_SECONDS", 0.4)
        monkeypatch.setattr(config, "QUERY_WARM_TIMEOUT_SECONDS", 0.2)
        observed = []

        def capture_embed(text, **kwargs):
            observed.append(kwargs["timeout_seconds"])
            return basis(0)

        monkeypatch.setattr(embeddings, "embed", capture_embed)

        search_module.search("cold-query")
        search_module.search("warm-query")

        assert observed == [0.4, 0.2]


# ═══════════════════════════════════════════════
#  词法降级、硬超时与异步反馈
# ═══════════════════════════════════════════════

class TestLexicalFallback:
    def test_abstract_is_indexed_for_embedding_degradation(self, fake_embedder):
        story_id = store.add_story(
            "unrelated title", "unrelated detail", [], basis(0),
            abstract="unique abstract recovery phrase",
        )
        fake_embedder.register("unique abstract recovery phrase", None)

        result = search_module.search("unique abstract recovery phrase")

        assert result["top_matches"][0]["story_id"] == story_id
        assert result["top_matches"][0]["content"] == (
            "unique abstract recovery phrase"
        )

    def test_embedding_unavailable_returns_fts_keyword_results(self, fake_embedder):
        story_id = store.add_story(
            "SQLite 锁冲突排查",
            "WAL 模式下先检查 busy timeout 与长事务。",
            ["sqlite", "database locked"],
            basis(0),
        )
        fake_embedder.register("SQLite 锁冲突", None)

        result = search_module.search("SQLite 锁冲突")

        assert result["mode"] == "lexical_fallback"
        assert result["degraded"] is True
        assert result["degraded_reason"] == "embedding_unavailable"
        assert result["result_state"] == "degraded_results"
        assert result["fallback_status"] == "ok"
        assert result["top_matches"][0]["story_id"] == story_id
        assert result["top_matches"][0]["retrieval_source"] == "lexical"

    def test_normal_empty_and_degraded_empty_are_distinguishable(self, fake_embedder):
        _seed("unrelated", basis(0))
        fake_embedder.register("normal-empty", basis(5))
        fake_embedder.register("degraded-empty", None)

        normal = search_module.search("normal-empty")
        degraded = search_module.search("degraded-empty")

        assert normal["result_state"] == "no_match"
        assert normal["degraded"] is False
        assert degraded["result_state"] == "degraded_empty"
        assert degraded["degraded"] is True

    def test_story_update_is_visible_to_fts_fallback(self, fake_embedder):
        story_id = _seed("old title", basis(0))
        store.update_story(story_id, title="new searchable phrase")
        fake_embedder.register("new searchable phrase", None)

        result = search_module.search("new searchable phrase")

        assert [item["story_id"] for item in result["top_matches"]] == [story_id]

    def test_warm_embedding_hard_timeout_falls_back_without_waiting_for_worker(
        self, monkeypatch
    ):
        story_id = store.add_story(
            "timeout fallback", "recover through keywords", ["timeout"], basis(0)
        )
        embeddings.mark_model_used()
        monkeypatch.setattr(config, "QUERY_WARM_TIMEOUT_SECONDS", 0.02)
        monkeypatch.setattr(config, "QUERY_FALLBACK_TIMEOUT_SECONDS", 0.05)

        def slow_embed(*args, **kwargs):
            time.sleep(0.2)
            return basis(0)

        monkeypatch.setattr(embeddings, "embed", slow_embed)
        started = time.perf_counter()
        result = search_module.search("timeout fallback")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.12
        assert result["degraded_reason"] == "embedding_timeout"
        assert result["top_matches"][0]["story_id"] == story_id

    def test_fallback_has_its_own_hard_timeout(self, fake_embedder, monkeypatch):
        fake_embedder.register("q", None)
        monkeypatch.setattr(config, "QUERY_FALLBACK_TIMEOUT_SECONDS", 0.02)

        def slow_fallback(*args, **kwargs):
            time.sleep(0.2)
            return []

        monkeypatch.setattr(store, "search_by_lexical", slow_fallback)
        started = time.perf_counter()
        result = search_module.search("q")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.12
        assert result["fallback_status"] == "timeout"
        assert result["result_state"] == "degraded_unavailable"

    def test_feedback_write_does_not_block_query(
        self, fake_embedder, monkeypatch
    ):
        story_id = _seed("A", basis(0))
        fake_embedder.register("q", basis(0))
        entered = threading.Event()
        release = threading.Event()
        original = store.apply_recall_feedback

        def blocked_feedback(story_ids, *, db_path=None):
            entered.set()
            release.wait(timeout=1.0)
            original(story_ids, db_path=db_path)

        monkeypatch.setattr(store, "apply_recall_feedback", blocked_feedback)
        started = time.perf_counter()
        result = search_module.search("q")
        elapsed = time.perf_counter() - started

        assert result["top_matches"][0]["story_id"] == story_id
        assert elapsed < 0.1
        assert entered.wait(timeout=0.5)
        assert store.get_story(story_id)["access_count"] == 0
        release.set()
        assert feedback.flush_feedback(timeout=1.0)
        assert store.get_story(story_id)["access_count"] == 1


# ═══════════════════════════════════════════════
#  结果格式化
# ═══════════════════════════════════════════════

class TestFormatResult:
    def test_format_empty_result(self):
        out = search_module.format_search_result(
            {"query": "q", "keywords": [], "top_matches": []}
        )
        assert "未找到" in out
        assert "q" in out

    def test_format_with_matches_and_related(self):
        result = {
            "query": "q",
            "keywords": [],
            "top_matches": [
                {
                    "story_id": 1, "title": "标题A", "content": "内容A",
                    "keywords": ["k1"], "similarity": 0.88,
                    "related": [
                        {"story_id": 2, "title": "关联B", "content": "B内容",
                         "weight": 0.9, "edge_type": "parent_child"},
                    ],
                }
            ],
        }
        out = search_module.format_search_result(result)
        assert "📌 #1 标题A" in out
        assert "标题A" in out
        assert "内容A" in out
        assert "k1" in out
        assert "🔗 #2" in out
        assert "关联B" in out
        assert "找到 1 条匹配记忆" in out

    def test_format_shows_applicability_and_environment_warning(self):
        result = {
            "query": "q",
            "keywords": [],
            "top_matches": [{
                "story_id": 1,
                "title": "容器经验",
                "content": "内容",
                "keywords": [],
                "similarity": 0.8,
                "environment": {
                    "tool": {"type": "cursor"},
                    "workspace": {"project_label": "payments"},
                    "runtime": {"kind": "devcontainer"},
                    "device": {"os_family": "linux", "arch": "arm64"},
                },
                "applicability": {
                    "applies_when": [{"runtime_kind": ["devcontainer"]}],
                    "excludes_when": ["k8s_coredns"],
                },
                "warnings": ["architecture differs: current=x86_64, story=arm64"],
                "related": [],
            }],
        }
        out = search_module.format_search_result(result)
        assert "来源环境" in out
        assert "适用于" in out
        assert "不适用于" in out
        assert "当前环境差异" in out
