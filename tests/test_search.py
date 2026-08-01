"""检索激活层测试：阈值过滤、关联激活、共同召回提权（每对仅一次）。

全程 mock ``embeddings.embed``（返回精确构造的查询向量），
``store.search_by_vector`` / 关联边读写走真实 SQLite，验证检索语义。
"""
from __future__ import annotations

import pytest

from storybook import store, search as search_module, config
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
        assert _weight(a, b) == pytest.approx(0.4, abs=1e-6)   # 第一次 +0.1

        search_module.search("q", top_k=3)
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
    def test_embedding_failure_returns_error(self, fake_embedder):
        fake_embedder.register("q", None)   # 模拟向量生成失败
        result = search_module.search("q", top_k=3)
        assert result["top_matches"] == []
        assert "error" in result

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
        assert "标题A" in out
        assert "内容A" in out
        assert "k1" in out
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
