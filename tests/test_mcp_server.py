"""MCP server 测试。

分两层：
  - 工具核心逻辑（recall_memories / get_story_detail / get_stats_overview 及裁剪函数）：
    直接调用模块级函数，**不依赖 mcp SDK**，复用 conftest 的隔离 DB + mock embedder。
  - FastMCP 装配（create_server）：需基础安装自带的 mcp SDK；
    ``pytest.importorskip`` 让核心逻辑测试在裁剪依赖环境中仍可运行。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from storybook import store
from storybook import context as context_module
from storybook import mcp_server
from ._helpers import basis, with_cos


def _seed(title: str, vec: list[float], keywords: list[str] | None = None) -> int:
    return store.add_story(title, "c", keywords or ["k"], vec)


# ═══════════════════════════════════════════════
#  裁剪函数
# ═══════════════════════════════════════════════

class TestTrimRelated:
    def test_drops_content_and_keeps_summary_fields(self):
        out = mcp_server._trim_related([
            {"story_id": 1, "title": "A", "content": "全文...", "weight": 0.5, "edge_type": "semantic"},
        ])
        assert out == [{"story_id": 1, "title": "A", "weight": 0.5, "edge_type": "semantic"}]
        assert "content" not in out[0]

    def test_handles_id_keyed_rows(self):
        """store.get_related_stories 返回的行用 id（非 story_id）作为主键。"""
        out = mcp_server._trim_related([
            {"id": 7, "title": "B", "content": "x", "embedding": [0.1] * 4, "weight": 0.9, "edge_type": "parent_child"},
        ])
        assert out[0]["story_id"] == 7
        assert "content" not in out[0]
        assert "embedding" not in out[0]

    def test_empty_input(self):
        assert mcp_server._trim_related([]) == []
        assert mcp_server._trim_related(None) == []


class TestBuildRecallResult:
    def test_shape_and_count(self):
        result = mcp_server._build_recall_result({
            "query": "q", "keywords": [],
            "top_matches": [
                {"story_id": 1, "title": "A", "content": "c1", "keywords": ["k1"],
                 "similarity": 0.88, "related": [{"story_id": 2, "title": "B", "content": "c2", "weight": 0.3, "edge_type": "semantic"}]},
            ],
        })
        assert result["query"] == "q"
        assert result["count"] == 1
        m = result["matches"][0]
        assert m["story_id"] == 1
        assert m["similarity"] == 0.88
        # 主命中保留 content；related 去掉 content
        assert "content" in m
        assert m["related"][0]["story_id"] == 2
        assert "content" not in m["related"][0]

    def test_empty_matches(self):
        result = mcp_server._build_recall_result({"query": "q", "keywords": [], "top_matches": []})
        assert result["count"] == 0
        assert result["matches"] == []


# ═══════════════════════════════════════════════
#  recall_memories
# ═══════════════════════════════════════════════

class TestRecallMemories:
    def test_empty_db_returns_count_zero(self, fake_embedder):
        fake_embedder.register("q", basis(0))
        out = mcp_server.recall_memories("q")
        assert out["query"] == "q"
        assert out["count"] == 0
        assert out["matches"] == []
        # 无匹配时无 error 噪声
        assert "error" not in out

    def test_no_match_above_threshold_returns_empty(self, fake_embedder):
        _seed("low", with_cos(0, 0.3))   # sim 0.3 < 0.50 -> 过滤
        fake_embedder.register("q", basis(0))
        out = mcp_server.recall_memories("q")
        assert out["count"] == 0
        assert out["matches"] == []

    def test_returns_match_with_similarity_and_related(self, fake_embedder):
        a = _seed("A", basis(0))
        b = _seed("B", basis(5))   # 与 query 正交，不命中，但与 A 有关联
        store.add_or_update_edge(a, b, 0.6, "semantic")
        fake_embedder.register("q", basis(0))

        out = mcp_server.recall_memories("q")
        assert out["count"] == 1
        m = out["matches"][0]
        assert m["story_id"] == a
        assert m["title"] == "A"
        assert "content" in m
        assert "similarity" in m
        # related 精简：含 B，无 content
        assert len(m["related"]) == 1
        assert m["related"][0]["story_id"] == b
        assert "content" not in m["related"][0]

    def test_top_k_limit(self, fake_embedder):
        for i in range(5):
            _seed(f"s{i}", with_cos(0, 0.9 - i * 0.02))
        fake_embedder.register("q", basis(0))
        out = mcp_server.recall_memories("q", top_k=3)
        assert out["count"] == 3

    def test_optional_context_and_strict_scope_are_exposed(self, fake_embedder):
        source_context = context_module.capture_context(
            tool_type="cursor", integration_mode="log_import",
            runtime_kind="devcontainer",
        )
        session_id = store.add_session("cursor", "raw", context=source_context)
        story_id = store.add_story(
            "dev only", "c", ["k"], basis(0), source_session_ids=[session_id]
        )
        fake_embedder.register("q", basis(0))
        current = context_module.capture_context(
            tool_type="codex", integration_mode="mcp", runtime_kind="local"
        )

        soft = mcp_server.recall_memories("q", context=current)
        assert soft["matches"][0]["story_id"] == story_id
        assert soft["matches"][0]["warnings"]

        strict = mcp_server.recall_memories(
            "q", context=current, scope="strict"
        )
        assert strict["matches"] == []
        assert strict["strict_filtered"] == 1

    def test_embedding_failure_returns_degraded_state(self, fake_embedder):
        fake_embedder.register("q", None)   # 模拟向量生成失败（Ollama 不可用等）
        out = mcp_server.recall_memories("q")
        assert out["count"] == 0
        assert out["degraded"] is True
        assert out["result_state"] == "degraded_empty"
        assert out["fallback_status"] == "ok"

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            mcp_server.recall_memories("")
        with pytest.raises(ValueError):
            mcp_server.recall_memories("   ")


# ═══════════════════════════════════════════════
#  get_story_detail
# ═══════════════════════════════════════════════

class TestGetStoryDetail:
    def test_not_found_raises(self):
        with pytest.raises(ValueError):
            mcp_server.get_story_detail(9999)

    def test_returns_detail_without_embedding(self, fake_embedder):
        sid = _seed("A", basis(0), keywords=["k1", "k2"])
        out = mcp_server.get_story_detail(sid)
        assert out["story_id"] == sid
        assert out["title"] == "A"
        assert out["keywords"] == ["k1", "k2"]
        assert out["related"] == []
        # 剥离 1024 维 embedding，避免臃肿
        assert "embedding" not in out
        assert "environments" in out
        assert "applicability" in out

    def test_includes_related_trimmed(self, fake_embedder):
        a = _seed("A", basis(0))
        b = _seed("B", basis(5))
        store.add_or_update_edge(a, b, 0.7, "semantic")
        out = mcp_server.get_story_detail(a)
        assert len(out["related"]) == 1
        assert out["related"][0]["story_id"] == b
        assert "content" not in out["related"][0]
        assert "embedding" not in out["related"][0]


# ═══════════════════════════════════════════════
#  get_stats_overview
# ═══════════════════════════════════════════════

class TestGetStatsOverview:
    def test_returns_stats_dict(self):
        _seed("A", basis(0))
        _seed("B", basis(1))
        out = mcp_server.get_stats_overview()
        assert out["stories"] == 2
        assert out["edges"] == 0
        assert "sessions" in out

    def test_empty_db_returns_zeros(self):
        out = mcp_server.get_stats_overview()
        assert out["stories"] == 0
        assert "error" not in out


# ═══════════════════════════════════════════════
#  prime_context_memories（晨间简报，转调 prime.prime_context）
# ═══════════════════════════════════════════════

class TestPrimeContextMemories:
    def test_marks_unknown_tool_with_mcp_integration(self, monkeypatch):
        captured = {}

        def fake_prime_context(**kwargs):
            captured.update(kwargs)
            return {"injected": False, "briefing": ""}

        monkeypatch.setattr(
            mcp_server.prime_module, "prime_context", fake_prime_context
        )

        mcp_server.prime_context_memories(cwd="/x/proj", first_prompt="q")

        assert captured["tool_type"] == "other"
        assert captured["integration_mode"] == "mcp"

    def test_injects_on_match(self, fake_embedder):
        a = _seed("A", basis(0))
        fake_embedder.register("q", basis(0))
        out = mcp_server.prime_context_memories(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is True
        assert out["count"] == 1
        assert out["matches"][0]["story_id"] == a
        assert out["briefing"] != ""

    def test_silent_on_no_match(self, fake_embedder):
        fake_embedder.register("q", basis(0))
        out = mcp_server.prime_context_memories(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is False
        assert out["briefing"] == ""
        assert out["count"] == 0

    def test_silent_on_embedding_failure(self, fake_embedder):
        fake_embedder.register("q", None)
        out = mcp_server.prime_context_memories(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is False
        assert out["note"] is not None
        assert "storybook doctor" in out["note"]


# ═══════════════════════════════════════════════
#  FastMCP 装配（需 mcp SDK）
# ═══════════════════════════════════════════════

pytest.importorskip("mcp")


class TestServerWiring:
    def _server(self):
        return mcp_server.create_server()

    @staticmethod
    def _extract_json(res) -> dict:
        """从 call_tool 返回中取出首个 TextContent 的 JSON。

        FastMCP 1.28 直接返回 ``list[TextContent]``；其它版本可能包成带 ``.content``
        的对象。两种都兼容。
        """
        content = res.content if hasattr(res, "content") else res
        text = next(c.text for c in content if hasattr(c, "text"))
        return json.loads(text)

    def test_three_tools_registered(self):
        mcp = self._server()
        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
        assert names == {"recall", "get_story", "stats", "prime_context"}

    def test_recall_input_schema_has_query(self):
        mcp = self._server()
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        props = tools["recall"].inputSchema["properties"]
        assert "query" in props
        assert "top_k" in props
        assert "context" in props
        assert "scope" in props
        # query 必填
        assert "query" in tools["recall"].inputSchema.get("required", [])

    def test_prime_context_schema_has_cwd_and_first_prompt(self):
        mcp = self._server()
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        props = tools["prime_context"].inputSchema["properties"]
        assert "cwd" in props
        assert "first_prompt" in props
        assert "top_k" in props
        # 均可选（无必填）
        assert tools["prime_context"].inputSchema.get("required", []) == []

    def test_recall_end_to_end_via_call_tool(self, fake_embedder):
        a = _seed("A", basis(0))
        fake_embedder.register("q", basis(0))
        mcp = self._server()
        res = asyncio.run(mcp.call_tool("recall", {"query": "q"}))
        data = self._extract_json(res)
        assert data["count"] == 1
        assert data["matches"][0]["story_id"] == a

    def test_get_story_end_to_end_via_call_tool(self, fake_embedder):
        a = _seed("A", basis(0))
        mcp = self._server()
        res = asyncio.run(mcp.call_tool("get_story", {"story_id": a}))
        data = self._extract_json(res)
        assert data["story_id"] == a
        assert "embedding" not in data

    def test_stats_end_to_end_via_call_tool(self):
        mcp = self._server()
        res = asyncio.run(mcp.call_tool("stats", {}))
        data = self._extract_json(res)
        assert "stories" in data

    def test_prime_context_end_to_end_via_call_tool(self, fake_embedder):
        a = _seed("A", basis(0))
        fake_embedder.register("q", basis(0))
        mcp = self._server()
        res = asyncio.run(mcp.call_tool(
            "prime_context", {"cwd": "/x/proj", "first_prompt": "q"}))
        data = self._extract_json(res)
        assert data["injected"] is True
        assert data["count"] == 1
        assert data["matches"][0]["story_id"] == a
        assert data["briefing"] != ""

    def test_prime_context_silent_when_no_match(self, fake_embedder):
        fake_embedder.register("q", basis(0))
        mcp = self._server()
        res = asyncio.run(mcp.call_tool(
            "prime_context", {"cwd": "/x/proj", "first_prompt": "q"}))
        data = self._extract_json(res)
        assert data["injected"] is False
        assert data["briefing"] == ""
