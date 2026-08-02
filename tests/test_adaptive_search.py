"""Adaptive hybrid retrieval gates, second-stage budgets, and fallbacks."""
from __future__ import annotations

import time

from storybook import adaptive, config, llm, search as search_module, store
from ._helpers import basis, with_cos


def _seed(
    title: str,
    vector: list[float],
    *,
    content: str = "memory detail",
    keywords: list[str] | None = None,
) -> int:
    return store.add_story(title, content, keywords or [], vector)


class TestFastHybrid:
    def test_fast_fuses_vector_and_lexical_without_generative_llm(
        self, fake_embedder, fake_llm
    ):
        story_id = _seed(
            "SQLite lock recovery",
            with_cos(0, 0.8),
            content="configure busy timeout",
            keywords=["sqlite", "lock"],
        )
        fake_embedder.register("SQLite lock", basis(0))

        result = search_module.search(
            "SQLite lock", retrieval_mode="fast", graph_enabled=False
        )

        assert result["retrieval_mode"] == "fast"
        assert result["mode"] == "vector"  # backward-compatible lane
        assert result["transform_used"] == []
        assert result["query_plan"]["skip_reason"] == "fast_mode"
        assert fake_llm.calls == {}
        match = result["top_matches"][0]
        assert match["story_id"] == story_id
        assert "_rerank_text" not in match
        assert result["rerank_trace"]["detail_status"] == "ok"
        assert result["rerank_trace"]["details_hydrated"] == 1
        assert match["retrieval_source"] == "hybrid"
        assert {path["source"] for path in match["source_paths"]} == {
            "vector", "lexical"
        }
        assert {
            "vector_similarity", "lexical_score", "rrf_score",
            "fusion_score", "rerank_score", "final_score",
        } <= set(match["score_components"])
        assert {
            "lexical", "fusion", "rerank", "total"
        } <= set(result["latency_ms"])

    def test_result_cache_isolated_by_retrieval_mode(
        self, fake_embedder, fake_llm
    ):
        _seed("exact incident", basis(0))
        fake_embedder.register("exact incident", basis(0))

        fast = search_module.search(
            "exact incident", retrieval_mode="fast", graph_enabled=False
        )
        auto = search_module.search(
            "exact incident", retrieval_mode="auto", graph_enabled=False
        )

        assert fast["mode"] == "vector"
        assert auto["mode"] == "vector"
        assert auto["retrieval_mode"] == "auto"
        assert fake_llm.calls == {}


class TestAutoGate:
    def test_simple_exact_query_does_not_trigger_transformation(
        self, fake_embedder, fake_llm
    ):
        story_id = _seed("exact incident", basis(0))
        fake_embedder.register("exact incident", basis(0))

        result = search_module.search(
            "exact incident", retrieval_mode="auto", graph_enabled=False
        )

        assert result["top_matches"][0]["story_id"] == story_id
        assert result["query_plan"]["should_transform"] is False
        assert result["query_plan"]["skip_reason"] == (
            "simple_high_confidence_query"
        )
        assert fake_llm.calls == {}

    def test_zero_result_auto_hyde_recovers_memory(
        self, fake_embedder, fake_llm
    ):
        story_id = _seed("hidden target", basis(0), content="target solution")
        fake_embedder.register("vague wording", basis(5))
        fake_embedder.register("hypothetical target solution", basis(0))
        fake_llm.transformation = {
            "rewrite": "",
            "queries": [],
            "hypothetical_document": "hypothetical target solution",
        }

        result = search_module.search(
            "vague wording", retrieval_mode="auto", graph_enabled=False
        )

        assert result["query_plan"]["should_transform"] is True
        assert "zero_results" in result["query_plan"]["trigger_reasons"]
        assert result["transform_used"] == ["hyde"]
        assert fake_llm.calls["transform_search_query"] == 1
        assert result["top_matches"][0]["story_id"] == story_id
        assert result["top_matches"][0]["retrieval_source"] == (
            "transformed_query"
        )
        assert any(
            path["source"] == "transformed_query"
            and path["transform"] == "hyde"
            for path in result["top_matches"][0]["source_paths"]
        )

    def test_transformation_can_be_disabled_with_explainable_decision(
        self, fake_embedder, fake_llm
    ):
        _seed("hidden target", basis(0))
        fake_embedder.register("no hit", basis(5))

        result = search_module.search(
            "no hit",
            retrieval_mode="auto",
            transform_enabled=False,
            graph_enabled=False,
        )

        assert result["top_matches"] == []
        assert result["query_plan"]["transform_enabled"] is False
        assert result["query_plan"]["skip_reason"] == "transformation_disabled"
        assert fake_llm.calls == {}

    def test_transform_timeout_returns_fast_fallback_with_reason(
        self, fake_embedder, monkeypatch
    ):
        story_id = _seed("weak fallback", with_cos(0, 0.55))
        fake_embedder.register("weak query", basis(0))
        monkeypatch.setattr(config, "QUERY_AUTO_SECOND_STAGE_TIMEOUT_SECONDS", 0.01)

        def slow_transform(*args, **kwargs):
            time.sleep(0.1)
            return {"rewrite": "x", "queries": [], "hypothetical_document": ""}

        monkeypatch.setattr(
            search_module.llm, "transform_search_query", slow_transform
        )
        result = search_module.search(
            "weak query", retrieval_mode="auto", graph_enabled=False
        )

        assert result["top_matches"][0]["story_id"] == story_id
        assert result["degraded"] is True
        assert "query_transform_timeout" in result["degraded_reasons"]
        assert result["result_state"] == "degraded_results"


class TestDeepAndReranker:
    def test_deep_uses_all_transforms_and_deep_graph_budget(
        self, fake_embedder, fake_llm
    ):
        story_id = _seed("deep target", basis(0))
        fake_embedder.register("deep question", basis(0))
        for text in ("deep rewrite", "deep subquery", "deep hypothetical"):
            fake_embedder.register(text, basis(0))
        fake_llm.transformation = {
            "rewrite": "deep rewrite",
            "queries": ["deep subquery"],
            "hypothetical_document": "deep hypothetical",
        }

        result = search_module.search(
            "deep question", retrieval_mode="deep", graph_enabled=True
        )

        assert result["top_matches"][0]["story_id"] == story_id
        assert result["query_plan"]["trigger_reasons"][0] == "explicit_deep"
        assert result["transform_used"] == ["rewrite", "multi_query", "hyde"]
        assert result["graph_trace"]["budgets"] == {
            "max_hops": config.GRAPH_DEEP_MAX_HOPS,
            "max_paths": config.GRAPH_DEEP_MAX_PATHS,
            "fan_out": config.GRAPH_DEEP_FAN_OUT,
            "time_ms": config.GRAPH_DEEP_TIME_BUDGET_MS,
            "tokens": config.GRAPH_DEEP_TOKEN_BUDGET,
        }

    def test_reranker_timeout_and_circuit_never_hide_results(
        self, fake_embedder, monkeypatch
    ):
        story_id = _seed("rerank target", basis(0))
        fake_embedder.register("rerank query", basis(0))
        monkeypatch.setattr(config, "RERANK_TIMEOUT_SECONDS", 0.005)
        monkeypatch.setattr(config, "RERANK_FAILURE_THRESHOLD", 2)

        def slow_rerank(*args, **kwargs):
            time.sleep(0.05)
            return []

        monkeypatch.setattr(adaptive, "_local_rerank", slow_rerank)
        first = search_module.search(
            "rerank query", graph_enabled=False
        )
        second = search_module.search(
            "rerank query", graph_enabled=False
        )
        third = search_module.search(
            "rerank query", graph_enabled=False
        )

        for result in (first, second, third):
            assert result["top_matches"][0]["story_id"] == story_id
            assert result["result_state"] == "degraded_results"
        assert first["rerank_trace"]["status"] == "timeout"
        assert second["rerank_trace"]["status"] == "timeout"
        assert third["rerank_trace"]["status"] == "circuit_open"
        assert third["degraded_reason"] == "reranker_circuit_open"

    def test_local_reranker_uses_bounded_candidate_detail(self):
        rows = [
            {
                "story_id": 1,
                "title": "generic title",
                "abstract": "generic abstract",
                "content": "unrelated outcome",
                "score": 0.80,
            },
            {
                "story_id": 2,
                "title": "another generic title",
                "abstract": "another generic abstract",
                "content": "memory dropped from 2GB to 50MB",
                "score": 0.79,
            },
        ]

        reranked = adaptive._local_rerank(
            "I remember only the outcome: memory dropped from 2GB to 50MB", rows
        )

        assert reranked[0]["story_id"] == 2
        assert reranked[0]["score_components"]["rerank_overlap"] > 0


class TestTransformationParser:
    def test_llm_transform_parser_honours_selection_bounds_and_timeout(
        self, monkeypatch
    ):
        captured = {}

        def fake_chat(prompt, system="", **kwargs):
            captured.update(kwargs)
            return (
                '{"rewrite":" normalized query ",'
                '"queries":["q1","q1","original","q2","q3","q4"],'
                '"hypothetical_document":"problem action outcome"}'
            )

        monkeypatch.setattr(llm, "_chat", fake_chat)
        result = llm.transform_search_query(
            "original", ["rewrite", "multi_query"], timeout_seconds=1.25
        )

        assert result == {
            "rewrite": "normalized query",
            "queries": ["q1", "q2", "q3"],
            "hypothetical_document": "",
        }
        assert captured["timeout_seconds"] == 1.25
        assert captured["num_predict"] == 384

    def test_llm_transform_parser_rejects_invalid_payload(self, monkeypatch):
        monkeypatch.setattr(llm, "_chat", lambda *args, **kwargs: "not json")
        assert llm.transform_search_query(
            "q", ["hyde"], timeout_seconds=1.0
        ) is None
