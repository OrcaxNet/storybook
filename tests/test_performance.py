"""查询性能观测与 benchmark 测试：确定性时钟、隔离文件/DB、无真实 Ollama。"""
from __future__ import annotations

import json
from itertools import chain

import pytest
from click.testing import CliRunner

from storybook import config, feedback, perf_benchmark, performance, search, store
from storybook.cli import cli

from ._helpers import basis


def _seed(title: str = "A", content: str = "content") -> int:
    return store.add_story(title, content, ["k"], basis(0))


class TestQueryDiagnostics:
    def test_search_log_excludes_query_story_content_paths_and_urls(self, fake_embedder):
        query = "SECRET raw query https://example.invalid/repo.git"
        content = "SECRET Story body /Users/alice/private/project"
        _seed(content=content)
        fake_embedder.register(query, basis(0))

        result = search.search(query)

        log_text = config.PERFORMANCE_LOG_PATH.read_text(encoding="utf-8")
        assert result["request_id"] in log_text
        assert query not in log_text
        assert content not in log_text
        assert str(config.DB_PATH) not in log_text
        assert "example.invalid" not in log_text

    def test_summary_uses_recent_window_and_reports_ratios(self):
        for i in range(120):
            performance.record_query_diagnostic(
                request_id=f"req-{i}",
                mode="cache" if i % 2 == 0 else (
                    "lexical_fallback" if i % 5 == 0 else "vector"
                ),
                latency_ms={
                    **performance.empty_latency(),
                    "total": float(i),
                    "embed": float(i) / 2,
                },
                result_count=1,
                cache_hit=i % 2 == 0,
                degraded=i % 5 == 0,
                degraded_reason="embedding_unavailable" if i % 5 == 0 else None,
            )

        summary = performance.summarize_query_performance(limit=100)

        assert summary["sample_size"] == 100
        assert summary["latency_ms"]["total"]["p50"] == pytest.approx(69.5)
        assert summary["latency_ms"]["total"]["p95"] == pytest.approx(114.05)
        assert summary["cache_hit_ratio"] == 0.5
        # 在最近 20..119 中，奇数且能被 5 整除的 lexical fallback 共 10 条。
        assert summary["fallback_ratio"] == 0.1
        assert summary["degraded_ratio"] == 0.2

    def test_status_performance_json(self):
        performance.record_query_diagnostic(
            request_id="req",
            mode="vector",
            latency_ms={**performance.empty_latency(), "total": 12.5},
            result_count=2,
        )

        result = CliRunner().invoke(cli, ["status", "--performance", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "ready"
        assert payload["performance"]["sample_size"] == 1
        assert payload["performance"]["latency_ms"]["total"]["p95"] == 12.5


@pytest.mark.parametrize(
    "delayed_stage", ["cache", "embed", "vector", "rerank", "graph", "serialize"]
)
def test_mocked_stage_latency_is_attributed_to_that_stage(
    delayed_stage, fake_embedder, monkeypatch
):
    _seed()
    fake_embedder.register("q", basis(0))
    durations = {
        stage: (0.025 if stage == delayed_stage else 0.0)
        for stage in ("cache", "embed", "vector", "rerank", "graph", "serialize")
    }
    current = 0.0
    values = [current]  # total start
    for stage in ("cache", "embed", "vector", "rerank", "graph", "serialize"):
        values.extend((current, current + durations[stage]))
        current += durations[stage]
    values.append(current)  # total end
    ticks = iter(values)
    monkeypatch.setattr(performance, "now", lambda: next(ticks))

    result = search.search("q", record_diagnostics=False)

    assert result["latency_ms"][delayed_stage] == 25.0
    assert result["latency_ms"]["total"] == 25.0
    other_stages = {
        stage for stage in ("cache", "embed", "vector", "rerank", "graph", "serialize")
        if stage != delayed_stage
    }
    assert all(result["latency_ms"][stage] == 0.0 for stage in other_stages)


class TestPerformanceBenchmark:
    def _register_fixed_vectors(self, fake_embedder):
        bench = perf_benchmark.benchmark_module.load_benchmark()
        for index, topic in enumerate(bench.topics):
            vector = basis(index)
            fake_embedder.register(topic.index_text(), vector)
            for query in topic.queries.values():
                fake_embedder.register(query, vector)
        return bench

    def test_small_warm_run_reports_stages_quality_and_no_raw_queries(
        self, fake_embedder
    ):
        bench = self._register_fixed_vectors(fake_embedder)

        report = perf_benchmark.run_performance_benchmark(
            story_count=30,
            query_count=6,
            repeats=2,
            concurrencies=(1, 2),
            model_state="warm",
        )

        assert report["dataset"]["stories"] == 30
        assert report["workload"]["requests_per_scenario"] == 12
        assert [row["sample_count"] for row in report["scenarios"]] == [12, 12]
        assert all(
            row["quality"]["overall"]["recall@3"] == 1.0
            for row in report["scenarios"]
        )
        assert set(report["scenarios"][0]["latency_ms"]) == set(
            performance.LATENCY_STAGES
        )
        serialized = json.dumps(report, ensure_ascii=False)
        for pair in chain.from_iterable(topic.queries.values() for topic in bench.topics):
            assert pair not in serialized
        assert "hostname" not in report["machine"]
        assert not config.PERFORMANCE_LOG_PATH.exists()

    def test_cold_run_unloads_once_per_concurrent_batch(self, fake_embedder):
        self._register_fixed_vectors(fake_embedder)
        calls = []

        report = perf_benchmark.run_performance_benchmark(
            story_count=24,
            query_count=2,
            repeats=2,
            concurrencies=(1, 2),
            model_state="cold",
            unload_model_fn=lambda: calls.append("unload"),
        )

        # 4 requests: concurrency=1 有 4 批，concurrency=2 有 2 批。
        assert len(calls) == 6
        assert report["model"]["state"] == "cold"

    def test_10k_reference_cache_and_warm_lanes_meet_gates(self, fake_embedder):
        """固定 mock embedding 隔离模型波动，守护索引/缓存自身的 10k 门槛。"""
        self._register_fixed_vectors(fake_embedder)

        report = perf_benchmark.run_performance_benchmark(
            story_count=10_000,
            query_count=6,
            repeats=2,
            concurrencies=(1,),
            model_state="warm",
        )

        scenario = report["scenarios"][0]
        assert scenario["latency_by_mode"]["cache"]["total"]["p95"] <= 80
        assert scenario["latency_by_mode"]["vector"]["total"]["p95"] <= 1_000
        assert scenario["quality"]["overall"]["recall@3"] >= 0.98

    def test_10k_fallback_lane_completes_within_500ms(self, fake_embedder):
        bench = self._register_fixed_vectors(fake_embedder)
        pairs = perf_benchmark._fixed_query_pairs(bench, 6)
        required_topics = {pair["topic_id"] for pair in pairs}

        with perf_benchmark._isolated_database():
            perf_benchmark._seed_dataset(
                bench, story_count=10_000, required_topics=required_topics
            )
            for pair in pairs:
                fake_embedder.register(pair["query"], None)

            durations = []
            for _ in range(2):
                for pair in pairs:
                    result = search.search(
                        pair["query"], top_k=5, record_diagnostics=False
                    )
                    assert result["mode"] == "lexical_fallback"
                    assert result["fallback_status"] == "ok"
                    durations.append(
                        result["latency_ms"]["fallback"]
                        + result["latency_ms"]["graph"]
                    )
            assert feedback.flush_feedback(timeout=5.0)

        assert performance.percentile(durations, 95) <= 500
