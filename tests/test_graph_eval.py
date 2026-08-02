"""Deterministic Memory Graph quality/performance benchmark self-tests."""
from storybook import graph_eval


def test_quality_benchmark_covers_edge_types_and_passes_recall_gate():
    report = graph_eval.run_quality_benchmark()

    assert set(report["dataset"]["edge_types"]) == {
        "semantic", "temporal", "causal", "same_environment",
        "parent_child", "co_recall", "supersedes",
    }
    assert report["dataset"]["includes_multi_hop"] is True
    assert (
        report["graph_rag"]["recall@5"]
        - report["vector_only"]["recall@5"]
    ) >= 0.10
    assert report["graph_rag"]["overall_recall@3"] >= (
        report["vector_only"]["overall_recall@3"] - 0.02
    )
    assert report["graph_rag"]["negative_false_positive_rate"] == 0.0


def test_performance_benchmark_reports_graph_percentiles():
    report = graph_eval.run_graph_performance_benchmark(
        story_count=100, repeats=5
    )

    assert report["stories"] == 100
    assert report["repeats"] == 5
    assert report["graph_ms"]["p50"] <= report["graph_ms"]["p95"]
    assert report["graph_ms"]["p95"] <= report["graph_ms"]["max"]
