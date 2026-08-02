"""Reproducible artificial quality and 10k-Story Graph RAG benchmark."""
from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from . import config, feedback, graph, performance, query_cache, store
from .perf_benchmark import machine_metadata


def run_memory_graph_benchmark(
    *, story_count: int = 10_000, repeats: int = 50
) -> dict:
    if story_count < 10:
        raise ValueError("story_count 必须至少为 10")
    if repeats < 1:
        raise ValueError("repeats 必须大于 0")
    quality = run_quality_benchmark()
    performance_report = run_graph_performance_benchmark(
        story_count=story_count, repeats=repeats
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "machine": machine_metadata(),
        "quality": quality,
        "performance": performance_report,
        "gates": {
            "associated_recall_at_5_lift_pp": round(
                (
                    quality["graph_rag"]["recall@5"]
                    - quality["vector_only"]["recall@5"]
                ) * 100,
                3,
            ),
            "overall_recall_at_3_regression_pp": round(
                (
                    quality["vector_only"]["overall_recall@3"]
                    - quality["graph_rag"]["overall_recall@3"]
                ) * 100,
                3,
            ),
            "graph_ms_p95": performance_report["graph_ms"]["p95"],
            "quality_passed": (
                quality["graph_rag"]["recall@5"]
                - quality["vector_only"]["recall@5"] >= 0.10
                and quality["graph_rag"]["overall_recall@3"]
                >= quality["vector_only"]["overall_recall@3"] - 0.02
            ),
            "latency_passed": performance_report["graph_ms"]["p95"] <= 120.0,
        },
        "privacy": {
            "raw_queries_recorded": False,
            "story_content_recorded": False,
            "absolute_paths_recorded": False,
            "hostname_recorded": False,
        },
    }


def run_quality_benchmark() -> dict:
    """Exercise every standard edge type, multi-hop and a negative case."""

    cases = (
        ("semantic", ("semantic",)),
        ("temporal", ("temporal",)),
        ("causal", ("causal",)),
        ("same_environment", ("same_environment",)),
        ("parent_child", ("parent_child",)),
        ("co_recall", ("co_recall",)),
        ("supersedes", ("supersedes",)),
        ("causal_multi_hop", ("causal", "causal")),
    )
    vector_hits = graph_hits = 0
    vector_rr = graph_rr = 0.0
    vector_overall = graph_overall = 0
    details = []
    with _isolated_database():
        for case_name, path_types in cases:
            seed = _add_eval_story(f"{case_name}-seed")
            decoys = [_add_eval_story(f"{case_name}-decoy-{i}") for i in range(4)]
            direct = [
                _direct(seed, 0.92),
                _direct(decoys[0], 0.70),
                _direct(decoys[1], 0.68),
                _direct(decoys[2], 0.66),
                _direct(decoys[3], 0.64),
            ]
            current = seed
            for hop, edge_type in enumerate(path_types):
                target = _add_eval_story(f"{case_name}-relevant-{hop}")
                if edge_type == "supersedes":
                    # Storage contract is replacement(new) -> old.
                    store.add_or_update_edge(
                        target, current, 1.0, edge_type,
                        provenance={"source": "benchmark", "case": case_name},
                    )
                else:
                    store.add_or_update_edge(
                        current, target, 1.0, edge_type,
                        provenance={"source": "benchmark", "case": case_name},
                    )
                current = target
            relevant = current
            expansion = graph.expand(direct)
            graph_ranked = _merge_for_eval(direct, expansion)
            vector_ids = [item["story_id"] for item in direct[:5]]
            graph_ids = [item["story_id"] for item in graph_ranked[:5]]
            vector_rank = _rank(vector_ids, relevant)
            graph_rank = _rank(graph_ids, relevant)
            vector_hits += int(vector_rank is not None)
            graph_hits += int(graph_rank is not None)
            vector_rr += 0.0 if vector_rank is None else 1.0 / vector_rank
            graph_rr += 0.0 if graph_rank is None else 1.0 / graph_rank
            # Superseded old Stories are intentionally not an overall target.
            if case_name != "supersedes":
                vector_overall += int(seed in vector_ids[:3])
                graph_overall += int(seed in graph_ids[:3])
            details.append({
                "case": case_name,
                "edge_types": list(path_types),
                "vector_hit@5": vector_rank is not None,
                "graph_hit@5": graph_rank is not None,
                "graph_rank": graph_rank,
                "path_length": len(path_types),
            })

        negative_seed = _add_eval_story("negative-seed")
        negative = graph.expand([_direct(negative_seed, 0.92)])

    count = len(cases)
    overall_count = count - 1
    return {
        "dataset": {
            "associated_cases": count,
            "negative_cases": 1,
            "edge_types": list(config.MEMORY_EDGE_TYPES),
            "includes_one_hop": True,
            "includes_multi_hop": True,
        },
        "vector_only": {
            "recall@5": round(vector_hits / count, 4),
            "mrr": round(vector_rr / count, 4),
            "overall_recall@3": round(vector_overall / overall_count, 4),
        },
        "graph_rag": {
            "recall@5": round(graph_hits / count, 4),
            "mrr": round(graph_rr / count, 4),
            "overall_recall@3": round(graph_overall / overall_count, 4),
            "negative_false_positive_rate": round(
                int(bool(negative["matches"])), 4
            ),
        },
        "cases": details,
    }


def run_graph_performance_benchmark(
    *, story_count: int = 10_000, repeats: int = 50
) -> dict:
    durations = []
    with _isolated_database() as db_path:
        _seed_performance_graph(story_count)
        seed = [_direct(1, 0.95)]
        graph.expand(seed)  # warm schema/page cache
        for _ in range(repeats):
            started = time.perf_counter()
            graph.expand(seed)
            durations.append((time.perf_counter() - started) * 1000.0)
        database_bytes = db_path.stat().st_size if db_path.exists() else 0
    return {
        "stories": story_count,
        "repeats": repeats,
        "active_edges": story_count - 1 + min(64, story_count - 2),
        "database_bytes": database_bytes,
        "graph_ms": {
            "p50": performance.percentile(durations, 50),
            "p95": performance.percentile(durations, 95),
            "p99": performance.percentile(durations, 99),
            "max": round(max(durations), 3),
        },
        "budgets": {
            "max_hops": config.GRAPH_MAX_HOPS,
            "max_paths": config.GRAPH_MAX_PATHS,
            "fan_out": config.GRAPH_FAN_OUT,
            "time_ms": config.GRAPH_TIME_BUDGET_MS,
            "tokens": config.GRAPH_TOKEN_BUDGET,
        },
    }


@contextlib.contextmanager
def _isolated_database():
    saved = config.DB_PATH
    feedback.flush_feedback(timeout=5.0)
    query_cache.clear()
    with tempfile.TemporaryDirectory(prefix="storybook-graph-eval-") as tmp_dir:
        config.DB_PATH = Path(tmp_dir) / "memory.db"
        try:
            store.init_db()
            yield config.DB_PATH
        finally:
            feedback.flush_feedback(timeout=5.0)
            query_cache.clear()
            config.DB_PATH = saved


def _add_eval_story(title: str) -> int:
    # Graph quality starts from an already-produced direct lane, so vectors are
    # irrelevant here; a deterministic unit vector keeps the Story active.
    vector = [0.0] * config.EMBED_DIM
    vector[0] = 1.0
    return store.add_story(title, f"{title} summary", [], vector)


def _direct(story_id: int, score: float) -> dict:
    return {"story_id": int(story_id), "similarity": score, "score": score}


def _merge_for_eval(direct: list[dict], expansion: dict) -> list[dict]:
    suppressed = set(expansion["suppressed_story_ids"])
    best = {
        item["story_id"]: dict(item)
        for item in direct if item["story_id"] not in suppressed
    }
    for item in expansion["matches"]:
        existing = best.get(item["story_id"])
        if existing is None or item["score"] > existing["score"]:
            best[item["story_id"]] = item
    return sorted(best.values(), key=lambda item: item["score"], reverse=True)


def _rank(ids: list[int], expected: int) -> int | None:
    try:
        return ids.index(expected) + 1
    except ValueError:
        return None


def _seed_performance_graph(story_count: int) -> None:
    db = store.get_db(load_vector_extension=False)
    try:
        db.executemany(
            """INSERT INTO stories (
                   id, global_id, profile_id, sync_state, title, abstract,
                   content, keywords, source_session_ids
               ) VALUES (?, ?, ?, 'local_only', ?, ?, ?, '[]', '[]')""",
            (
                (
                    story_id,
                    str(uuid.uuid4()),
                    config.PROFILE_ID,
                    f"story-{story_id}",
                    f"summary-{story_id}",
                    f"detail-{story_id}",
                )
                for story_id in range(1, story_count + 1)
            ),
        )
        edge_rows = []
        provenance = json.dumps({"source": "benchmark"}, sort_keys=True)
        for story_id in range(1, story_count):
            edge_rows.append((
                str(uuid.uuid4()), config.PROFILE_ID, story_id, story_id + 1,
                0.95, "causal", 1, provenance,
            ))
        # A bounded high-degree hub validates that fan-out is independent of
        # total graph size and that hub degree lookup remains indexed.
        for target in range(3, min(story_count + 1, 67)):
            edge_rows.append((
                str(uuid.uuid4()), config.PROFILE_ID, 1, target,
                0.8, "semantic", 0, provenance,
            ))
        db.executemany(
            """INSERT INTO edges (
                   global_id, profile_id, sync_state, source_id, target_id,
                   weight, edge_type, directed, provenance_json
               ) VALUES (?, ?, 'local_only', ?, ?, ?, ?, ?, ?)""",
            edge_rows,
        )
        db.commit()
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Storybook Memory Graph benchmark")
    parser.add_argument("--stories", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_memory_graph_benchmark(
        story_count=args.stories, repeats=args.repeats
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
