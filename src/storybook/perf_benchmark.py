"""可重复的 Storybook 查询性能基准。

默认口径：10k Story、固定 50 条查询、每条重复 20 次、并发 1/5。
benchmark 使用隔离临时库，不污染用户数据；报告只含聚合指标和无内容元数据。
"""
from __future__ import annotations

import contextlib
import os
import platform
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

from . import config, embeddings, performance, store
from . import search as search_module
from .eval import benchmark as benchmark_module
from .eval import metrics

DEFAULT_STORY_COUNT = 10_000
DEFAULT_QUERY_COUNT = 50
DEFAULT_REPEATS = 20
DEFAULT_CONCURRENCIES = (1, 5)
DATASET_SEED = 20260729
_VARIANTS = ("exact", "synonym", "cross_lang")


def run_performance_benchmark(
    *,
    story_count: int = DEFAULT_STORY_COUNT,
    query_count: int = DEFAULT_QUERY_COUNT,
    repeats: int = DEFAULT_REPEATS,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
    model_state: str = "warm",
    benchmark_path: Path | str | None = None,
    unload_model_fn: Callable[[], None] | None = None,
) -> dict:
    """运行固定性能 workload 并返回可 JSON 序列化的聚合报告。"""
    if story_count < 1:
        raise ValueError("story_count 必须大于 0")
    if query_count < 1 or repeats < 1:
        raise ValueError("query_count 和 repeats 必须大于 0")
    if model_state not in {"warm", "cold"}:
        raise ValueError("model_state 只能是 warm 或 cold")
    if not concurrencies or any(c < 1 for c in concurrencies):
        raise ValueError("concurrencies 必须是正整数")

    bench = benchmark_module.load_benchmark(benchmark_path)
    pairs = _fixed_query_pairs(bench, query_count)
    required_topics = {pair["topic_id"] for pair in pairs}
    if story_count < len(required_topics):
        raise ValueError(
            f"story_count 至少为固定查询覆盖的 topic 数量 {len(required_topics)}"
        )

    started = time.perf_counter()
    with _isolated_database() as db_path:
        topic_to_story = _seed_dataset(
            bench, story_count=story_count, required_topics=required_topics
        )
        scenarios = []
        for concurrency in concurrencies:
            scenarios.append(
                _run_scenario(
                    pairs=pairs,
                    topic_to_story=topic_to_story,
                    repeats=repeats,
                    concurrency=concurrency,
                    model_state=model_state,
                    unload_model_fn=unload_model_fn,
                )
            )
        database_bytes = db_path.stat().st_size if db_path.exists() else 0

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "machine": machine_metadata(),
        "model": {
            "embedding": config.EMBED_MODEL,
            "dimension": config.EMBED_DIM,
            "state": model_state,
        },
        "dataset": {
            "stories": story_count,
            "seed": DATASET_SEED,
            "query_count": len(pairs),
            "variants": list(_VARIANTS),
            "database_bytes": database_bytes,
        },
        "workload": {
            "repeats_per_query": repeats,
            "concurrencies": list(concurrencies),
            "requests_per_scenario": len(pairs) * repeats,
        },
        "scenarios": scenarios,
        "setup_and_run_ms": round((time.perf_counter() - started) * 1000, 3),
        "privacy": {
            "raw_queries_recorded": False,
            "story_content_recorded": False,
            "absolute_paths_recorded": False,
            "repository_urls_recorded": False,
            "hostname_recorded": False,
        },
    }


def format_benchmark_report(report: dict) -> str:
    """生成适合终端查看的基准摘要。"""
    dataset = report["dataset"]
    model = report["model"]
    machine = report["machine"]
    workload = report["workload"]
    lines = [
        "📈 Storybook 查询性能基准",
        (
            f"dataset: {dataset['stories']} stories | embedding: "
            f"{model['embedding']} | dim: {model['dimension']}"
        ),
        (
            f"machine: {machine['os']}/{machine['arch']} | cpu: "
            f"{machine['cpu_count']} cores | memory: {machine['memory_bytes']} bytes"
        ),
        (
            f"model_state: {model['state']} | runs: {dataset['query_count']} queries "
            f"× {workload['repeats_per_query']} repeats"
        ),
    ]
    for scenario in report["scenarios"]:
        total = scenario["latency_ms"]["total"]
        lines.append(
            f"concurrency={scenario['concurrency']}: "
            f"total p50/p95/p99={total['p50']:.3f}/{total['p95']:.3f}/"
            f"{total['p99']:.3f}ms | cache={scenario['cache_hit_ratio']:.1%} | "
            f"fallback={scenario['fallback_ratio']:.1%}"
        )
        stage_bits = []
        for stage in performance.LATENCY_STAGES[:-1]:
            stage_bits.append(
                f"{stage}={scenario['latency_ms'][stage]['p95']:.3f}"
            )
        lines.append("  stage p95(ms): " + " · ".join(stage_bits))
        quality = scenario["quality"]["overall"]
        lines.append(
            f"  quality: recall@1/3/5={quality['recall@1']:.1%}/"
            f"{quality['recall@3']:.1%}/{quality['recall@5']:.1%} · "
            f"MRR={quality['mrr']:.4f}"
        )
    return "\n".join(lines)


def machine_metadata() -> dict:
    """采集可复现所需机器信息，刻意排除 hostname、用户名和路径。"""
    cpu = platform.processor() or "unknown"
    memory_bytes = 0
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "os": platform.system() or "unknown",
        "os_release": platform.release() or "unknown",
        "arch": platform.machine() or "unknown",
        "cpu": cpu,
        "cpu_count": os.cpu_count() or 0,
        "memory_bytes": memory_bytes,
    }


def unload_embedding_model() -> None:
    """通过 Ollama 官方 keep_alive=0 语义卸载 embedding 模型。"""
    response = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json={"model": config.EMBED_MODEL, "keep_alive": 0, "stream": False},
        timeout=15,
    )
    response.raise_for_status()


@contextlib.contextmanager
def _isolated_database():
    saved = config.DB_PATH
    with tempfile.TemporaryDirectory(prefix="storybook-perf-") as tmp_dir:
        config.DB_PATH = Path(tmp_dir) / "performance.db"
        try:
            store.init_db()
            yield config.DB_PATH
        finally:
            config.DB_PATH = saved


def _fixed_query_pairs(bench, query_count: int) -> list[dict]:
    pairs = []
    for topic in bench.topics:
        for variant in _VARIANTS:
            query = topic.queries.get(variant)
            if query:
                pairs.append({
                    "topic_id": topic.id,
                    "variant": variant,
                    "query": query,
                })
    if query_count > len(pairs):
        raise ValueError(
            f"固定查询集只有 {len(pairs)} 条，不能请求 {query_count} 条"
        )
    return pairs[:query_count]


def _seed_dataset(bench, *, story_count: int, required_topics: set[str]) -> dict[str, int]:
    topic_to_story: dict[str, int] = {}
    topics = [topic for topic in bench.topics if topic.id in required_topics]
    for topic in topics:
        vector = embeddings.embed(topic.index_text())
        if not vector:
            raise RuntimeError(f"benchmark topic embedding 失败: {topic.id}")
        topic_to_story[topic.id] = store.add_story(
            topic.title, topic.content, topic.keywords, vector, source_session_ids=[]
        )

    remaining = story_count - len(topics)
    if remaining <= 0:
        return topic_to_story

    rng = np.random.default_rng(DATASET_SEED)
    db = store.get_db()
    try:
        next_id = max(topic_to_story.values(), default=0) + 1
        inserted = 0
        while inserted < remaining:
            batch_size = min(256, remaining - inserted)
            vectors = rng.standard_normal(
                (batch_size, config.EMBED_DIM), dtype=np.float32
            )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.where(norms == 0, 1.0, norms)
            for offset, vector in enumerate(vectors):
                story_id = next_id + inserted + offset
                blob = vector.astype(np.float32, copy=False).tobytes()
                db.execute(
                    """INSERT INTO stories
                       (id, title, content, keywords, embedding, source_session_ids)
                       VALUES (?, ?, ?, '[]', ?, '[]')""",
                    (
                        story_id,
                        f"Synthetic benchmark story {story_id:05d}",
                        "Deterministic synthetic benchmark content.",
                        blob,
                    ),
                )
                db.execute(
                    "INSERT INTO story_vectors (story_id, embedding) VALUES (?, ?)",
                    (story_id, blob),
                )
            inserted += batch_size
        db.commit()
    finally:
        db.close()
    return topic_to_story


def _run_scenario(
    *,
    pairs: list[dict],
    topic_to_story: dict[str, int],
    repeats: int,
    concurrency: int,
    model_state: str,
    unload_model_fn: Callable[[], None] | None,
) -> dict:
    tasks = [pair for _ in range(repeats) for pair in pairs]
    if (
        model_state == "warm"
        and not embeddings.embed("storybook performance benchmark warmup")
    ):
        raise RuntimeError("embedding 预热失败")

    unload = unload_model_fn or unload_embedding_model
    outputs = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for start in range(0, len(tasks), concurrency):
            batch = tasks[start:start + concurrency]
            if model_state == "cold":
                unload()
            futures = [
                executor.submit(
                    _query_once, pair, topic_to_story[pair["topic_id"]]
                )
                for pair in batch
            ]
            outputs.extend(future.result() for future in futures)

    latency = {}
    for stage in performance.LATENCY_STAGES:
        values = [row["latency_ms"][stage] for row in outputs]
        latency[stage] = {
            "p50": performance.percentile(values, 50),
            "p95": performance.percentile(values, 95),
            "p99": performance.percentile(values, 99),
        }

    fallback_rows = [row for row in outputs if row["mode"] == "lexical_fallback"]
    return {
        "concurrency": concurrency,
        "sample_count": len(outputs),
        "latency_ms": latency,
        "cache_hit_ratio": _ratio(
            sum(row["mode"] == "cache" for row in outputs), len(outputs)
        ),
        "fallback_ratio": _ratio(len(fallback_rows), len(outputs)),
        "degraded_ratio": _ratio(
            sum(row["degraded"] for row in outputs), len(outputs)
        ),
        "fallback": {
            "attempts": len(fallback_rows),
            "success_rate": (
                _ratio(sum(row["target_hit"] for row in fallback_rows), len(fallback_rows))
                if fallback_rows else None
            ),
            "false_empty_rate": (
                _ratio(sum(not row["match_ids"] for row in fallback_rows), len(fallback_rows))
                if fallback_rows else None
            ),
        },
        "quality": _quality(outputs),
    }


def _query_once(pair: dict, target_story_id: int) -> dict:
    result = search_module.search(
        pair["query"], top_k=5, record_diagnostics=False
    )
    match_ids = [match["story_id"] for match in result.get("top_matches", [])]
    return {
        "variant": pair["variant"],
        "target_story_id": target_story_id,
        "target_hit": target_story_id in match_ids,
        "match_ids": match_ids,
        "mode": result.get("mode", "error"),
        "degraded": bool(result.get("degraded")),
        "latency_ms": result["latency_ms"],
    }


def _quality(rows: list[dict]) -> dict:
    def aggregate(items: list[dict]) -> dict:
        if not items:
            return {
                "count": 0,
                "recall@1": 0.0,
                "recall@3": 0.0,
                "recall@5": 0.0,
                "mrr": 0.0,
            }
        result = {"count": len(items)}
        for k in (1, 3, 5):
            result[f"recall@{k}"] = round(
                sum(
                    metrics.recall_at_k(
                        row["match_ids"], [row["target_story_id"]], k
                    )
                    for row in items
                ) / len(items),
                4,
            )
        result["mrr"] = round(
            sum(
                metrics.mrr(row["match_ids"], [row["target_story_id"]])
                for row in items
            ) / len(items),
            4,
        )
        return result

    return {
        "overall": aggregate(rows),
        "by_variant": {
            variant: aggregate([row for row in rows if row["variant"] == variant])
            for variant in _VARIANTS
        },
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
