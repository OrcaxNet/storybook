"""
检索激活模块 — 用户输入问题 → Top3 story + 关联推送
"""
import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from typing import TypeVar

from . import config, embeddings, feedback, performance, query_cache, store

logger = logging.getLogger(__name__)

T = TypeVar("T")


def search(
    query: str, top_k: int | None = None, *, record_diagnostics: bool = True
) -> dict:
    """
    检索记忆：
    1. 直接对 query 生成向量
    2. 向量检索 Top-K story
    3. 每个 story 下方按权重展示关联 story

    返回: {query, keywords, top_matches: [{story, similarity, related: [...]}]}
    """
    top_k = max(1, top_k or config.TOP_K_SEARCH)
    normalized_query = query.strip()
    request_id = uuid.uuid4().hex
    latency = performance.empty_latency()
    total_started = performance.now()
    model_state = embeddings.model_state()
    index_version = store.get_index_version()

    def finish(result: dict, *, mode: str, degraded: bool = False,
               degraded_reason: str | None = None) -> dict:
        result.update({
            "request_id": request_id,
            "mode": mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "index_version": index_version,
            "cache_hit": mode == "cache",
            "model_state": model_state,
            "latency_ms": latency,
        })

        serialize_started = performance.now()
        # MCP/CLI 最终编码发生在外层；这里做等价编码探针，把 payload 大小成本归入
        # 查询链路，同时不持久化序列化结果。
        json.dumps(result, ensure_ascii=False)
        latency["serialize"] = performance.elapsed_ms(serialize_started)
        latency["total"] = performance.elapsed_ms(total_started)
        result["latency_ms"] = dict(latency)

        if record_diagnostics:
            performance.record_query_diagnostic(
                request_id=request_id,
                mode=mode,
                latency_ms=latency,
                result_count=len(result.get("top_matches", [])),
                cache_hit=mode == "cache",
                degraded=degraded,
                degraded_reason=degraded_reason,
                model_state=model_state,
            )
        return result

    # ── Step 0: index_version 隔离的结果缓存 ──
    cache_started = performance.now()
    identity = query_cache.index_identity(index_version)
    cached = query_cache.get_result(identity, normalized_query, top_k)
    latency["cache"] = performance.elapsed_ms(cache_started)
    if cached is not None:
        cached["query"] = query
        cached["feedback_queued"] = feedback.enqueue_recall_feedback(
            [match["story_id"] for match in cached.get("top_matches", [])]
        )
        cached["query_vector_cache_hit"] = True
        return finish(cached, mode="cache")

    # ── Step 1: 生成查询向量 ──
    # 直接 embed 原始 query。早先用 llm.extract_keywords 抽关键词再拼 query 做 embedding，
    # 但搜索 query 通常很短，关键词提取器为凑 5-10 个配额会幻觉出无关词（如"内存泄漏""无限循环"），
    # 污染向量、把检索带偏。短 query 本身已是聚焦点，直接 embed 更准且省一次 LLM 调用。
    query_vec = query_cache.get_query_vector(identity, normalized_query)
    query_vector_cache_hit = query_vec is not None
    embed_failure_reason = None
    if query_vec is None:
        embed_started = performance.now()
        timeout_seconds = (
            config.QUERY_WARM_TIMEOUT_SECONDS
            if model_state == "warm"
            else config.QUERY_COLD_TIMEOUT_SECONDS
        )
        call_status, query_vec = _call_with_deadline(
            lambda: embeddings.embed(
                normalized_query,
                timeout_seconds=timeout_seconds,
                keep_alive=config.EMBED_KEEP_ALIVE,
            ),
            timeout_seconds,
        )
        latency["embed"] = performance.elapsed_ms(embed_started)
        if call_status == "timeout":
            embed_failure_reason = "embedding_timeout"
        elif call_status == "error" or not query_vec:
            embed_failure_reason = "embedding_unavailable"
        else:
            embeddings.mark_model_used()
            query_cache.set_query_vector(identity, normalized_query, query_vec)
    keywords = []   # 不再提取；保留字段以兼容返回结构

    if not query_vec:
        return _lexical_fallback(
            query=query,
            normalized_query=normalized_query,
            top_k=top_k,
            latency=latency,
            finish=finish,
            degraded_reason=embed_failure_reason or "embedding_unavailable",
            query_vector_cache_hit=query_vector_cache_hit,
        )

    # ── Step 2: 向量检索 ──
    vector_started = performance.now()
    try:
        matches = store.search_by_vector(query_vec, top_k=top_k * 2)  # 多取一些再过滤
    except Exception:  # noqa: BLE001 -- vec0 损坏/缺失时必须快速降级
        latency["vector"] = performance.elapsed_ms(vector_started)
        return _lexical_fallback(
            query=query,
            normalized_query=normalized_query,
            top_k=top_k,
            latency=latency,
            finish=finish,
            degraded_reason="vector_index_unavailable",
            query_vector_cache_hit=query_vector_cache_hit,
        )
    latency["vector"] = performance.elapsed_ms(vector_started)

    # 过滤低于阈值的
    rerank_started = performance.now()
    matches = [m for m in matches if m["similarity"] >= config.SIM_THRESHOLD_SEARCH]
    matches = matches[:top_k]
    latency["rerank"] = performance.elapsed_ms(rerank_started)

    if not matches:
        result = {
            "query": query,
            "keywords": keywords,
            "top_matches": [],
            "result_state": "no_match",
            "query_vector_cache_hit": query_vector_cache_hit,
            "feedback_queued": True,
        }
        query_cache.set_result(
            identity, normalized_query, top_k, _cacheable_result(result)
        )
        return finish(result, mode="vector")

    # ── Step 3: 关联激活 ──
    graph_started = performance.now()
    top_matches = _attach_related(matches, retrieval_source="vector")
    latency["graph"] = performance.elapsed_ms(graph_started)

    feedback_queued = feedback.enqueue_recall_feedback(
        [match["story_id"] for match in matches]
    )

    result = {
        "query": query,
        "keywords": keywords,
        "top_matches": top_matches,
        "result_state": "results",
        "query_vector_cache_hit": query_vector_cache_hit,
        "feedback_queued": feedback_queued,
    }
    query_cache.set_result(
        identity, normalized_query, top_k, _cacheable_result(result)
    )

    return finish(result, mode="vector")


def _lexical_fallback(
    *,
    query: str,
    normalized_query: str,
    top_k: int,
    latency: dict[str, float],
    finish: Callable[..., dict],
    degraded_reason: str,
    query_vector_cache_hit: bool,
) -> dict:
    fallback_started = performance.now()
    call_status, fallback_output = _call_with_deadline(
        lambda: _run_lexical_fallback(normalized_query, top_k),
        config.QUERY_FALLBACK_TIMEOUT_SECONDS,
    )

    if call_status == "ok":
        matches, top_matches, fallback_ms, graph_ms = fallback_output
        latency["fallback"] = fallback_ms
        latency["graph"] = graph_ms
        feedback_queued = feedback.enqueue_recall_feedback(
            [match["story_id"] for match in matches or []]
        )
        result_state = "degraded_results" if top_matches else "degraded_empty"
        fallback_status = "ok"
    else:
        latency["fallback"] = performance.elapsed_ms(fallback_started)
        top_matches = []
        feedback_queued = True
        result_state = "degraded_unavailable"
        fallback_status = call_status

    return finish(
        {
            "query": query,
            "keywords": [],
            "top_matches": top_matches,
            "result_state": result_state,
            "fallback_status": fallback_status,
            "query_vector_cache_hit": query_vector_cache_hit,
            "feedback_queued": feedback_queued,
        },
        mode="lexical_fallback",
        degraded=True,
        degraded_reason=degraded_reason,
    )


def _run_lexical_fallback(
    normalized_query: str, top_k: int
) -> tuple[list[dict], list[dict], float, float]:
    """在同一个 500ms deadline 内完成词法检索与关联读取。"""

    fallback_started = time.perf_counter()
    matches = store.search_by_lexical(
        normalized_query,
        top_k=top_k,
        timeout_seconds=config.QUERY_FALLBACK_TIMEOUT_SECONDS,
    )
    fallback_ms = round((time.perf_counter() - fallback_started) * 1000, 3)
    graph_started = time.perf_counter()
    top_matches = _attach_related(matches, retrieval_source="lexical")
    graph_ms = round((time.perf_counter() - graph_started) * 1000, 3)
    return matches, top_matches, fallback_ms, graph_ms


def _attach_related(matches: list[dict], *, retrieval_source: str) -> list[dict]:
    related_by_story = store.get_related_stories_batch(
        [match["story_id"] for match in matches], limit=5
    )
    top_matches = []
    for match in matches:
        related = related_by_story.get(match["story_id"], [])
        top_matches.append({
            "story_id": match["story_id"],
            "title": match["title"],
            "content": match["content"],
            "keywords": match["keywords"],
            "similarity": match["similarity"],
            "retrieval_source": retrieval_source,
            "related": [
                {
                    "story_id": item["id"],
                    "title": item["title"],
                    "content": item["content"],
                    "weight": item.get("weight", 0),
                    "edge_type": item.get("edge_type", "semantic"),
                }
                for item in related
            ],
        })
    return top_matches


def _cacheable_result(result: dict) -> dict:
    return {
        key: value for key, value in result.items()
        if key not in {"feedback_queued", "query_vector_cache_hit"}
    }


def _call_with_deadline(
    callback: Callable[[], T], timeout_seconds: float
) -> tuple[str, T | None]:
    """在 daemon 线程中执行不可控依赖，并在 deadline 到达时立即返回。"""

    output: queue.Queue[tuple[str, T | None]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            output.put_nowait(("ok", callback()))
        except Exception:  # noqa: BLE001 -- 错误正文可能含路径，不写查询日志
            output.put_nowait(("error", None))

    worker = threading.Thread(target=run, name="storybook-query-deadline", daemon=True)
    worker.start()
    worker.join(max(0.001, timeout_seconds))
    if worker.is_alive():
        return "timeout", None
    try:
        return output.get_nowait()
    except queue.Empty:
        return "error", None


def format_search_result(result: dict) -> str:
    """格式化搜索结果为可读文本"""
    lines = []
    lines.append(f"🔍 搜索: {result['query']}")
    if result.get("latency_ms"):
        lines.append(
            f"   {result.get('mode', 'unknown')} · "
            f"{result['latency_ms'].get('total', 0):.1f}ms"
        )

    if result.get("keywords"):
        lines.append(f"   关键词: {', '.join(result['keywords'])}")

    if result.get("degraded") and result.get("top_matches"):
        lines.append(
            f"   ⚠️ 向量路径已降级（{result.get('degraded_reason', 'unknown')}），"
            "以下为关键词 fallback 结果"
        )

    if not result.get("top_matches"):
        if result.get("degraded"):
            lines.append(
                "\n   ⚠️ 向量检索不可用，关键词降级未找到匹配；"
                "这不等同于确认没有相关记忆"
            )
        else:
            lines.append("\n   ❌ 未找到匹配的记忆")
        return "\n".join(lines)

    lines.append(f"\n   找到 {len(result['top_matches'])} 条匹配记忆:\n")

    for i, m in enumerate(result["top_matches"], 1):
        sim_bar = "█" * int(m["similarity"] * 10) + "░" * (10 - int(m["similarity"] * 10))
        lines.append(f"   ┌─ Top {i} [{sim_bar} {m['similarity']:.1%}] ─────────────────")
        lines.append(f"   │ 📌 {m['title']}")
        lines.append(f"   │ {m['content'][:200]}")
        lines.append(f"   │ 关键词: {', '.join(m['keywords'])}")

        if m.get("related"):
            lines.append("   │")
            lines.append("   │ 💭 联想到的相关记忆:")
            for j, r in enumerate(m["related"], 1):
                weight_bar = "█" * int(r["weight"] * 10) + "░" * (10 - int(r["weight"] * 10))
                tag = "🔗" if r["edge_type"] == "parent_child" else "💭"
                lines.append(f"   │   {tag} [{weight_bar} {r['weight']:.2f}] {r['title']}")
                lines.append(f"   │      {r['content'][:100]}")

        lines.append(f"   └{'─' * 50}")
        lines.append("")

    return "\n".join(lines)
