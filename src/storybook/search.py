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

from . import adaptive, config, embeddings, feedback, graph as graph_module, performance
from . import llm
from . import query_cache, store
from . import context as context_module

logger = logging.getLogger(__name__)

T = TypeVar("T")


def search(
    query: str,
    top_k: int | None = None,
    *,
    context: dict | None = None,
    scope: str = "profile",
    graph_enabled: bool | None = None,
    retrieval_mode: str | None = None,
    transform_enabled: bool | None = None,
    rerank_enabled: bool | None = None,
    record_diagnostics: bool = True,
) -> dict:
    """Retrieve memories through fast, auto, or explicitly deep recall.

    ``mode`` in the response retains the v0.2 execution-lane contract
    (cache/vector/lexical_fallback).  ``retrieval_mode`` exposes the new public
    strategy without breaking older MCP/CLI consumers.
    """
    top_k = max(1, top_k or config.TOP_K_SEARCH)
    retrieval_mode = adaptive.resolve_mode(retrieval_mode)
    graph_enabled = (
        config.GRAPH_DEFAULT_ENABLED if graph_enabled is None
        else bool(graph_enabled)
    )
    if scope not in ("profile", "strict"):
        raise ValueError("scope 必须是 profile 或 strict")
    current_context = (
        context_module.normalize_envelope(context, profile_id=config.PROFILE_ID)
        if context is not None else None
    )
    normalized_query = adaptive.normalize_query(query)
    if not normalized_query:
        raise ValueError("query 不能为空")
    request_id = uuid.uuid4().hex
    latency = performance.empty_latency()
    total_started = performance.now()
    model_state = embeddings.model_state()
    index_version = store.get_index_version()

    def finish(
        result: dict,
        *,
        mode: str,
        degraded: bool = False,
        degraded_reason: str | None = None,
    ) -> dict:
        degraded_reasons = list(dict.fromkeys(
            reason for reason in result.get("degraded_reasons", []) if reason
        ))
        if degraded_reason and degraded_reason not in degraded_reasons:
            degraded_reasons.insert(0, degraded_reason)
        primary_degraded_reason = degraded_reasons[0] if degraded_reasons else None
        result.update({
            "request_id": request_id,
            "mode": mode,
            "retrieval_mode": retrieval_mode,
            "degraded": bool(degraded or degraded_reasons),
            "degraded_reason": primary_degraded_reason,
            "degraded_reasons": degraded_reasons,
            "index_version": index_version,
            "cache_hit": mode == "cache",
            "model_state": model_state,
            "latency_ms": latency,
            "scope": scope,
            "context": current_context,
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
                degraded=bool(degraded or degraded_reasons),
                degraded_reason=primary_degraded_reason,
                model_state=model_state,
            )
        return result

    # ── Step 0: index_version 隔离的结果缓存 ──
    cache_started = performance.now()
    identity = query_cache.index_identity(index_version)
    # Environment-aware results depend on the supplied envelope and scope. Keep
    # the existing fast result cache for context-free profile searches only;
    # the query-vector cache remains safe and shared for every variant.
    result_cache_enabled = current_context is None and scope == "profile"
    result_cache_query = (
        f"{normalized_query}\x1fgraph={int(graph_enabled)}"
        f"\x1fmode={retrieval_mode}"
        f"\x1ftransform={int(config.QUERY_TRANSFORM_ENABLED if transform_enabled is None else bool(transform_enabled))}"
        f"\x1frerank={int(config.RERANK_ENABLED if rerank_enabled is None else bool(rerank_enabled))}"
    )
    cached = (
        query_cache.get_result(identity, result_cache_query, top_k)
        if result_cache_enabled else None
    )
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
            current_context=current_context,
            scope=scope,
            graph_enabled=graph_enabled,
            retrieval_mode=retrieval_mode,
            transform_enabled=transform_enabled,
            rerank_enabled=rerank_enabled,
        )

    # ── Step 2: 向量检索 ──
    vector_started = performance.now()
    try:
        matches = store.search_by_vector(
            query_vec, top_k=max(top_k * 4, top_k)
        )
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
            current_context=current_context,
            scope=scope,
            graph_enabled=graph_enabled,
            retrieval_mode=retrieval_mode,
            transform_enabled=transform_enabled,
            rerank_enabled=rerank_enabled,
        )
    latency["vector"] = performance.elapsed_ms(vector_started)

    # ── Step 3: 常态 lexical lane + RRF fusion ──
    candidate_limit = max(top_k * 4, top_k, min(64, config.RERANK_TOP_N))
    vector_matches = [
        match for match in matches
        if match["similarity"] >= config.SIM_THRESHOLD_SEARCH
    ]
    lexical_started = performance.now()
    lexical_status, lexical_matches = _call_with_deadline(
        lambda: store.search_by_lexical(
            normalized_query,
            top_k=candidate_limit,
            timeout_seconds=config.QUERY_FALLBACK_TIMEOUT_SECONDS,
        ),
        config.QUERY_FALLBACK_TIMEOUT_SECONDS,
    )
    latency["lexical"] = performance.elapsed_ms(lexical_started)
    degraded_reasons: list[str] = []
    if lexical_status != "ok" or lexical_matches is None:
        lexical_matches = []
        degraded_reasons.append("lexical_index_unavailable")

    fusion_started = performance.now()
    matches = adaptive.fuse_rankings(
        vector_matches,
        lexical_matches,
        limit=candidate_limit,
    )
    matches, strict_filtered = _environment_rerank(
        matches,
        current_context=current_context,
        scope=scope,
        top_k=candidate_limit,
    )
    latency["fusion"] = performance.elapsed_ms(fusion_started)

    # ── Step 4: explainable auto/deep gate + independently budgeted stage ──
    query_plan = adaptive.plan_query(
        normalized_query,
        matches,
        requested_mode=retrieval_mode,
        current_context=current_context,
        strict_filtered=strict_filtered,
        transform_enabled=transform_enabled,
    )
    transform_used: list[str] = []
    transform_trace = {
        "status": "skipped",
        "generated_queries": 0,
        "retrievals_completed": 0,
        "degraded_reasons": [],
    }
    if query_plan["should_transform"]:
        second_stage_timeout = (
            config.QUERY_DEEP_SECOND_STAGE_TIMEOUT_SECONDS
            if retrieval_mode == "deep"
            else config.QUERY_AUTO_SECOND_STAGE_TIMEOUT_SECONDS
        )
        if retrieval_mode == "deep":
            elapsed_seconds = max(0.0, performance.now() - total_started)
            reserved_seconds = (
                config.GRAPH_DEEP_TIME_BUDGET_MS / 1000.0
                + config.RERANK_TIMEOUT_SECONDS
                + 0.05
            )
            second_stage_timeout = min(
                second_stage_timeout,
                max(
                    0.001,
                    config.QUERY_DEEP_TOTAL_TIMEOUT_SECONDS
                    - elapsed_seconds
                    - reserved_seconds,
                ),
            )
        transform_started = performance.now()
        transform_status, transformed = _call_with_deadline(
            lambda: _run_second_stage(
                normalized_query,
                selected_transforms=query_plan["selected_transforms"],
                retrieval_mode=retrieval_mode,
                identity=identity,
                candidate_limit=candidate_limit,
            ),
            second_stage_timeout,
        )
        latency["transform"] = performance.elapsed_ms(transform_started)
        if transform_status == "ok" and transformed is not None:
            transform_trace = transformed["trace"]
            transform_used = transformed["transform_used"]
            degraded_reasons.extend(transformed["degraded_reasons"])
            matches = adaptive.merge_transformed_rankings(
                matches,
                transformed["rankings"],
                mode=retrieval_mode,
                limit=candidate_limit,
            )
            matches, extra_strict_filtered = _environment_rerank(
                matches,
                current_context=current_context,
                scope=scope,
                top_k=candidate_limit,
            )
            strict_filtered += extra_strict_filtered
        else:
            reason = (
                "query_transform_timeout"
                if transform_status == "timeout"
                else "query_transform_unavailable"
            )
            degraded_reasons.append(reason)
            transform_trace = {
                "status": transform_status,
                "generated_queries": 0,
                "retrievals_completed": 0,
                "degraded_reasons": [reason],
            }

    if not matches:
        result = {
            "query": query,
            "keywords": keywords,
            "top_matches": [],
            "result_state": "no_match",
            "query_vector_cache_hit": query_vector_cache_hit,
            "feedback_queued": True,
            "strict_filtered": strict_filtered,
            "graph_enabled": graph_enabled,
            "transform_used": transform_used,
            "query_plan": query_plan,
            "transform_trace": transform_trace,
            "rerank_trace": {
                "enabled": bool(config.RERANK_ENABLED),
                "status": "not_run",
                "top_n": 0,
                "degraded_reason": None,
            },
            "degraded_reasons": degraded_reasons,
            "truncated": False,
            "truncated_reasons": [],
            "graph_trace": _empty_graph_trace(matches),
        }
        if result_cache_enabled and not degraded_reasons:
            query_cache.set_result(
                identity, result_cache_query, top_k, _cacheable_result(result)
            )
        return finish(result, mode="vector", degraded=bool(degraded_reasons))

    # ── Step 5: 受模式预算的 Memory Graph 扩散与去重排序 ──
    graph_started = performance.now()
    matches, graph_info, graph_strict_filtered = _graph_rerank(
        matches,
        top_k=candidate_limit,
        retrieval_source="vector",
        graph_enabled=graph_enabled,
        current_context=current_context,
        scope=scope,
        graph_budget=(
            {
                "max_hops": config.GRAPH_DEEP_MAX_HOPS,
                "max_paths": config.GRAPH_DEEP_MAX_PATHS,
                "fan_out": config.GRAPH_DEEP_FAN_OUT,
                "time_budget_ms": min(
                    config.GRAPH_DEEP_TIME_BUDGET_MS,
                    max(
                        0.0,
                        (
                            config.QUERY_DEEP_TOTAL_TIMEOUT_SECONDS
                            - max(0.0, performance.now() - total_started)
                            - config.RERANK_TIMEOUT_SECONDS
                        ) * 1000.0,
                    ),
                ),
                "token_budget": config.GRAPH_DEEP_TOKEN_BUDGET,
            }
            if retrieval_mode == "deep" else None
        ),
    )
    strict_filtered += graph_strict_filtered
    latency["graph"] = performance.elapsed_ms(graph_started)
    if graph_info["trace"].get("status") == "degraded":
        degraded_reasons.append(
            graph_info["trace"].get("degraded_reason", "graph_unavailable")
        )

    # ── Step 6: bounded local reranker with timeout/circuit fallback ──
    rerank_started = performance.now()
    matches, rerank_trace = adaptive.rerank(
        normalized_query,
        matches,
        enabled=rerank_enabled,
    )
    latency["rerank"] = performance.elapsed_ms(rerank_started)
    if rerank_trace.get("degraded_reason"):
        degraded_reasons.append(rerank_trace["degraded_reason"])
    matches = matches[:top_k]
    top_matches = _attach_related(matches)

    feedback_queued = feedback.enqueue_recall_feedback(
        [match["story_id"] for match in top_matches]
    )

    result = {
        "query": query,
        "keywords": keywords,
        "top_matches": top_matches,
        "result_state": "degraded_results" if degraded_reasons else "results",
        "query_vector_cache_hit": query_vector_cache_hit,
        "feedback_queued": feedback_queued,
        "strict_filtered": strict_filtered,
        "graph_enabled": graph_enabled,
        "transform_used": transform_used,
        "query_plan": query_plan,
        "transform_trace": transform_trace,
        "rerank_trace": rerank_trace,
        "degraded_reasons": degraded_reasons,
        "truncated": graph_info["truncated"],
        "truncated_reasons": graph_info["truncated_reasons"],
        "graph_trace": graph_info["trace"],
    }
    if result_cache_enabled and not degraded_reasons:
        query_cache.set_result(
            identity, result_cache_query, top_k, _cacheable_result(result)
        )

    return finish(result, mode="vector", degraded=bool(degraded_reasons))


def _run_second_stage(
    normalized_query: str,
    *,
    selected_transforms: list[str],
    retrieval_mode: str,
    identity: str,
    candidate_limit: int,
) -> dict:
    """Generate and retrieve transformed queries inside the caller deadline."""

    transform_timeout = (
        config.QUERY_DEEP_TRANSFORM_TIMEOUT_SECONDS
        if retrieval_mode == "deep"
        else config.QUERY_AUTO_TRANSFORM_TIMEOUT_SECONDS
    )
    payload = llm.transform_search_query(
        normalized_query,
        selected_transforms,
        timeout_seconds=transform_timeout,
    )
    if not payload:
        return {
            "rankings": [],
            "transform_used": [],
            "degraded_reasons": ["query_transform_unavailable"],
            "trace": {
                "status": "unavailable",
                "generated_queries": 0,
                "retrievals_completed": 0,
                "degraded_reasons": ["query_transform_unavailable"],
            },
        }

    variants: list[tuple[str, str]] = []
    if "rewrite" in selected_transforms and payload.get("rewrite"):
        variants.append(("rewrite", str(payload["rewrite"])))
    if "multi_query" in selected_transforms:
        for value in payload.get("queries", [])[:max(1, config.QUERY_MULTI_QUERY_LIMIT)]:
            if value:
                variants.append(("multi_query", str(value)))
    if "hyde" in selected_transforms and payload.get("hypothetical_document"):
        variants.append(("hyde", str(payload["hypothetical_document"])))

    deduplicated: list[tuple[str, str]] = []
    seen = {normalized_query}
    for transform, value in variants:
        candidate = adaptive.normalize_query(value)
        if candidate and candidate not in seen:
            deduplicated.append((transform, candidate))
            seen.add(candidate)
    variants = deduplicated[:max(1, config.QUERY_MULTI_QUERY_LIMIT + 2)]

    rankings: list[dict] = []
    used: list[str] = []
    degraded_reasons: list[str] = []
    for query_index, (transform, transformed_query) in enumerate(variants):
        vector = query_cache.get_query_vector(identity, transformed_query)
        if vector is None:
            embed_timeout = min(
                config.QUERY_WARM_TIMEOUT_SECONDS,
                max(0.05, transform_timeout),
            )
            embed_status, vector = _call_with_deadline(
                lambda value=transformed_query: embeddings.embed(
                    value,
                    timeout_seconds=embed_timeout,
                    keep_alive=config.EMBED_KEEP_ALIVE,
                ),
                embed_timeout,
            )
            if embed_status == "timeout":
                degraded_reasons.append("transformed_embedding_timeout")
            elif embed_status != "ok" or not vector:
                degraded_reasons.append("transformed_embedding_unavailable")
            else:
                embeddings.mark_model_used()
                query_cache.set_query_vector(identity, transformed_query, vector)

        vector_matches = []
        if vector:
            try:
                vector_matches = [
                    item for item in store.search_by_vector(
                        vector, top_k=candidate_limit
                    )
                    if item["similarity"] >= config.SIM_THRESHOLD_SEARCH
                ]
            except Exception:  # noqa: BLE001
                degraded_reasons.append("vector_index_unavailable")

        lexical_status, lexical_matches = _call_with_deadline(
            lambda value=transformed_query: store.search_by_lexical(
                value,
                top_k=candidate_limit,
                timeout_seconds=config.QUERY_FALLBACK_TIMEOUT_SECONDS,
            ),
            config.QUERY_FALLBACK_TIMEOUT_SECONDS,
        )
        if lexical_status != "ok" or lexical_matches is None:
            lexical_matches = []
            degraded_reasons.append("lexical_index_unavailable")
        fused = adaptive.fuse_rankings(
            vector_matches, lexical_matches, limit=candidate_limit
        )
        if fused:
            rankings.append({
                "transform": transform,
                "query_index": query_index,
                "matches": fused,
            })
        if transform not in used:
            used.append(transform)

    stable_reasons = list(dict.fromkeys(degraded_reasons))
    status = "ok" if variants else "unavailable"
    if not variants:
        stable_reasons.append("query_transform_unavailable")
    return {
        "rankings": rankings,
        "transform_used": used,
        "degraded_reasons": stable_reasons,
        "trace": {
            "status": status,
            "generated_queries": len(variants),
            "retrievals_completed": len(rankings),
            "transforms": used,
            "degraded_reasons": stable_reasons,
        },
    }


def _lexical_fallback(
    *,
    query: str,
    normalized_query: str,
    top_k: int,
    latency: dict[str, float],
    finish: Callable[..., dict],
    degraded_reason: str,
    query_vector_cache_hit: bool,
    current_context: dict | None,
    scope: str,
    graph_enabled: bool,
    retrieval_mode: str,
    transform_enabled: bool | None,
    rerank_enabled: bool | None,
) -> dict:
    fallback_started = performance.now()
    call_status, fallback_output = _call_with_deadline(
        lambda: _run_lexical_fallback(
            normalized_query,
            top_k,
            current_context=current_context,
            scope=scope,
            graph_enabled=graph_enabled,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
        ),
        config.QUERY_FALLBACK_TIMEOUT_SECONDS,
    )

    if call_status == "ok":
        (
            matches, top_matches, fallback_ms, graph_ms, strict_filtered,
            graph_info, rerank_trace, fallback_degraded_reasons,
        ) = fallback_output
        latency["fallback"] = fallback_ms
        latency["graph"] = graph_ms
        latency["rerank"] = rerank_trace.get("elapsed_ms", 0.0)
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
        strict_filtered = 0
        graph_info = {
            "truncated": False,
            "truncated_reasons": [],
            "trace": _empty_graph_trace([]),
        }
        rerank_trace = {
            "enabled": bool(config.RERANK_ENABLED),
            "status": "not_run",
            "top_n": 0,
            "degraded_reason": None,
        }
        fallback_degraded_reasons = []

    query_plan = adaptive.plan_query(
        normalized_query,
        top_matches,
        requested_mode=retrieval_mode,
        current_context=current_context,
        strict_filtered=strict_filtered,
        transform_enabled=transform_enabled,
    )
    # A missing primary embedding is already on the fast fallback path.  Do not
    # let a generative second stage delay or disguise that usable lexical result.
    query_plan["should_transform"] = False
    query_plan["effective_mode"] = "fast_fallback"
    query_plan["skip_reason"] = "primary_embedding_unavailable"
    degraded_reasons = [degraded_reason, *fallback_degraded_reasons]

    return finish(
        {
            "query": query,
            "keywords": [],
            "top_matches": top_matches,
            "strict_filtered": strict_filtered,
            "result_state": result_state,
            "fallback_status": fallback_status,
            "query_vector_cache_hit": query_vector_cache_hit,
            "feedback_queued": feedback_queued,
            "graph_enabled": graph_enabled,
            "transform_used": [],
            "query_plan": query_plan,
            "transform_trace": {
                "status": "skipped",
                "generated_queries": 0,
                "retrievals_completed": 0,
                "degraded_reasons": [],
            },
            "rerank_trace": rerank_trace,
            "degraded_reasons": degraded_reasons,
            "truncated": graph_info["truncated"],
            "truncated_reasons": graph_info["truncated_reasons"],
            "graph_trace": graph_info["trace"],
        },
        mode="lexical_fallback",
        degraded=True,
        degraded_reason=degraded_reason,
    )


def _run_lexical_fallback(
    normalized_query: str,
    top_k: int,
    *,
    current_context: dict | None,
    scope: str,
    graph_enabled: bool,
    retrieval_mode: str,
    rerank_enabled: bool | None,
) -> tuple[list[dict], list[dict], float, float, int, dict, dict, list[str]]:
    """在同一个 500ms deadline 内完成词法检索与关联读取。"""

    fallback_started = time.perf_counter()
    lexical_matches = store.search_by_lexical(
        normalized_query,
        top_k=max(top_k * 4, top_k, min(64, config.RERANK_TOP_N)),
        timeout_seconds=config.QUERY_FALLBACK_TIMEOUT_SECONDS,
    )
    fallback_ms = round((time.perf_counter() - fallback_started) * 1000, 3)
    candidate_limit = max(top_k * 4, top_k, min(64, config.RERANK_TOP_N))
    matches = adaptive.fuse_rankings([], lexical_matches, limit=candidate_limit)
    matches, strict_filtered = _environment_rerank(
        matches,
        current_context=current_context,
        scope=scope,
        top_k=candidate_limit,
    )
    graph_started = time.perf_counter()
    matches, graph_info, graph_strict_filtered = _graph_rerank(
        matches,
        top_k=candidate_limit,
        retrieval_source="lexical",
        graph_enabled=graph_enabled,
        current_context=current_context,
        scope=scope,
        graph_budget=(
            {
                "max_hops": config.GRAPH_DEEP_MAX_HOPS,
                "max_paths": config.GRAPH_DEEP_MAX_PATHS,
                "fan_out": config.GRAPH_DEEP_FAN_OUT,
                "time_budget_ms": config.GRAPH_DEEP_TIME_BUDGET_MS,
                "token_budget": config.GRAPH_DEEP_TOKEN_BUDGET,
            }
            if retrieval_mode == "deep" else None
        ),
    )
    strict_filtered += graph_strict_filtered
    matches, rerank_trace = adaptive.rerank(
        normalized_query, matches, enabled=rerank_enabled
    )
    matches = matches[:top_k]
    top_matches = _attach_related(matches)
    graph_ms = round((time.perf_counter() - graph_started) * 1000, 3)
    degraded_reasons = []
    if graph_info["trace"].get("status") == "degraded":
        degraded_reasons.append(
            graph_info["trace"].get("degraded_reason", "graph_unavailable")
        )
    if rerank_trace.get("degraded_reason"):
        degraded_reasons.append(rerank_trace["degraded_reason"])
    return (
        matches, top_matches, fallback_ms, graph_ms, strict_filtered, graph_info,
        rerank_trace, degraded_reasons,
    )


def _environment_rerank(
    matches: list[dict],
    *,
    current_context: dict | None,
    scope: str,
    top_k: int,
) -> tuple[list[dict], int]:
    """Apply the same semantic-first environment policy to every search lane.

    Similarities emitted by the vector and lexical stores are rounded to four
    decimals.  Environment fit therefore only breaks ties inside one semantic
    bucket; it must never make a less relevant Story outrank a more relevant
    one.  The public ``score`` mirrors that ordering without adding a positive
    bonus that can saturate at 1.0 and erase the tie-break signal.
    """

    reranked = []
    strict_filtered = 0
    for match in matches:
        fit = context_module.evaluate_story_context(
            current_context,
            match.get("environments"),
            match.get("applicability"),
        )
        if scope == "strict" and fit["strict_excluded"]:
            strict_filtered += 1
            continue
        match.update(fit)
        semantic_score = float(
            match.get("fusion_score", match.get("score", match["similarity"]))
        )
        match["score"] = _environment_rank_score(
            semantic_score,
            fit["environment_score"],
            has_context=current_context is not None,
        )
        components = match.setdefault("score_components", {})
        components.update({
            "environment_score": fit["environment_score"],
            "final_score": match["score"],
        })
        reranked.append(match)
    return (
        sorted(
            reranked,
            key=lambda item: (
                item.get("fusion_score", item["similarity"]),
                item["environment_score"],
                item.get("similarity", 0.0),
            ),
            reverse=True,
        )[:top_k],
        strict_filtered,
    )


def _environment_rank_score(
    similarity: float,
    environment_score: float,
    *,
    has_context: bool,
) -> float:
    """Encode the environment tie-break below one similarity score quantum."""

    similarity = max(0.0, min(1.0, float(similarity)))
    if not has_context:
        return round(similarity, 4)

    environment_score = max(-1.0, min(1.0, float(environment_score)))
    normalized_fit = (environment_score + 1.0) / 2.0
    similarity_quantum = 10 ** -4
    tie_break_span = (
        similarity_quantum
        * max(0.0, min(1.0, config.ENVIRONMENT_SCORE_WEIGHT))
    )
    # Penalising relative to the best possible fit keeps the score in [0, 1]
    # and preserves a visible difference even when similarity is exactly 1.0.
    penalty = (1.0 - normalized_fit) * tie_break_span
    return round(max(0.0, similarity - penalty), 8)


def _graph_rerank(
    matches: list[dict],
    *,
    top_k: int,
    retrieval_source: str,
    graph_enabled: bool,
    current_context: dict | None,
    scope: str,
    graph_budget: dict | None = None,
) -> tuple[list[dict], dict, int]:
    """Merge direct seeds with deduplicated graph candidates."""

    direct = []
    for match in matches:
        item = dict(match)
        components = dict(item.get("score_components", {}))
        components.update({
            "direct_similarity": item["similarity"],
            "environment_score": item.get("environment_score", 0.0),
            "graph_score": 0.0,
            "final_score": item.get("score", item["similarity"]),
        })
        item.update({
            "retrieval_source": item.get("retrieval_source", retrieval_source),
            "seed_story_id": item["story_id"],
            "graph_path": [],
            "score_components": components,
        })
        direct.append(item)
    if not graph_enabled:
        return direct[:top_k], {
            "truncated": False,
            "truncated_reasons": [],
            "trace": _empty_graph_trace(direct, status="disabled"),
        }, 0

    try:
        expansion = graph_module.expand(direct, **(graph_budget or {}))
    except Exception:  # noqa: BLE001 -- graph failure must preserve direct recall
        logger.warning("Memory Graph 扩散失败，已回退直接检索")
        trace = _empty_graph_trace(direct, status="degraded")
        trace["degraded_reason"] = "graph_unavailable"
        return direct[:top_k], {
            "truncated": False,
            "truncated_reasons": [],
            "trace": trace,
        }, 0

    strict_filtered = 0
    graph_matches = []
    for candidate in expansion["matches"]:
        fit = context_module.evaluate_story_context(
            current_context,
            candidate.get("environments"),
            candidate.get("applicability"),
        )
        if scope == "strict" and fit["strict_excluded"]:
            strict_filtered += 1
            continue
        candidate.update(fit)
        base_score = candidate["graph_score"]
        candidate["score"] = _environment_rank_score(
            base_score,
            fit["environment_score"],
            has_context=current_context is not None,
        )
        candidate["score_components"].update({
            "environment_score": fit["environment_score"],
            "final_score": candidate["score"],
        })
        graph_matches.append(candidate)

    suppressed = set(expansion["suppressed_story_ids"])
    best: dict[int, dict] = {}
    for item in [*direct, *graph_matches]:
        story_id = int(item["story_id"])
        if story_id in suppressed:
            continue
        existing = best.get(story_id)
        if existing is None or item["score"] > existing["score"]:
            best[story_id] = item
    ranked = sorted(
        best.values(),
        key=lambda item: (
            item["score"],
            item.get("similarity", 0.0),
            -int(item["story_id"]),
        ),
        reverse=True,
    )[:top_k]
    trace = dict(expansion["trace"])
    trace["status"] = "ok"
    trace["returned_story_ids"] = [item["story_id"] for item in ranked]
    return ranked, {
        "truncated": expansion["truncated"],
        "truncated_reasons": expansion["truncated_reasons"],
        "trace": trace,
    }, strict_filtered


def _empty_graph_trace(
    matches: list[dict], *, status: str = "not_run"
) -> dict:
    return {
        "status": status,
        "seed_story_ids": [item["story_id"] for item in matches],
        "expanded_candidates": 0,
        "paths_considered": 0,
        "tokens_used": 0,
        "cycles_suppressed": 0,
        "path_policy_suppressed": 0,
        "superseded_suppressed": 0,
        "elapsed_ms": 0.0,
        "budgets": {},
    }


def _attach_related(matches: list[dict]) -> list[dict]:
    related_by_story = store.get_related_stories_batch(
        [match["story_id"] for match in matches], limit=5
    )
    top_matches = []
    for match in matches:
        related = related_by_story.get(match["story_id"], [])
        rendered = {
            "story_id": match["story_id"],
            "title": match["title"],
            "abstract": match.get("abstract", match["content"]),
            "content": match["content"],
            "truncated": bool(match.get("truncated")),
            "keywords": match["keywords"],
            "similarity": match["similarity"],
            "score": match.get("score", match["similarity"]),
            "fusion_score": match.get(
                "fusion_score", match.get("score", match["similarity"])
            ),
            "environment_score": match.get("environment_score", 0.0),
            "environment": match.get("matched_environment"),
            "environments": match.get("environments", []),
            "applicability": match.get("applicability", {}),
            "warnings": match.get("warnings", []),
            "retrieval_source": match.get("retrieval_source", "vector"),
            "seed_story_id": match.get("seed_story_id", match["story_id"]),
            "graph_path": match.get("graph_path", []),
            "score_components": match.get("score_components", {}),
            "source_paths": match.get("source_paths", []),
            "related": [
                {
                    "story_id": item["id"],
                    "title": item["title"],
                    "content": item.get("abstract") or item["content"],
                    "truncated": bool(
                        item.get("abstract")
                        and item.get("abstract") != item.get("content")
                    ),
                    "weight": item.get("weight", 0),
                    "edge_type": item.get("edge_type", "semantic"),
                    "directed": bool(item.get("directed")),
                    "direction": item.get("edge_direction", "undirected"),
                    "provenance": item.get("edge_provenance", {}),
                    "version": item.get("edge_version", 1),
                }
                for item in related
            ],
        }
        top_matches.append(rendered)
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
            f"   {result.get('retrieval_mode', 'fast')}/"
            f"{result.get('mode', 'unknown')} · "
            f"{result['latency_ms'].get('total', 0):.1f}ms"
        )

    if result.get("transform_used"):
        lines.append(
            f"   查询增强: {', '.join(result['transform_used'])} · "
            f"触发原因: {', '.join(result.get('query_plan', {}).get('trigger_reasons', []))}"
        )

    if result.get("keywords"):
        lines.append(f"   关键词: {', '.join(result['keywords'])}")

    if result.get("degraded") and result.get("top_matches"):
        lines.append(
            f"   ⚠️ 检索组件已降级（{', '.join(result.get('degraded_reasons', [])) or 'unknown'}），"
            "以下为可用 fallback 结果"
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
        if m.get("environment"):
            lines.append(
                f"   │ 来源环境: {context_module.environment_label(m['environment'])}"
            )
        applies, excludes = context_module.applicability_labels(m.get("applicability"))
        if applies:
            lines.append(f"   │ 适用于: {'; '.join(applies)}")
        if excludes:
            lines.append(f"   │ 不适用于: {'; '.join(excludes)}")
        for warning in m.get("warnings", []):
            lines.append(f"   │ ⚠ 当前环境差异: {warning}")

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
