"""
检索激活模块 — 用户输入问题 → Top3 story + 关联推送
"""
import json
import logging
import uuid

from . import config, embeddings, performance, store
from . import context as context_module

logger = logging.getLogger(__name__)


def search(
    query: str,
    top_k: int | None = None,
    *,
    context: dict | None = None,
    scope: str = "profile",
    record_diagnostics: bool = True,
) -> dict:
    """
    检索记忆：
    1. 直接对 query 生成向量
    2. 向量检索 Top-K story
    3. 每个 story 下方按权重展示关联 story

    返回: {query, keywords, top_matches: [{story, similarity, related: [...]}]}
    """
    top_k = top_k or config.TOP_K_SEARCH
    if scope not in ("profile", "strict"):
        raise ValueError("scope 必须是 profile 或 strict")
    current_context = (
        context_module.normalize_envelope(context, profile_id=config.PROFILE_ID)
        if context is not None else None
    )
    request_id = uuid.uuid4().hex
    latency = performance.empty_latency()
    total_started = performance.now()

    def finish(result: dict, *, mode: str, degraded: bool = False,
               degraded_reason: str | None = None) -> dict:
        result.update({
            "request_id": request_id,
            "mode": mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
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
                degraded=degraded,
                degraded_reason=degraded_reason,
            )
        return result

    # ── Step 1: 生成查询向量 ──
    # 直接 embed 原始 query。早先用 llm.extract_keywords 抽关键词再拼 query 做 embedding，
    # 但搜索 query 通常很短，关键词提取器为凑 5-10 个配额会幻觉出无关词（如"内存泄漏""无限循环"），
    # 污染向量、把检索带偏。短 query 本身已是聚焦点，直接 embed 更准且省一次 LLM 调用。
    embed_started = performance.now()
    query_vec = embeddings.embed(query)
    latency["embed"] = performance.elapsed_ms(embed_started)
    keywords = []   # 不再提取；保留字段以兼容返回结构

    if not query_vec:
        return finish(
            {
                "query": query,
                "keywords": [],
                "top_matches": [],
                "error": "向量生成失败",
            },
            mode="unavailable",
            degraded=True,
            degraded_reason="embedding_unavailable",
        )

    # ── Step 2: 向量检索 ──
    vector_started = performance.now()
    matches = store.search_by_vector(query_vec, top_k=max(top_k * 4, top_k))
    latency["vector"] = performance.elapsed_ms(vector_started)

    # 过滤低于阈值的
    rerank_started = performance.now()
    matches = [m for m in matches if m["similarity"] >= config.SIM_THRESHOLD_SEARCH]
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
        # Environment is a bounded secondary signal; semantic relevance stays dominant.
        match.update(fit)
        match["score"] = round(
            max(0.0, min(
                1.0,
                match["similarity"]
                + config.ENVIRONMENT_SCORE_WEIGHT * fit["environment_score"],
            )),
            4,
        )
        reranked.append(match)
    matches = sorted(
        reranked,
        key=lambda item: (item["score"], item["similarity"]),
        reverse=True,
    )[:top_k]
    latency["rerank"] = performance.elapsed_ms(rerank_started)

    if not matches:
        return finish(
            {
                "query": query,
                "keywords": keywords,
                "top_matches": [],
                "strict_filtered": strict_filtered,
            },
            mode="vector",
        )

    # ── Step 3: 关联激活 ──
    graph_started = performance.now()
    top_matches = []
    for m in matches:
        # 增加访问计数
        store.increment_access_count(m["story_id"])

        # 获取关联 story
        related = store.get_related_stories(m["story_id"], limit=5)

        # 提升被一起检索到的 story 之间的关联权重（无向，每对仅提升一次）
        for other_m in matches:
            if other_m["story_id"] > m["story_id"]:
                store.increment_edge_weight(m["story_id"], other_m["story_id"])

        top_matches.append({
            "story_id": m["story_id"],
            "title": m["title"],
            "content": m["content"],
            "keywords": m["keywords"],
            "similarity": m["similarity"],
            "score": m.get("score", m["similarity"]),
            "environment_score": m.get("environment_score", 0.0),
            "environment": m.get("matched_environment"),
            "environments": m.get("environments", []),
            "applicability": m.get("applicability", {}),
            "warnings": m.get("warnings", []),
            "related": [
                {
                    "story_id": r["id"],
                    "title": r["title"],
                    "content": r["content"],
                    "weight": r.get("weight", 0),
                    "edge_type": r.get("edge_type", "semantic"),
                }
                for r in related
            ],
        })
    latency["graph"] = performance.elapsed_ms(graph_started)

    return finish(
        {
            "query": query,
            "keywords": keywords,
            "top_matches": top_matches,
            "strict_filtered": strict_filtered,
        },
        mode="vector",
    )


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

    if not result.get("top_matches"):
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
