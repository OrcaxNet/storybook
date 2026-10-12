"""MCP server：把记忆检索暴露给 MCP-aware agent（如 Claude Code）运行时召回。

作为独立 stdio 进程运行，**复用**现有 ``search.search`` / ``store.get_story`` /
``store.get_stats`` / ``prime.prime_context``，不重复实现检索逻辑。

工具：
  - recall(query, top_k?)        缓存/向量/词法降级 + 关联激活（反馈写异步入队）
  - get_story(story_id)          查看单条记忆详情（含关联记忆）
  - stats()                      记忆库概况
  - prime_context(cwd, first_prompt?, top_k?)
                                  会话启动主动注入（晨间简报）：基于 cwd + 首条提问召回并生成
                                  ≤2k token 的精简摘要，相关度不足时静默不注入

启动方式（二选一，均为独立进程，不依赖 CLI 运行态）：
  book mcp                        # 经 CLI 入口
  python -m storybook.mcp_server  # 直接跑模块（editable 安装后即可）

MCP SDK 保持延迟导入，便于核心逻辑隔离测试；v0.2 起已是基础安装依赖。
"""
from __future__ import annotations

import logging

from . import embeddings, feedback, store  # embeddings 提供预热；DB 初始化由 _ensure_db 显式触发
from . import prime as prime_module
from . import search as search_module

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  启动准备
# ═══════════════════════════════════════════════

def _ensure_db() -> None:
    """启动时确保 schema 存在（best-effort）。

    失败仅记录日志，不阻断启动：工具调用时会按各自路径报错并给出可操作提示。
    这样在全新环境（尚未 ``book init``）下 server 仍可启动，
    ``recall`` 返回空、``stats`` 返回 0、``get_story`` 报不存在。
    """
    try:
        store.init_db()
    except Exception as e:  # noqa: BLE001
        logger.warning("init_db 失败，工具调用时将按需报错: %s", e)


def _prewarm_embedding() -> None:
    """MCP 启动时 best-effort 预热，失败后查询仍可走词法降级。"""

    if not embeddings.prewarm():
        logger.warning("embedding 预热失败；recall 将在需要时走快速降级")


# ═══════════════════════════════════════════════
#  结果裁剪：对 agent 友好（精简、含相似度）
# ═══════════════════════════════════════════════

def _trim_related(related: list[dict], *, verbose: bool = True) -> list[dict]:
    """精简关联 story：保留 id/标题/权重/边类型，去掉全文 content 与 embedding。

    关联记忆是次要信息，agent 需要详情可再调 ``get_story``，避免一次 recall 塞入
    十几条全文 story 造成上下文噪声。
    """
    trimmed = []
    for r in related or []:
        sid = r.get("story_id", r.get("id"))
        trimmed.append({
            "story_id": sid,
            "title": r.get("title"),
            "weight": r.get("weight", 0),
            "edge_type": r.get("edge_type", "semantic"),
        })
        if verbose:
            trimmed[-1].update({
                "directed": bool(r.get("directed")),
                "direction": r.get("direction", r.get("edge_direction", "undirected")),
                "provenance": r.get("provenance", r.get("edge_provenance", {})),
                "version": r.get("version", r.get("edge_version", 1)),
            })
    return trimmed


def _compact_environment(env: dict | None) -> dict | None:
    """把完整 ContextEnvelope 压缩为判断适用性所需的少量字段。

    完整 envelope 含 provenance/哈希等 30+ 字段，每条召回结果重复多份会让一次
    recall 占用数千 token；Agent 只需要"在哪个工具/项目/系统/运行时下发生"。
    """
    if not isinstance(env, dict):
        return None
    tool = env.get("tool") or {}
    workspace = env.get("workspace") or {}
    device = env.get("device") or {}
    runtime = env.get("runtime") or {}
    compact = {
        "tool": tool.get("type"),
        "tool_version": tool.get("version"),
        "project": workspace.get("project_label"),
        "branch": workspace.get("branch"),
        "os": device.get("os_family"),
        "runtime": runtime.get("kind"),
        "captured_at": env.get("captured_at"),
    }
    return {k: v for k, v in compact.items() if v not in (None, "", "unknown")}


def _compact_environments(envs: list | None, limit: int = 3) -> list[dict]:
    seen: list[dict] = []
    for env in envs or []:
        compact = _compact_environment(env)
        if compact is None:
            continue
        key = {k: v for k, v in compact.items() if k != "captured_at"}
        if any({k: v for k, v in s.items() if k != "captured_at"} == key for s in seen):
            continue
        seen.append(compact)
    return seen[-limit:]


def _build_recall_result(result: dict, *, verbose: bool = False) -> dict:
    """把 ``search.search`` 的返回裁剪为对 agent 友好的结构。

    与 CLI ``search`` 数据结构一致（query / matches 含 story_id/title/content/
    keywords/similarity/related），唯一精简：related 条目去掉全文 content。
    新增 ``count`` 字段便于 agent 快速判断是否有命中。
    """
    matches = []
    for m in result.get("top_matches", []):
        matches.append({
            "story_id": m["story_id"],
            "title": m["title"],
            "abstract": m.get("abstract", m["content"]),
            "content": m["content"],
            "truncated": bool(m.get("truncated")),
            "keywords": m["keywords"],
            "similarity": m["similarity"],
            "score": m.get("score", m["similarity"]),
            "fusion_score": m.get("fusion_score", m.get("score", m["similarity"])),
            "environment_score": m.get("environment_score", 0.0),
            "environment": (
                m.get("environment") if verbose
                else _compact_environment(m.get("environment"))
            ),
            "environments": (
                m.get("environments", []) if verbose
                else _compact_environments(m.get("environments"))
            ),
            "applicability": m.get("applicability", {}),
            "warnings": m.get("warnings", []),
            "retrieval_source": m.get("retrieval_source", "vector"),
            "seed_story_id": m.get("seed_story_id", m["story_id"]),
            "related": _trim_related(m.get("related", []), verbose=verbose),
        })
        if verbose:
            matches[-1].update({
                "graph_path": m.get("graph_path", []),
                "score_components": m.get("score_components", {}),
                "source_paths": m.get("source_paths", []),
            })
    out = {
        "query": result.get("query"),
        "count": len(matches),
        "matches": matches,
        "request_id": result.get("request_id"),
        "mode": result.get("mode", "vector"),
        "retrieval_mode": result.get("retrieval_mode", "fast"),
        "degraded": bool(result.get("degraded")),
        "degraded_reason": result.get("degraded_reason"),
        "degraded_reasons": result.get("degraded_reasons", []),
        "result_state": result.get("result_state", "results" if matches else "no_match"),
        "fallback_status": result.get("fallback_status"),
        "cache_hit": bool(result.get("cache_hit")),
        "index_version": result.get("index_version"),
        "scope": result.get("scope", "profile"),
        "strict_filtered": result.get("strict_filtered", 0),
        "graph_enabled": bool(result.get("graph_enabled")),
        "truncated": bool(result.get("truncated")),
        "truncated_reasons": result.get("truncated_reasons", []),
        "transform_used": result.get("transform_used", []),
    }
    if verbose:
        # 诊断字段：默认不返回，避免占用 Agent 主任务上下文。
        out.update({
            "latency_ms": result.get("latency_ms", {}),
            "graph_trace": result.get("graph_trace", {}),
            "query_plan": result.get("query_plan", {}),
            "transform_trace": result.get("transform_trace", {}),
            "rerank_trace": result.get("rerank_trace", {}),
        })
    return out


# ═══════════════════════════════════════════════
#  工具核心逻辑（模块级，便于不依赖 mcp 直接单测）
# ═══════════════════════════════════════════════

def recall_memories(
    query: str,
    top_k: int = 3,
    context: dict | None = None,
    scope: str = "profile",
    graph_enabled: bool | None = None,
    retrieval_mode: str = "fast",
    transform_enabled: bool | None = None,
    rerank_enabled: bool | None = None,
    verbose: bool = False,
    telemetry_origin: str | None = None,
) -> dict:
    """召回与查询相关的记忆（向量检索 + 关联激活）。

    复用 ``search.search``；access_count 与共同召回边权反馈异步入队，
    不占用 recall 响应热路径。

    返回 ``{query, count, matches: [...]}``；``count == 0`` 表示无匹配
    （记忆库为空或无相关记忆），此时 ``matches`` 为空列表，不返回噪声。
    embedding 不可用或超时时返回 ``degraded=true``，并尝试 500ms 内的词法降级；
    ``result_state`` 可区分正常无匹配与降级空结果。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    result = search_module.search(
        query,
        top_k=top_k,
        context=context,
        scope=scope,
        graph_enabled=graph_enabled,
        retrieval_mode=retrieval_mode,
        transform_enabled=transform_enabled,
        rerank_enabled=rerank_enabled,
    )
    if telemetry_origin and result.get("request_id"):
        workspace = ((result.get("context") or {}).get("workspace") or {})
        feedback.enqueue_recall_event(
            request_id=result["request_id"],
            origin=telemetry_origin,
            query=query,
            matches=result.get("top_matches", []),
            workspace_path_hash=workspace.get("path_fingerprint"),
            result_state=result.get("result_state"),
        )
    return _build_recall_result(result, verbose=verbose)


def get_story_detail(
    story_id: int, *, verbose: bool = False, record_click: bool = False
) -> dict:
    """查看单条记忆详情（含关联记忆）。

    复用 ``store.get_story`` + ``store.get_related_stories``。输出剥离 embedding
    向量（1024 维浮点，对 agent 无用且臃肿）。不存在时抛 ``ValueError``。
    """
    story = store.get_story(story_id)
    if not story:
        raise ValueError(f"Story #{story_id} 不存在")

    if record_click:
        feedback.enqueue_story_click(story["id"])
    related = store.get_related_stories(story_id, limit=10)
    return {
        "story_id": story["id"],
        "title": story["title"],
        "abstract": story.get("abstract", ""),
        "content": story["content"],
        "detail": story.get("detail", {}),
        "sources": story.get("sources", []),
        "keywords": story.get("keywords", []),
        "access_count": story.get("access_count", 0),
        "version": story.get("version", 1),
        "embedding_model": story.get("embedding_model"),
        "embedding_version": story.get("embedding_version"),
        "embedding_content_hash": story.get("embedding_content_hash"),
        "parent_id": story.get("parent_id"),
        "source_session_ids": story.get("source_session_ids", []),
        "environments": (
            story.get("environments", []) if verbose
            else _compact_environments(story.get("environments"), limit=5)
        ),
        "applicability": story.get("applicability", {}),
        "created_at": story.get("created_at"),
        "updated_at": story.get("updated_at"),
        # 每个修订都带完整快照，默认只给版本链，避免一条详情膨胀为全部历史。
        "revisions": [
            revision if verbose else {
                k: revision.get(k) for k in ("version", "event_type", "created_at")
            }
            for revision in store.get_story_revisions(story_id)
        ],
        "related": _trim_related(related, verbose=verbose),
    }


def get_stats_overview() -> dict:
    """记忆库概况。复用 ``store.get_stats``。

    全 0 通常表示记忆库为空（先 ``book run``
    沉淀记忆）。schema 未初始化时返回 0 值 + ``note``，不抛错。
    """
    try:
        return store.get_stats()
    except Exception as e:  # noqa: BLE001  -- schema 未初始化等
        return {
            "sessions": 0,
            "pending": 0,
            "processed": 0,
            "stories": 0,
            "edges": 0,
            "root_stories": 0,
            "child_stories": 0,
            "note": f"读取统计失败：{e}。可能数据库未初始化，请运行 `book init`。",
        }


def prime_context_memories(cwd: str = "", first_prompt: str = "",
                           top_k: int = 5,
                           telemetry_origin: str | None = None) -> dict:
    """会话启动主动注入（晨间简报）。直接转调 ``prime.prime_context``。

    单独包一层仅为与 ``recall_memories`` / ``get_story_detail`` / ``get_stats_overview``
    保持"模块级核心逻辑 + MCP 装配"的分层一致，便于不依赖 mcp SDK 直接单测。
    语义见 ``prime.prime_context``：相关度不足 / 召回为空 / embedding API 不可用时
    ``injected=False``、``briefing=""``，不抛错（晨间简报须非侵入）。
    """
    return prime_module.prime_context(
        cwd=cwd,
        first_prompt=first_prompt,
        top_k=top_k,
        tool_type="other",
        integration_mode="mcp",
        telemetry_origin=telemetry_origin,
    )


# ═══════════════════════════════════════════════
#  FastMCP server 装配（延迟导入 mcp）
# ═══════════════════════════════════════════════

def create_server():
    """构造并返回 FastMCP server，注册四个工具。

    ``mcp`` SDK 在此处延迟导入；缺失时由 ``main`` 转为可操作提示。
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("storybook")

    @mcp.tool()
    def recall(
        query: str,
        top_k: int = 3,
        context: dict | None = None,
        scope: str = "profile",
        graph_enabled: bool | None = None,
        retrieval_mode: str = "fast",
        transform_enabled: bool | None = None,
        rerank_enabled: bool | None = None,
        verbose: bool = False,
    ) -> dict:
        """召回与当前任务/问题相关的记忆（向量检索 + 关联激活）。

        在开始一项新任务前调用，复用过往相似经历。返回 count 表示命中数；
        count=0 表示无相关记忆，应直接继续而非反复重试。related 仅含摘要，
        需要某条关联的全文请用 get_story。embedding 不可用或超时时返回
        degraded 状态与词法 fallback；可用 `book doctor` 排查环境。

        Args:
            query: 自然语言查询，描述当前任务或问题。
            top_k: 最多返回的匹配记忆数，默认 3。
            context: 可选 ContextEnvelope；缺省时仅按 Profile 语义召回。
            scope: profile（默认软加权）、project（仅当前项目）或 strict（环境冲突硬过滤）。
            graph_enabled: 是否启用受预算图扩散；缺省跟随本地配置，false 回退直接检索。
            retrieval_mode: fast（无生成式 LLM）、auto（按门控进入第二阶段）或 deep（显式深度召回）。
            transform_enabled: 可选总开关；false 时 auto/deep 只返回 fast fallback。
            rerank_enabled: 可选本地 reranker 开关；故障时保留融合结果。
            verbose: 返回完整环境、图路径与检索诊断（排查用；默认精简）。
        """
        return recall_memories(
            query,
            top_k=top_k,
            context=context,
            scope=scope,
            graph_enabled=graph_enabled,
            retrieval_mode=retrieval_mode,
            transform_enabled=transform_enabled,
            rerank_enabled=rerank_enabled,
            verbose=verbose,
            telemetry_origin="mcp_recall",
        )

    @mcp.tool()
    def get_story(story_id: int, verbose: bool = False) -> dict:
        """查看单条记忆的完整详情（含关联记忆）。

        当 recall 返回的某条记忆需要展开看全文/关联时调用。story_id 来自 recall
        返回的 matches[].story_id 或 related[].story_id。不存在会报错。

        Args:
            story_id: 记忆编号。
            verbose: 返回完整环境与修订快照（默认精简）。
        """
        return get_story_detail(story_id, verbose=verbose, record_click=True)

    @mcp.tool()
    def stats() -> dict:
        """查看记忆库概况（会话/Story/关联边数量）。

        全 0 通常意味着记忆库为空，需先导入并加工会话
        （`book run`）。
        """
        return get_stats_overview()

    @mcp.tool()
    def prime_context(cwd: str = "", first_prompt: str = "", top_k: int = 5) -> dict:
        """会话启动主动注入（晨间简报）：基于当前项目目录 + 首条提问召回相关记忆并生成精简摘要。

        在新会话开始、读到用户首条提问后调用，实现"下意识回忆"过往相似经历。
        返回 ``injected=True`` 时 ``briefing`` 为可直接呈现给用户的简报（≤2k token）；
        ``injected=False`` 表示无相关记忆或环境不可用（不报错，静默不注入，``note`` 给出原因）。
        命中记忆会自增 access_count（与 recall 同源副作用）。

        Args:
            cwd: 当前项目目录（agent 的工作目录）。为空且 first_prompt 也为空时静默不注入。
            first_prompt: 用户的首条提问/任务描述（可选；无则仅用 cwd 派生查询）。
            top_k: 最多考虑的候选记忆数，默认 5（再按 token 预算裁剪）。
        """
        out = prime_context_memories(
            cwd=cwd, first_prompt=first_prompt, top_k=top_k,
            telemetry_origin="mcp_prime",
        )
        # 完整 ContextEnvelope 对 Agent 无用；只保留精简环境，降低每次会话开头的 token。
        out["context"] = _compact_environment(out.get("context"))
        return out

    return mcp


def main() -> None:
    """MCP server 入口：确保 DB -> 装配 server -> 以 stdio 运行。"""
    _ensure_db()
    _prewarm_embedding()
    try:
        mcp = create_server()
    except ModuleNotFoundError as e:
        if "mcp" in str(e).lower():
            raise SystemExit(
                "未安装 MCP SDK。请重新安装 Storybook 基础依赖"
            )
        raise
    mcp.run()  # 默认 stdio 传输


if __name__ == "__main__":
    main()
