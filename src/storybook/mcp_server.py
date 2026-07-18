"""MCP server：把记忆检索暴露给 MCP-aware agent（如 Claude Code）运行时召回。

作为独立 stdio 进程运行，**复用**现有 ``search.search`` / ``store.get_story`` /
``store.get_stats``，不重复实现检索逻辑。

工具：
  - recall(query, top_k?)  向量检索 + 关联激活（与 CLI ``search`` 同源，含 access_count/边权提权副作用）
  - get_story(story_id)    查看单条记忆详情（含关联记忆）
  - stats()                记忆库概况

启动方式（二选一，均为独立进程，不依赖 CLI 运行态）：
  storybook mcp                   # 经 CLI 入口（需安装 [mcp] extra）
  python -m storybook.mcp_server  # 直接跑模块（editable 安装后即可）

MCP SDK 采用延迟导入：base 安装无需 ``mcp`` 依赖，仅在使用 MCP server 时才需要。
"""
from __future__ import annotations

import logging
from typing import Any

from . import config  # noqa: F401  -- 触发 config 初始化（.env 自动加载、目录创建）
from . import store
from . import search as search_module

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  启动准备
# ═══════════════════════════════════════════════

def _ensure_db() -> None:
    """启动时确保 schema 存在（best-effort）。

    失败仅记录日志，不阻断启动：工具调用时会按各自路径报错并给出可操作提示。
    这样在全新环境（尚未 ``storybook init``）下 server 仍可启动，
    ``recall`` 返回空、``stats`` 返回 0、``get_story`` 报不存在。
    """
    try:
        store.init_db()
    except Exception as e:  # noqa: BLE001
        logger.warning("init_db 失败，工具调用时将按需报错: %s", e)


# ═══════════════════════════════════════════════
#  结果裁剪：对 agent 友好（精简、含相似度）
# ═══════════════════════════════════════════════

def _trim_related(related: list[dict]) -> list[dict]:
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
    return trimmed


def _build_recall_result(result: dict) -> dict:
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
            "content": m["content"],
            "keywords": m["keywords"],
            "similarity": m["similarity"],
            "related": _trim_related(m.get("related", [])),
        })
    return {
        "query": result.get("query"),
        "count": len(matches),
        "matches": matches,
    }


# ═══════════════════════════════════════════════
#  工具核心逻辑（模块级，便于不依赖 mcp 直接单测）
# ═══════════════════════════════════════════════

def recall_memories(query: str, top_k: int = 3) -> dict:
    """召回与查询相关的记忆（向量检索 + 关联激活）。

    复用 ``search.search``，继承其 access_count 自增与共同召回边权提权副作用
    （与人脑"反复回忆加深记忆路径"的隐喻一致）。

    返回 ``{query, count, matches: [...]}``；``count == 0`` 表示无匹配
    （记忆库为空或无相关记忆），此时 ``matches`` 为空列表，不返回噪声。
    embedding 不可用（Ollama 未运行等）时抛 ``RuntimeError`` 并给出可操作提示。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    result = search_module.search(query, top_k=top_k)
    if result.get("error"):
        # search.search 在向量生成失败时返回 error 键；对 agent 而言这是环境问题而非"无匹配"
        raise RuntimeError(
            f"recall 失败：{result['error']}。请确认 Ollama 已运行且 embedding 模型"
            f"（{config.EMBED_MODEL}）可用，可运行 `storybook doctor` 排查。"
        )
    return _build_recall_result(result)


def get_story_detail(story_id: int) -> dict:
    """查看单条记忆详情（含关联记忆）。

    复用 ``store.get_story`` + ``store.get_related_stories``。输出剥离 embedding
    向量（1024 维浮点，对 agent 无用且臃肿）。不存在时抛 ``ValueError``。
    """
    story = store.get_story(story_id)
    if not story:
        raise ValueError(f"Story #{story_id} 不存在")

    related = store.get_related_stories(story_id, limit=10)
    return {
        "story_id": story["id"],
        "title": story["title"],
        "content": story["content"],
        "keywords": story.get("keywords", []),
        "access_count": story.get("access_count", 0),
        "version": story.get("version", 1),
        "parent_id": story.get("parent_id"),
        "source_session_ids": story.get("source_session_ids", []),
        "created_at": story.get("created_at"),
        "updated_at": story.get("updated_at"),
        "related": _trim_related(related),
    }


def get_stats_overview() -> dict:
    """记忆库概况。复用 ``store.get_stats``。

    全 0 通常表示记忆库为空（先 ``storybook import-data`` + ``storybook process``
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
            "note": f"读取统计失败：{e}。可能数据库未初始化，请运行 `storybook init`。",
        }


# ═══════════════════════════════════════════════
#  FastMCP server 装配（延迟导入 mcp）
# ═══════════════════════════════════════════════

def create_server():
    """构造并返回 FastMCP server，注册三个工具。

    ``mcp`` SDK 在此处延迟导入：未安装 [mcp] extra 时抛 ModuleNotFoundError，
    由 ``main`` 转为可操作提示。
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("storybook")

    @mcp.tool()
    def recall(query: str, top_k: int = 3) -> dict:
        """召回与当前任务/问题相关的记忆（向量检索 + 关联激活）。

        在开始一项新任务前调用，复用过往相似经历。返回 count 表示命中数；
        count=0 表示无相关记忆，应直接继续而非反复重试。related 仅含摘要，
        需要某条关联的全文请用 get_story。embedding 不可用时会报错
        （先确认 Ollama 运行，或 `storybook doctor`）。

        Args:
            query: 自然语言查询，描述当前任务或问题。
            top_k: 最多返回的匹配记忆数，默认 3。
        """
        return recall_memories(query, top_k=top_k)

    @mcp.tool()
    def get_story(story_id: int) -> dict:
        """查看单条记忆的完整详情（含关联记忆）。

        当 recall 返回的某条记忆需要展开看全文/关联时调用。story_id 来自 recall
        返回的 matches[].story_id 或 related[].story_id。不存在会报错。

        Args:
            story_id: 记忆编号。
        """
        return get_story_detail(story_id)

    @mcp.tool()
    def stats() -> dict:
        """查看记忆库概况（会话/Story/关联边数量）。

        全 0 通常意味着记忆库为空，需先导入并加工会话
        （`storybook import-data` + `storybook process`）。
        """
        return get_stats_overview()

    return mcp


def main() -> None:
    """MCP server 入口：确保 DB -> 装配 server -> 以 stdio 运行。"""
    _ensure_db()
    try:
        mcp = create_server()
    except ModuleNotFoundError as e:
        if "mcp" in str(e).lower():
            raise SystemExit(
                "未安装 MCP SDK。请运行：uv pip install -e \".[mcp]\""
            )
        raise
    mcp.run()  # 默认 stdio 传输


if __name__ == "__main__":
    main()
