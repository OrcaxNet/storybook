"""会话启动主动注入（上下文预热 / 晨间简报）。

新会话开始时，基于 cwd（项目目录）+ 可选首条提问，主动召回 top-N 相关 story，
生成精简"相关记忆"摘要注入上下文--更贴近项目初衷：人脑处理一个事项时会自动
"下意识回忆"相关经历，而非等被问到才去检索。

两条触发路径共享本模块的召回 + 预算控制逻辑（都不重复实现检索，复用 ``search.search``）：

  - Claude Code ``SessionStart`` hook -> ``book prime`` CLI（hook 友好的纯文本输出，
    写到 stdout 即被注入为额外上下文；无输出即不注入）。
  - MCP server ``prime_context(cwd, first_prompt?)`` 工具 -> agent 在读到用户首条提问后
    主动调用，拿回结构化简报自行呈现。

设计要点
--------
  - **复用** ``search.search`` 召回（含 ``access_count`` 自增与共同召回边权提权副作用--
    晨间简报本身即一次"回忆"，提权符合"反复回忆加深记忆路径"的隐喻）。
  - **更高相关度门槛** ``config.PRIME_MIN_SIMILARITY``（默认 0.60，高于检索 0.50）：
    主动注入是"不打扰"的，只有较相关的记忆才进简报，避免噪声。
  - **token 预算控制** ``config.PRIME_TOKEN_BUDGET``（默认 2000）：超额时按相似度从低到高
    丢弃候选，并对单条摘要按字符裁剪，保证简报不污染上下文。
  - **静默不注入**：召回为空 / 相关度不足 / embedding API 不可用时，``injected=False``、
    ``briefing=""``、不抛错（晨间简报须非侵入：宁可静默也不报错污染上下文）。
"""
from __future__ import annotations

import logging
import re

from . import config
from . import search as search_module
from . import context as context_module

logger = logging.getLogger(__name__)

# CJK / 日韩字符范围，用于 token 估算时区分"1 字符≈1 token"的中日韩文与其余文本
_CJK_RE = re.compile(
    r"[㐀-鿿豈-﫿"      # CJK 统一表意 / 兼容
    r"぀-ヿㇰ-ㇿ"        # 日文假名
    r"가-힯]"                    # 韩文谚文
)

# 过于通用、信息量低的路径段，从 cwd 派生 query 时跳过
_GENERIC_SEGMENTS = frozenset({
    "src", "code", "app", "apps", "application", "applications",
    "project", "projects", "workspace", "workspaces", "work",
    "home", "users", "usr", "var", "tmp", "temp", "dev", "etc",
    "bin", "lib", "libs", "dist", "build", "out", "node_modules",
    "packages", "main", "master", "trunk",
})


# ═══════════════════════════════════════════════
#  token 估算（预算控制用，非精确计数）
# ═══════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """粗估 ``text`` 的 token 数，用于简报预算控制（非精确计数）。

    混合中英文启发式：CJK / 日韩字符约 1 token，其余约 4 字符/token。
    刻意偏保守（略高估），避免实际 token 超出 ``PRIME_TOKEN_BUDGET``。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _CJK_RE.match(ch))
    other = len(text) - cjk
    return cjk + (other + 3) // 4   # +3 向上取整


# ═══════════════════════════════════════════════
#  query 构造
# ═══════════════════════════════════════════════

def _query_from_cwd(cwd: str) -> str:
    """从 cwd 派生检索 query：取最有信息量的路径段（通常是项目名）。

    basename 一般就是项目名（如 ``payment-service``）；若 basename 过于通用
    （src/code/app 等）则向前找第一个有信息量的段；都没有时回退到末段。
    ``-`` / ``_`` 替换为空格，让 embedding 更接近自然语言。
    """
    if not cwd:
        return ""
    parts = [p for p in cwd.replace("\\", "/").split("/") if p]
    if not parts:
        return cwd
    for p in reversed(parts):
        if p.startswith("."):
            continue
        if p.lower() in _GENERIC_SEGMENTS:
            continue
        return p.replace("-", " ").replace("_", " ")
    return parts[-1].replace("-", " ").replace("_", " ")


def build_query(cwd: str = "", first_prompt: str = "") -> str:
    """构造检索 query：首条提问优先（任务最强信号），无则用 cwd 派生的项目上下文。

    SessionStart hook 触发时尚无首条提问，故仅用 cwd；MCP ``prime_context`` 在 agent
    读到首条提问后调用，传入 ``first_prompt`` 作为主信号。两者皆空时返回空串（上层据此静默）。
    """
    first_prompt = (first_prompt or "").strip()
    if first_prompt:
        return first_prompt
    return _query_from_cwd(cwd or "")


# ═══════════════════════════════════════════════
#  简报渲染 + token 预算控制
# ═══════════════════════════════════════════════

_HEADER = "📖 Storybook 晨间简报：以下过往记忆可能与当前任务相关（按相似度排序）"
_FOOTER = "（来自本地 Storybook 记忆库；调用 recall / get_story 可查看详情。不相关可忽略。）"


def _excerpt(content: str, max_chars: int) -> str:
    """截取 ``content`` 的前 ``max_chars`` 字符，超长则截断加省略号。"""
    content = (content or "").strip()
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content
    return content[: max(0, max_chars - 1)].rstrip() + "…"


def _format_entry(match: dict, excerpt: str) -> str:
    """渲染单条记忆为简报条目：``• [相似度%] 标题`` + 缩进摘要。"""
    sim = match.get("similarity", 0.0)
    pct = f"{int(round(sim * 100))}%"
    line = f"• [{pct}] {match.get('title', '')}"
    if excerpt:
        # 摘要可能含换行，逐行缩进对齐
        indented = "\n  ".join(excerpt.splitlines())
        line += f"\n  {indented}"
    return line


def _shrink_to_fit(match: dict, budget: int) -> tuple[str | None, bool]:
    """把单条记忆的摘要压缩到其整条条目 ≤ ``budget`` token。

    返回 ``(excerpt_or_None, shrunk)``。``None`` 表示连"标题行"都放不下（整条丢弃）。
    用"砍半摘要 + 重估"循环保证终止且最终 fit（启发式不精确，需校验兜底）。
    """
    overhead = estimate_tokens(_format_entry(match, ""))  # 仅标题行的开销
    room_tokens = budget - overhead
    if room_tokens <= 0:
        return None, True
    # 保守换算：1 token ≈ 2 字符（混合 CJK 偏保守，宁短勿超）
    room_chars = int(room_tokens * 2)
    content = (match.get("content") or "").strip()
    excerpt = _excerpt(content, room_chars) if room_chars > 0 else ""
    # 启发式不精确，校验后必要时继续砍半直至 fit
    while estimate_tokens(_format_entry(match, excerpt)) > budget:
        if not excerpt:
            return None, True
        half = max(0, len(excerpt) // 2)
        excerpt = _excerpt(content, half)
        if half == 0:
            return None, True
    return excerpt, True


def _build_briefing(candidates: list[dict], token_budget: int
                    ) -> tuple[str, list[dict], bool]:
    """把候选 story（已按相似度降序）渲染为 token 预算内的简报。

    返回 ``(briefing, matches, truncated)``：
      - ``briefing``：可注入的纯文本（无候选能塞下时为空串）。
      - ``matches``：最终纳入简报的条目（含 ``story_id/title/similarity/excerpt/keywords``）。
      - ``truncated``：是否有候选因预算被丢弃或摘要被压缩。
    """
    # 预留 header / footer 与结构开销余量
    reserved = estimate_tokens(_HEADER) + estimate_tokens(_FOOTER) + 60
    budget = max(0, token_budget - reserved)

    # 1. 每条按固定摘要上限渲染
    entries = [
        {"match": m, "excerpt": _excerpt(m.get("content", ""),
                                        config.PRIME_CONTENT_EXCERPT_CHARS),
         "shrunk": False}
        for m in candidates
    ]

    def total_cost(ents: list[dict]) -> int:
        return sum(estimate_tokens(_format_entry(e["match"], e["excerpt"])) for e in ents)

    # 2. 总额超预算时，从相似度最低（末尾）开始丢弃；只剩一条时压缩其摘要
    truncated = False
    while entries and total_cost(entries) > budget:
        if len(entries) == 1:
            e = entries[0]
            e["excerpt"], e["shrunk"] = _shrink_to_fit(e["match"], budget)
            if e["excerpt"] is None:
                entries.pop()
            truncated = True
            break
        entries.pop()  # 丢弃相似度最低的一条
        truncated = True

    if not entries:
        return "", [], truncated

    lines = [_HEADER, ""]
    matches = []
    for e in entries:
        lines.append(_format_entry(e["match"], e["excerpt"]))
        m = e["match"]
        matches.append({
            "story_id": m["story_id"],
            "title": m.get("title", ""),
            "similarity": m["similarity"],
            "excerpt": e["excerpt"],
            "keywords": m.get("keywords", []),
        })
        if e["shrunk"]:
            truncated = True
    lines += ["", _FOOTER]
    return "\n".join(lines), matches, truncated


# ═══════════════════════════════════════════════
#  主动注入核心
# ═══════════════════════════════════════════════

def prime_context(
    cwd: str = "",
    first_prompt: str = "",
    top_k: int = None,
    token_budget: int = None,
    *,
    tool_type: str = "other",
    integration_mode: str = "manual",
) -> dict:
    """主动召回并生成"晨间简报"。

    基于 ``cwd`` + 可选 ``first_prompt`` 构造 query，复用 ``search.search`` 召回 top-N
    story，按 ``PRIME_MIN_SIMILARITY`` 过滤后渲染为 ≤ ``token_budget`` token 的精简摘要。

    返回::

        {
          "cwd": str,            # 传入（或回退）的项目目录
          "query": str,          # 实际用于召回的 query
          "count": int,          # 纳入简报的记忆数
          "injected": bool,      # 是否产出了可注入简报（count>0）
          "briefing": str,       # 简报纯文本（injected=False 时为 ""）
          "matches": [           # 纳入简报的记忆（结构化）
            {story_id, title, similarity, excerpt, keywords}, ...
          ],
          "truncated": bool,     # 是否因 token 预算丢弃/压缩了候选
          "note": str | None,    # 静默原因（无 query / 召回失败等），便于排查
        }

    **静默不注入**（``injected=False``、``briefing=""``、不抛错）的情况：
      - 无可用 query（cwd 与 first_prompt 均为空）
      - 召回为空或全部低于 ``PRIME_MIN_SIMILARITY``（相关度不足，避免噪声）
      - embedding API 不可用--``note`` 给出排查提示，但不报错

    复用 ``search.search`` 的全部语义与副作用（``access_count`` 自增、共同召回边权提权）。
    """
    token_budget = token_budget if token_budget is not None else config.PRIME_TOKEN_BUDGET
    top_k = top_k or config.PRIME_TOP_K

    query = build_query(cwd, first_prompt)
    current_context = context_module.capture_context(
        tool_type=tool_type,
        integration_mode=integration_mode,
        workspace_path=cwd or None,
    )
    base = {
        "cwd": cwd,
        "query": query,
        "count": 0,
        "injected": False,
        "briefing": "",
        "matches": [],
        "truncated": False,
        "note": None,
        "context": current_context,
    }

    if not query:
        base["note"] = "无可用查询（cwd / first_prompt 均为空），跳过注入。"
        return base

    try:
        result = search_module.search(
            query,
            top_k=top_k,
            context=current_context,
            scope="profile",
        )
    except Exception as e:  # noqa: BLE001  -- schema 未初始化 / DB 锁等：晨间简报须非侵入
        base["note"] = (
            f"召回异常：{e}。可能数据库未初始化，请运行 `book init`；"
            f"或运行 `book doctor` 排查。"
        )
        return base

    if result.get("error"):
        # embedding 不可用等环境问题：晨间简报须非侵入，静默不注入并记 note 供排查
        base["note"] = (
            f"召回失败：{result['error']}。请确认 embedding API 与模型"
            f"（{config.EMBED_MODEL}）可用，可运行 `book doctor` 排查。"
        )
        return base

    if result.get("degraded") and not result.get("top_matches"):
        reason = result.get("degraded_reason") or "embedding_unavailable"
        base["note"] = (
            f"召回已降级（{reason}），关键词 fallback 未命中；"
            f"这不等同于确认无相关记忆。可运行 `book doctor` 排查。"
        )
        return base

    # 主动注入用更高相关度门槛，避免弱相关记忆污染每次会话开头
    candidates = [
        m for m in result.get("top_matches", [])
        if m["similarity"] >= config.PRIME_MIN_SIMILARITY
    ]
    if not candidates:
        # 相关度不足：静默不注入，无 note（这是正常情况，非异常）
        if result.get("degraded"):
            base["note"] = (
                "召回使用关键词降级，但候选未达到主动注入门槛；未注入上下文。"
            )
        return base

    # search.search 已按相似度降序返回，显式保证一次
    candidates.sort(key=lambda m: m["similarity"], reverse=True)

    briefing, matches, truncated = _build_briefing(candidates, token_budget)
    if not matches:
        # 候选都因 token 预算被丢弃（极端小预算）
        base["truncated"] = truncated
        base["note"] = "候选记忆因 token 预算过小全部被裁剪，未注入。"
        return base

    base.update({
        "count": len(matches),
        "injected": True,
        "briefing": briefing,
        "matches": matches,
        "truncated": truncated,
    })
    return base
