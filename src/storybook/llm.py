"""
LLM 处理层 — 封装 Ollama LLM API
提供：摘要生成、关键词提取、分裂判断、Story 拆分、合并
"""
import json
import logging
from typing import Optional

import requests

from . import config
from . import story_v2

logger = logging.getLogger(__name__)


def _chat(
    prompt: str,
    system: str = "",
    *,
    timeout_seconds: float = 120,
    num_predict: int | None = None,
) -> Optional[str]:
    """调用 Ollama chat API，返回纯文本响应"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        options = {"temperature": 0.3, "num_ctx": 8192}
        if num_predict is not None:
            options["num_predict"] = max(32, int(num_predict))
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "think": config.LLM_THINK,
                "options": options,
            },
            timeout=max(0.1, float(timeout_seconds)),
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error("LLM 调用失败: %s", e)
        return None


def _generate(prompt: str) -> Optional[str]:
    """调用 Ollama generate API（更轻量）"""
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": config.LLM_THINK,
                "options": {"temperature": 0.3, "num_ctx": 8192},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        logger.error("LLM generate 失败: %s", e)
        return None


# ═══════════════════════════════════════════════
#  核心 LLM 操作
# ═══════════════════════════════════════════════

def transform_search_query(
    query: str,
    transformations: list[str],
    *,
    timeout_seconds: float,
) -> dict | None:
    """Generate only the explicitly gated search transformations in one call.

    The caller owns the outer hard deadline.  This function also passes the
    remaining budget to ``requests`` so a timed-out worker does not retain the
    local Ollama connection for the historical 120-second formation timeout.
    """

    allowed = [
        item for item in dict.fromkeys(transformations)
        if item in {"rewrite", "multi_query", "hyde"}
    ]
    if not allowed:
        return None
    prompt = f"""你是本地记忆检索查询规划器。只生成请求的检索辅助表示，不回答问题。

原始查询：{query}
启用策略：{', '.join(allowed)}

输出严格 JSON 对象，字段如下：
{{
  "rewrite": "语义不丢失、适合检索的单一改写；未启用 rewrite 时为空字符串",
  "queries": ["最多 {max(1, config.QUERY_MULTI_QUERY_LIMIT)} 条互补检索查询；未启用 multi_query 时为空数组"],
  "hypothetical_document": "一段可能命中相关经历的简短假设性记忆摘要；未启用 hyde 时为空字符串"
}}

约束：
- 保留技术名、错误文本、版本和环境条件，不虚构路径、主机名或凭据。
- multi-query 各自覆盖原问题的不同子意图，不重复原句。
- HyDE 只描述可能的“问题—行动—结果”记忆形态，不声称它真实发生。
- 只输出 JSON，不输出 Markdown。
"""
    result = _chat(
        prompt,
        system="你负责生成可审计、最小化的检索辅助表示。",
        timeout_seconds=timeout_seconds,
        num_predict=384,
    )
    if not result:
        return None
    try:
        start = result.find("{")
        end = result.rfind("}")
        if start < 0 or end <= start:
            return None
        decoded = json.loads(result[start:end + 1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None

    rewrite = _bounded_search_text(decoded.get("rewrite")) if "rewrite" in allowed else ""
    hypothetical = (
        _bounded_search_text(decoded.get("hypothetical_document"), limit=1200)
        if "hyde" in allowed else ""
    )
    queries = []
    if "multi_query" in allowed and isinstance(decoded.get("queries"), list):
        for value in decoded["queries"]:
            candidate = _bounded_search_text(value)
            if candidate and candidate != query.strip() and candidate not in queries:
                queries.append(candidate)
            if len(queries) >= max(1, config.QUERY_MULTI_QUERY_LIMIT):
                break
    return {
        "rewrite": rewrite,
        "queries": queries,
        "hypothetical_document": hypothetical,
    }


def _bounded_search_text(value, *, limit: int = 600) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()[:max(1, limit)]

def extract_keywords(text: str) -> list[str]:
    """从技术文本提取 5-10 个核心关键词"""
    prompt = f"""从以下技术文本中提取5-10个核心技术关键词。
要求：
1. 包含技术栈名（如React、Python）
2. 包含问题类型（如内存泄漏、类型错误）
3. 包含关键解决方案术语
4. 中英文混合，保持原始语言

文本：
{text[:2000]}

以JSON数组格式输出，如：["React", "useEffect", "无限循环"]
只输出JSON数组，不要其他内容。"""

    result = _chat(prompt)
    if not result:
        return []

    # 尝试解析 JSON
    try:
        # 找到 JSON 数组部分
        start = result.find("[")
        end = result.rfind("]")
        if start != -1 and end != -1:
            keywords = json.loads(result[start:end + 1])
            return [k.strip() for k in keywords if isinstance(k, str)][:10]
    except (json.JSONDecodeError, ValueError):
        pass

    # fallback: 按逗号/换行分割
    parts = result.replace("[", "").replace("]", "").replace('"', "").split(",")
    return [p.strip() for p in parts if p.strip()][:10]


def form_stories(session_content: str) -> list[dict]:
    """Form one or more independently reusable Story v2 candidates.

    Boundaries follow conclusions and applicability, never a fixed character
    count.  A long atomic experience remains one Story; a short session with
    two independent outcomes becomes two Stories sharing the same Session.
    """

    prompt = f"""你是 Agent 经历记忆整理器。把下面会话切分为一个或多个可独立复用的 Story。

切分原则：
- 每个 Story 只承载一个可独立复用的结论及其适用环境。
- 同一问题的连续排查、动作和结果必须保持在同一个 Story，即使内容很长。
- 两个问题有独立结果或不同适用条件时必须拆成两个 Story，即使会话很短。
- 不按字符数硬切分，不丢失失败尝试、证据或环境边界。
- abstract 是有预算的检索摘要；detail 必须保留完整问题、动作、结果与教训。

只输出 JSON 数组。每项格式：
{{
  "title": "简短标题",
  "abstract": "关键结论与适用条件摘要",
  "detail": {{
    "problem": "完整问题背景",
    "actions": ["按顺序的动作"],
    "outcome": "结果",
    "pitfalls": ["失败教训或空数组"],
    "evidence": ["可在原会话定位的证据描述"],
    "applicability": {{"applies_when": [], "excludes_when": []}}
  }},
  "sources": [{{"evidence": ["原会话中的证据描述"]}}],
  "keywords": ["检索关键词"]
}}

会话内容：
{session_content}
"""
    result = _chat(prompt)
    if result:
        try:
            start = result.find("[")
            end = result.rfind("]")
            if start >= 0 and end > start:
                decoded = json.loads(result[start:end + 1])
                if isinstance(decoded, list):
                    stories = [
                        story_v2.normalize_story_payload(item)
                        for item in decoded if isinstance(item, dict)
                    ]
                    if stories:
                        return stories
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Story v2 formation JSON 解析失败，使用无损 fallback")

    # Local model failures must not turn persistence into destructive clipping.
    return [story_v2.normalize_story_payload({
        "title": "未命名记忆",
        "abstract": session_content,
        "detail": {
            "problem": session_content,
            "actions": [],
            "outcome": "",
            "pitfalls": [],
            "evidence": ["原始 Session 全文"],
            "applicability": {"applies_when": [], "excludes_when": []},
        },
        "sources": [{"evidence": ["原始 Session 全文"]}],
    }, fallback_content=session_content)]


def summarize_session(session_content: str) -> dict:
    """Legacy one-Story formatter retained for older integrations."""
    prompt = f"""你是一个代码记忆管理专家。请将以下AI编程会话浓缩为不超过400字的结构化记忆。

要求：
1. 格式："问题：... 步骤：1.... 2.... 结果：..."
2. 保留核心技术细节和解决方案
3. 去除寒暄、重复、无效内容
4. 聚焦单个coding问题的完整解决逻辑
5. 总字数不超过400字

会话内容：
{session_content[:6000]}

请按以下格式输出（严格遵循）：
TITLE: <简短标题，10-20字>
CONTENT: <问题-步骤-结果格式的记忆文本>"""

    result = _chat(prompt)
    if not result:
        return {"title": "未命名记忆", "content": session_content}

    title = "未命名记忆"
    content = result

    # 解析 TITLE 和 CONTENT
    if "TITLE:" in result:
        parts = result.split("CONTENT:", 1)
        title_part = parts[0].replace("TITLE:", "").strip()
        title = title_part.split("\n")[0].strip()[:50]
        if len(parts) > 1:
            content = parts[1].strip()

    return {"title": title, "content": content}


def merge_stories(old_content: str, new_content: str) -> dict:
    """合并旧 story 和新会话内容，返回 {title, content}"""
    prompt = f"""请合并以下两段技术记忆，生成一段不超过400字的综合记忆。

要求：
1. 保留两段记忆的核心技术信息
2. 去除重复内容
3. 保持"问题-步骤-结果"格式
4. 如果两段记忆涉及不同问题，保留两个问题的解决逻辑
5. 总字数不超过400字

旧记忆：
{old_content}

新内容：
{new_content[:2000]}

请按以下格式输出（严格遵循）：
TITLE: <简短标题，10-20字>
CONTENT: <合并后的记忆文本>"""

    result = _chat(prompt)
    if not result:
        return {"title": "合并记忆", "content": old_content + "\n" + new_content}

    title = "合并记忆"
    content = result
    if "TITLE:" in result:
        parts = result.split("CONTENT:", 1)
        title = parts[0].replace("TITLE:", "").strip().split("\n")[0].strip()[:50]
        if len(parts) > 1:
            content = parts[1].strip()

    return {"title": title, "content": content}


def judge_split(merged_text: str) -> bool:
    """按独立结论/适用性判断是否分裂，不使用字符阈值。"""

    prompt = f"""判断以下技术记忆是否包含两个或以上独立可复用的子步骤。

判断标准：
- "配置ESLint规则" 和 "修复useEffect依赖" = 2个独立子步骤 → SPLIT:YES
- "检查依赖数组" 和 "使用useCallback" = 同一问题的连续步骤 → SPLIT:NO

记忆内容：
{merged_text[:1000]}

只输出 SPLIT:YES 或 SPLIT:NO，不要其他内容。"""

    result = _chat(prompt)
    if not result:
        return False
    return "SPLIT:YES" in result.upper()


def split_story(merged_text: str) -> list[dict]:
    """将过长/过复杂的 story 拆分为多个子 story"""
    prompt = f"""请将以下技术记忆拆分为多个独立的子记忆。

要求：
1. 每个子记忆聚焦一个独立可复用的结论与适用条件
2. 同一问题的完整证据与动作不可因长度拆散
3. 保持"问题-步骤-结果"格式
4. 仅在存在多个独立结论时拆分

记忆内容：
{merged_text[:2000]}

请按以下格式输出每个子记忆：
=== SUB-STORY 1 ===
TITLE: <标题>
CONTENT: <内容>
=== SUB-STORY 2 ===
TITLE: <标题>
CONTENT: <内容>"""

    result = _chat(prompt)
    if not result:
        return [{"title": "拆分记忆", "content": merged_text}]

    # 解析子 story
    sub_stories = []
    parts = result.split("=== SUB-STORY")
    for idx, part in enumerate(parts[1:], 1):  # 跳过第一个（可能是空或前导文本）
        lines = part.strip()
        title = "子记忆"
        content = lines

        if "TITLE:" in lines:
            t_parts = lines.split("CONTENT:", 1)
            title = t_parts[0].replace("TITLE:", "").strip().split("\n")[0].strip().lstrip("0123456789 ===").strip()[:50]
            if len(t_parts) > 1:
                content = t_parts[1].strip()

        # 标题兜底：LLM 偶尔输出空标题（如 "TITLE: \nCONTENT: ..."），用内容前缀补上
        if not title.strip():
            _fallback = content.strip()
            for _prefix in ("问题：", "问题:", "TITLE:"):
                if _fallback.startswith(_prefix):
                    _fallback = _fallback[len(_prefix):].strip()
                    break
            title = _fallback[:24] or f"子记忆 {idx}"

        sub_stories.append({"title": title, "content": content})

    return sub_stories if sub_stories else [{"title": "拆分记忆", "content": merged_text}]


def extract_problem_summary(raw_content: str) -> str:
    """从原始会话中提取问题摘要（≤100字）"""
    prompt = f"""请用一句话（不超过100字）概括以下编程会话中用户要解决的核心问题：

{raw_content[:3000]}

只输出问题摘要，不要其他内容。"""

    result = _chat(prompt)
    return result[:100] if result else ""
