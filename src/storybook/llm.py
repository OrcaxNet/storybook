"""
LLM 处理层 — 封装 Ollama LLM API
提供：摘要生成、关键词提取、分裂判断、Story 拆分、合并
"""
import json
import logging
from typing import Optional

import requests

from . import config

logger = logging.getLogger(__name__)


def _chat(prompt: str, system: str = "") -> Optional[str]:
    """调用 Ollama chat API，返回纯文本响应"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "think": config.LLM_THINK,
                "options": {"temperature": 0.3, "num_ctx": 8192},
            },
            timeout=120,
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


def summarize_session(session_content: str) -> dict:
    """将会话浓缩为 ≤400字 的标准 story，返回 {title, content}"""
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
        return {"title": "未命名记忆", "content": session_content[:400]}

    title = "未命名记忆"
    content = result

    # 解析 TITLE 和 CONTENT
    if "TITLE:" in result:
        parts = result.split("CONTENT:", 1)
        title_part = parts[0].replace("TITLE:", "").strip()
        title = title_part.split("\n")[0].strip()[:50]
        if len(parts) > 1:
            content = parts[1].strip()

    # 截断到 400 字
    if len(content) > config.STORY_MAX_CHARS:
        content = content[:config.STORY_MAX_CHARS - 3] + "..."

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
        return {"title": "合并记忆", "content": (old_content + "\n" + new_content)[:400]}

    title = "合并记忆"
    content = result
    if "TITLE:" in result:
        parts = result.split("CONTENT:", 1)
        title = parts[0].replace("TITLE:", "").strip().split("\n")[0].strip()[:50]
        if len(parts) > 1:
            content = parts[1].strip()

    if len(content) > config.STORY_MAX_CHARS:
        content = content[:config.STORY_MAX_CHARS - 3] + "..."

    return {"title": title, "content": content}


def judge_split(merged_text: str) -> bool:
    """判断合并后的 story 是否需要分裂"""
    if len(merged_text) > config.STORY_MAX_CHARS:
        return True

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
1. 每个子记忆不超过400字
2. 每个子记忆聚焦一个独立可复用的子问题
3. 保持"问题-步骤-结果"格式
4. 通常拆分为2-3个子记忆

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
        return [{"title": "拆分记忆", "content": merged_text[:400]}]

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

        if len(content) > config.STORY_MAX_CHARS:
            content = content[:config.STORY_MAX_CHARS - 3] + "..."

        sub_stories.append({"title": title, "content": content})

    return sub_stories if sub_stories else [{"title": "拆分记忆", "content": merged_text[:400]}]


def extract_problem_summary(raw_content: str) -> str:
    """从原始会话中提取问题摘要（≤100字）"""
    prompt = f"""请用一句话（不超过100字）概括以下编程会话中用户要解决的核心问题：

{raw_content[:3000]}

只输出问题摘要，不要其他内容。"""

    result = _chat(prompt)
    return result[:100] if result else ""
