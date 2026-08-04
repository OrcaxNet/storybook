"""
LLM 处理层 — 封装 DeepSeek Anthropic-compatible Messages API
提供：摘要生成、关键词提取、分裂判断、Story 拆分、合并
"""
import json
import logging
from typing import Any, Optional

import requests

from . import config
from . import inference_cache
from . import story_v2

logger = logging.getLogger(__name__)


def _chat(
    prompt: str,
    system: str = "",
    *,
    timeout_seconds: float = 120,
    num_predict: int | None = None,
    response_schema: dict | None = None,
) -> Optional[str | dict]:
    """Call DeepSeek's Anthropic-compatible API.

    When ``response_schema`` is provided, the request forces one tool call and
    returns its already-decoded ``input`` object.  DeepSeek's Anthropic
    compatibility layer supports ``input_schema`` and named ``tool_choice``;
    plain text remains accepted only as a compatibility fallback for older
    gateways and deterministic test doubles.
    """

    cache_payload = {
        "provider": config.LLM_PROVIDER,
        "base_url": config.LLM_BASE_URL.rstrip("/"),
        "model": config.LLM_MODEL,
        "thinking": config.LLM_THINK,
        "prompt": prompt,
        "system": system,
        "num_predict": num_predict,
        "response_schema": response_schema,
    }
    cached = inference_cache.get("llm-v1", cache_payload)
    if isinstance(cached, (str, dict)):
        return cached

    if config.LLM_PROVIDER != "ollama" and not config.LLM_API_KEY:
        logger.error(
            "LLM request failed provider=%s category=credentials_missing",
            config.LLM_PROVIDER,
        )
        return None

    max_tokens = 4096 if num_predict is None else max(32, int(num_predict))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if config.LLM_PROVIDER == "ollama":
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }
        if response_schema is not None:
            payload["format"] = response_schema
        url = f"{config.LLM_BASE_URL.rstrip('/')}/api/chat"
        headers = {}
    elif config.LLM_PROVIDER == "api":
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "storybook_output", "schema": response_schema},
            }
        url = f"{config.LLM_BASE_URL.rstrip('/')}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    else:
        payload = {
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "thinking": {"type": "enabled" if config.LLM_THINK else "disabled"},
        }
        if system:
            payload["system"] = system
        if response_schema is not None:
            payload["tools"] = [{
                "name": "submit_structured_output",
                "description": "Return the requested structured result.",
                "input_schema": response_schema,
            }]
            payload["tool_choice"] = {
                "type": "tool", "name": "submit_structured_output",
            }
        url = f"{config.LLM_BASE_URL.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": config.LLM_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    try:
        resp = requests.post(url, headers=headers, json=payload,
                             timeout=max(0.1, float(timeout_seconds)))
        resp.raise_for_status()
        data = resp.json()
        if config.LLM_PROVIDER in {"ollama", "api"}:
            if config.LLM_PROVIDER == "ollama":
                message = data.get("message", {}) if isinstance(data, dict) else {}
            else:
                choices = data.get("choices", []) if isinstance(data, dict) else []
                message = choices[0].get("message", {}) if choices else {}
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty content")
            result = _decode_json_object(text) if response_schema is not None else text.strip()
            if response_schema is not None and not _matches_schema(result, response_schema):
                raise ValueError("invalid structured output")
            inference_cache.set("llm-v1", cache_payload, result)
            return result
        blocks = data.get("content") if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            raise ValueError("invalid content")
        if response_schema is not None:
            for block in blocks:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == "submit_structured_output"
                    and _matches_schema(block.get("input"), response_schema)
                ):
                    inference_cache.set("llm-v1", cache_payload, block["input"])
                    return block["input"]
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()
        if not text:
            raise ValueError("empty content")
        inference_cache.set("llm-v1", cache_payload, text)
        return text
    except requests.exceptions.Timeout:
        category, status = "timeout", None
    except requests.exceptions.HTTPError as exc:
        category = "http_error"
        status = exc.response.status_code if exc.response is not None else None
    except (ValueError, TypeError, json.JSONDecodeError):
        category, status = "invalid_response", None
    except requests.exceptions.RequestException:
        category, status = "network_error", None
    except Exception:  # noqa: BLE001 -- provider failures preserve fallback semantics
        category, status = "unexpected_error", None
    if status is None:
        logger.error(
            "LLM request failed provider=%s category=%s",
            config.LLM_PROVIDER,
            category,
        )
    else:
        logger.error(
            "LLM request failed provider=%s category=%s status=%s",
            config.LLM_PROVIDER,
            category,
            status,
        )
    return None


def _matches_schema(value: Any, schema: dict) -> bool:
    """Validate the JSON-schema subset used by Storybook tool outputs."""

    if "enum" in schema and value not in schema["enum"]:
        return False
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        return all(
            key not in value or _matches_schema(value[key], child)
            for key, child in properties.items()
        )
    if expected == "array":
        return isinstance(value, list) and all(
            _matches_schema(item, schema.get("items", {})) for item in value
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _decode_json_object(text: str) -> dict | None:
    """Decode one JSON object without marker slicing.

    ``raw_decode`` permits harmless leading/trailing prose from a legacy
    gateway while still requiring the decoded value itself to be an object.
    """

    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            decoded, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def _structured_call(
    prompt: str,
    schema: dict,
    *,
    system: str = "",
    timeout_seconds: float = 120,
    num_predict: int | None = None,
) -> dict | None:
    """Return a schema-valid object from tool input or legacy JSON text."""

    result = _chat(
        prompt,
        system=system,
        timeout_seconds=timeout_seconds,
        num_predict=num_predict,
        response_schema=schema,
    )
    decoded = result if isinstance(result, dict) else _decode_json_object(result)
    if decoded is not None and _matches_schema(decoded, schema):
        return decoded
    logger.warning(
        "LLM structured output invalid provider=%s",
        config.LLM_PROVIDER,
    )
    return None


def _object_schema(properties: dict[str, dict]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_KEYWORDS_SCHEMA = _object_schema({
    "keywords": {"type": "array", "items": {"type": "string"}},
})
_MEMORY_SCHEMA = _object_schema({
    "title": {"type": "string"},
    "content": {"type": "string"},
})
_SPLIT_DECISION_SCHEMA = _object_schema({
    "should_split": {"type": "boolean"},
})
_SPLIT_STORIES_SCHEMA = _object_schema({
    "stories": {"type": "array", "items": _MEMORY_SCHEMA},
})
_QUERY_TRANSFORM_SCHEMA = _object_schema({
    "rewrite": {"type": "string"},
    "queries": {"type": "array", "items": {"type": "string"}},
    "hypothetical_document": {"type": "string"},
})
_STORY_SCHEMA = _object_schema({
    "title": {"type": "string"},
    "abstract": {"type": "string"},
    "detail": _object_schema({
        "problem": {"type": "string"},
        "actions": {"type": "array", "items": {"type": "string"}},
        "outcome": {"type": "string"},
        "pitfalls": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "applicability": _object_schema({
            "applies_when": {"type": "array", "items": {"type": "string"}},
            "excludes_when": {"type": "array", "items": {"type": "string"}},
        }),
    }),
    "sources": {
        "type": "array",
        "items": _object_schema({
            "evidence": {"type": "array", "items": {"type": "string"}},
        }),
    },
    "keywords": {"type": "array", "items": {"type": "string"}},
})
_STORIES_SCHEMA = _object_schema({
    "stories": {"type": "array", "items": _STORY_SCHEMA},
})


def _generate(prompt: str) -> Optional[str]:
    """Compatibility wrapper for callers that historically used generate."""
    return _chat(prompt)


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
    decoded = _structured_call(
        prompt,
        _QUERY_TRANSFORM_SCHEMA,
        system="你负责生成可审计、最小化的检索辅助表示。",
        timeout_seconds=timeout_seconds,
        num_predict=384,
    )
    if decoded is None:
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

以 JSON 对象输出，如：{{"keywords": ["React", "useEffect", "无限循环"]}}。"""

    result = _structured_call(prompt, _KEYWORDS_SCHEMA)
    if result is None:
        return []
    return [keyword.strip() for keyword in result["keywords"] if keyword.strip()][:10]


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

输出 JSON 对象，顶层字段为 ``stories`` 数组。每项格式：
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
    result = _structured_call(prompt, _STORIES_SCHEMA)
    if result is not None:
        stories = [
            story_v2.normalize_story_payload(item)
            for item in result["stories"]
        ]
        if stories:
            return stories
    logger.warning("Story v2 formation 结构化输出失败，使用无损 fallback")

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

输出 JSON 对象：
{{"title": "简短标题，10-20字", "content": "问题-步骤-结果格式的记忆文本"}}"""

    result = _structured_call(prompt, _MEMORY_SCHEMA)
    if result is None:
        return {"title": "未命名记忆", "content": session_content}
    return {
        "title": result["title"].strip()[:50] or "未命名记忆",
        "content": result["content"].strip() or session_content,
    }


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

输出 JSON 对象：
{{"title": "简短标题，10-20字", "content": "合并后的记忆文本"}}"""

    result = _structured_call(prompt, _MEMORY_SCHEMA)
    if result is None:
        return {"title": "合并记忆", "content": old_content + "\n" + new_content}
    return {
        "title": result["title"].strip()[:50] or "合并记忆",
        "content": result["content"].strip() or old_content + "\n" + new_content,
    }


def judge_split(merged_text: str) -> bool:
    """按独立结论/适用性判断是否分裂，不使用字符阈值。"""

    prompt = f"""判断以下技术记忆是否包含两个或以上独立可复用的子步骤。

判断标准：
- "配置ESLint规则" 和 "修复useEffect依赖" = 2个独立子步骤
- "检查依赖数组" 和 "使用useCallback" = 同一问题的连续步骤

记忆内容：
{merged_text[:1000]}

输出 JSON 对象：{{"should_split": true 或 false}}。"""

    result = _structured_call(prompt, _SPLIT_DECISION_SCHEMA)
    if result is None:
        return False
    return result["should_split"]


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

输出 JSON 对象，格式：
{{"stories": [{{"title": "标题", "content": "内容"}}]}}"""

    result = _structured_call(prompt, _SPLIT_STORIES_SCHEMA)
    if result is None:
        return [{"title": "拆分记忆", "content": merged_text}]
    sub_stories = []
    for index, story in enumerate(result["stories"], start=1):
        content = story["content"].strip()
        title = story["title"].strip()[:50]
        if not title:
            title = content[:24] or f"子记忆 {index}"
        sub_stories.append({"title": title, "content": content})
    return sub_stories or [{"title": "拆分记忆", "content": merged_text}]


def extract_problem_summary(raw_content: str) -> str:
    """从原始会话中提取问题摘要（≤100字）"""
    prompt = f"""请用一句话（不超过100字）概括以下编程会话中用户要解决的核心问题：

{raw_content[:3000]}

只输出问题摘要，不要其他内容。"""

    result = _chat(prompt)
    return result[:100] if result else ""
