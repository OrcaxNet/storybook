"""Story v2 normalization, rendering, hashing, and embedding representations.

The storage layer deliberately keeps the legacy ``content`` field.  For v2
Stories it is a lossless, human-readable rendering of ``detail``; older MCP
clients can therefore keep reading ``title``/``content`` while newer clients
consume the structured fields.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import config
from . import context as context_module


DETAIL_FIELDS = (
    "problem",
    "actions",
    "outcome",
    "pitfalls",
    "evidence",
    "applicability",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return result


def normalize_sources(
    sources: Any = None, source_session_ids: list[int] | None = None
) -> list[dict]:
    """Return stable source descriptors without inventing evidence.

    A source can carry an optional evidence locator supplied by the formation
    model.  Session ids remain local database references; cross-device identity
    is provided by the Session's ``global_id`` when details are expanded.
    """

    result: list[dict] = []
    seen: set[tuple] = set()
    raw_sources = sources if isinstance(sources, list) else []
    for raw in raw_sources:
        if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
            item = {"session_id": int(raw), "evidence": []}
        elif isinstance(raw, dict):
            session_id = raw.get("session_id")
            if isinstance(session_id, str) and session_id.isdigit():
                session_id = int(session_id)
            item = {
                "session_id": session_id if isinstance(session_id, int) else None,
                "session_global_id": _text(raw.get("session_global_id")) or None,
                "evidence": _string_list(
                    raw.get("evidence", raw.get("evidence_refs"))
                ),
            }
        else:
            continue
        key = (
            item.get("session_id"),
            item.get("session_global_id"),
            tuple(item.get("evidence", [])),
        )
        if key not in seen:
            seen.add(key)
            result.append(item)

    normalized_session_ids = [
        int(value) for value in (source_session_ids or [])
        if isinstance(value, int) or str(value).isdigit()
    ]
    if len(normalized_session_ids) == 1:
        for item in result:
            if item.get("session_id") is None and not item.get("session_global_id"):
                item["session_id"] = normalized_session_ids[0]

    represented = {item.get("session_id") for item in result}
    for session_id in normalized_session_ids:
        if not isinstance(session_id, int) and not str(session_id).isdigit():
            continue
        if session_id not in represented:
            result.append({"session_id": session_id, "evidence": []})
            represented.add(session_id)
    return result


def normalize_detail(
    detail: Any,
    *,
    legacy_content: str = "",
    applicability: dict | str | None = None,
) -> dict:
    raw = detail if isinstance(detail, dict) else {}
    normalized_applicability = context_module.normalize_applicability(
        raw.get("applicability", applicability)
    )
    problem = _text(raw.get("problem")) or legacy_content.strip()
    return {
        "problem": problem,
        "actions": _string_list(raw.get("actions")),
        "outcome": _text(raw.get("outcome")),
        "pitfalls": _string_list(raw.get("pitfalls")),
        "evidence": _string_list(raw.get("evidence")),
        "applicability": normalized_applicability,
    }


def render_detail(detail: dict) -> str:
    """Render every structured field without truncating persisted evidence."""

    lines: list[str] = []
    if detail.get("problem"):
        lines.append(f"问题：{detail['problem']}")
    if detail.get("actions"):
        lines.append("行动：")
        lines.extend(
            f"{index}. {action}"
            for index, action in enumerate(detail["actions"], start=1)
        )
    if detail.get("outcome"):
        lines.append(f"结果：{detail['outcome']}")
    if detail.get("pitfalls"):
        lines.append("注意：" + "；".join(detail["pitfalls"]))
    if detail.get("evidence"):
        lines.append("证据：" + "；".join(detail["evidence"]))
    applicability = context_module.normalize_applicability(
        detail.get("applicability")
    )
    if applicability.get("applies_when"):
        lines.append(
            "适用于：" + json.dumps(
                applicability["applies_when"], ensure_ascii=False, sort_keys=True
            )
        )
    if applicability.get("excludes_when"):
        lines.append(
            "不适用于：" + json.dumps(
                applicability["excludes_when"], ensure_ascii=False, sort_keys=True
            )
        )
    return "\n".join(lines)


def _abstract_fallback(detail: dict, content: str) -> str:
    parts = [detail.get("problem", ""), detail.get("outcome", "")]
    value = "；".join(part.strip() for part in parts if part and part.strip())
    return value or content.strip()


def bound_abstract(value: str) -> tuple[str, bool]:
    """Apply the intentional summary budget, never the detail persistence path."""

    value = value.strip()
    if len(value) <= config.STORY_ABSTRACT_MAX_CHARS:
        return value, False
    return value[: config.STORY_ABSTRACT_MAX_CHARS].rstrip(), True


def normalize_story_payload(
    payload: dict | None,
    *,
    fallback_content: str = "",
    source_session_ids: list[int] | None = None,
) -> dict:
    raw = payload if isinstance(payload, dict) else {}
    explicit_v2 = any(
        raw.get(key) is not None for key in ("abstract", "detail", "sources")
    )
    title = _text(raw.get("title")) or "未命名记忆"
    legacy_content = _text(raw.get("content")) or fallback_content.strip()
    applicability = context_module.normalize_applicability(
        raw.get("applicability")
        or (raw.get("detail") or {}).get("applicability")
        if isinstance(raw.get("detail"), dict)
        else raw.get("applicability")
    )
    detail = normalize_detail(
        raw.get("detail"),
        legacy_content=legacy_content,
        applicability=applicability,
    )
    content = (
        render_detail(detail)
        if isinstance(raw.get("detail"), dict) and raw.get("detail")
        else legacy_content
    ) or render_detail(detail)
    abstract, abstract_truncated = bound_abstract(
        _text(raw.get("abstract")) or _abstract_fallback(detail, content)
    )
    sources = normalize_sources(raw.get("sources"), source_session_ids)
    keywords = _string_list(raw.get("keywords"))
    return {
        "title": title,
        "abstract": abstract,
        "abstract_truncated": abstract_truncated,
        "detail": detail,
        "content": content,
        "sources": sources,
        "source_session_ids": list(dict.fromkeys(source_session_ids or [])),
        "applicability": detail["applicability"],
        "keywords": keywords,
        "_v2_explicit": explicit_v2,
    }


def embedding_fields(story: dict) -> dict[str, str]:
    applicability = context_module.normalize_applicability(
        story.get("applicability")
        or (story.get("detail") or {}).get("applicability")
    )
    return {
        "title": _text(story.get("title")),
        "abstract": _text(story.get("abstract")),
        "applicability": json.dumps(
            applicability, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "full": _text(story.get("content"))
        or render_detail(normalize_detail(story.get("detail"))),
    }


def embedding_input(story: dict, representation: str | None = None) -> str:
    """Build one documented embedding representation.

    ``default`` is deliberately compact and stable: title + abstract +
    applicability.  Other modes exist for the isolated ablation runner.
    """

    representation = representation or config.EMBED_REPRESENTATION
    fields = embedding_fields(story)
    if representation == "legacy":
        keywords = " ".join(_string_list(story.get("keywords")))
        return " ".join(part for part in (keywords, fields["full"]) if part)
    if representation == "full":
        return "\n".join(
            part for part in (
                fields["title"], fields["abstract"], fields["full"],
                fields["applicability"],
            ) if part
        )
    return "\n".join(
        part for part in (
            fields["title"], fields["abstract"], fields["applicability"]
        ) if part
    )


def content_hash(story: dict, representation: str | None = None) -> str:
    material = {
        "representation": representation or config.EMBED_REPRESENTATION,
        "text": embedding_input(story, representation),
    }
    encoded = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recall_summary(story: dict) -> tuple[str, bool]:
    """Return the default recall text and whether presentation clipped it."""

    abstract = _text(story.get("abstract"))
    if abstract:
        return abstract, bool(
            story.get("abstract_truncated", False)
            or (story.get("content") and abstract != _text(story.get("content")))
        )
    content = _text(story.get("content"))
    if len(content) <= config.RECALL_SUMMARY_MAX_CHARS:
        return content, False
    return content[: config.RECALL_SUMMARY_MAX_CHARS].rstrip(), True
