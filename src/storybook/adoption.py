"""召回采纳判定：把 Agent 会话日志当作"业务埋点"，在做梦周期离线挖掘反馈。

Adoption feedback without interrupting the working agent.

The MCP server records every agent-facing recall as an exposure
(``recall_events`` / ``recall_exposures``).  When the dream cycle later
processes the Session in which that recall happened, it asks the generation
model whether the exposed Stories were actually *used*, *ignored* or
*misleading*.  Verdicts feed the bounded adoption prior used by ranking, the
co-adoption ``co_recall`` edges, and ``book feedback report``.

Matching a Session to its recalls uses the Session time window
(``session.started_at`` .. ``captured_at``) and, when both sides know it, the
workspace path fingerprint.  Sessions without a storybook tool marker are
skipped without any model call.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta

from . import llm, store

logger = logging.getLogger(__name__)

# Session transcript markers that a memory tool was called.
_RECALL_MARKER = re.compile(
    r"\[tool\][^\n]*\b(?:prime_context|recall|get_story)\b"
    r"|mcp__storybook__",
)
# Recall events may start slightly before the adapter's reported start.
_WINDOW_SLACK = timedelta(minutes=2)


def _parse_time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


def session_window(session) -> tuple[str, str, str | None] | None:
    """Return ``(start, end, workspace_hash)`` for a Session row, or None."""

    try:
        envelope = json.loads(session["context_json"] or "{}")
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    started = _parse_time((envelope.get("session") or {}).get("started_at"))
    captured = _parse_time(envelope.get("captured_at"))
    if started is None:
        return None
    end = captured if captured and captured >= started else started + timedelta(hours=12)
    workspace = (envelope.get("workspace") or {}).get("path_fingerprint")
    return (
        (started - _WINDOW_SLACK).isoformat(),
        (end + _WINDOW_SLACK).isoformat(),
        workspace,
    )


def has_recall_marker(raw_content: str) -> bool:
    return bool(_RECALL_MARKER.search(raw_content or ""))


def prepare_judgment(session) -> dict | None:
    """Judge recalls inside one Session (model call, no SQLite writes).

    Returns ``{"events": [...], "verdicts": {...}, "missing": str}`` or None
    when there is nothing to judge or the model output is unusable.
    """

    raw_content = session["raw_content"] or ""
    window = session_window(session)
    if window is None:
        return None
    start, end, workspace = window
    try:
        events = store.unjudged_recall_events(
            start=start, end=end, workspace_path_hash=workspace
        )
    except Exception:  # noqa: BLE001 -- 反馈是旁路，不能阻断做梦周期
        logger.warning("读取召回埋点失败，跳过采纳判定")
        return None
    if not has_recall_marker(raw_content):
        # SessionStart hook 注入不会在转录中留下工具调用；只有同一工作区的
        # hook 事件才可无标记匹配，避免把并行会话的召回张冠李戴。
        events = [
            e for e in events
            if e["origin"] == "hook_prime" and workspace
            and e["workspace_path_hash"] == workspace
        ]
    if not events:
        return None
    recalls = [
        {
            "query": event["query"],
            "origin": event["origin"],
            "stories": [
                {
                    "story_id": x["story_id"],
                    "title": x["title"],
                    "abstract": x["abstract"] or x["content"],
                }
                for x in event["exposures"]
            ],
        }
        for event in events
    ]
    try:
        judged = llm.judge_recall_adoption(raw_content, recalls)
    except Exception:  # noqa: BLE001
        logger.warning("采纳判定调用失败，跳过")
        return None
    if judged is None:
        return None
    return {"events": events, **judged}


def apply_judgment(session_id: int, judgment: dict | None) -> int:
    """Persist verdicts per event; returns the number of judged exposures."""

    if not judgment:
        return 0
    verdicts: dict[int, tuple[str, str]] = judgment.get("verdicts", {})
    applied = 0
    missing = judgment.get("missing") or None
    for index, event in enumerate(judgment.get("events", [])):
        exposed = {int(x["story_id"]) for x in event["exposures"]}
        event_verdicts = {sid: verdicts[sid] for sid in exposed if sid in verdicts}
        # A Story exposed but not mentioned by the judge counts as ignored.
        for sid in exposed - event_verdicts.keys():
            event_verdicts[sid] = ("ignored", "")
        try:
            store.apply_recall_verdicts(
                event["id"], session_id, event_verdicts,
                # 缺口描述属于整段会话；挂在最后一个事件上避免重复计数。
                missing_note=missing if index == len(judgment["events"]) - 1 else None,
            )
        except Exception:  # noqa: BLE001
            logger.warning("写入采纳判定失败 event=%s", event.get("id"))
            continue
        applied += len(event_verdicts)
    return applied
