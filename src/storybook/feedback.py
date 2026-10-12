"""非阻塞 recall 反馈队列：查询读路径与 SQLite 写入解耦。"""
from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from datetime import UTC, datetime
from dataclasses import dataclass, field
from pathlib import Path

from . import config, store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FeedbackJob:
    db_path: str
    story_ids: tuple[int, ...]


@dataclass(frozen=True)
class _EventJob:
    """Agent 侧召回埋点（曝光）或 get_story 点击。"""

    db_path: str
    kind: str  # "event" | "click"
    payload: dict = field(default_factory=dict)


_QUEUE: queue.Queue[_FeedbackJob | _EventJob] = queue.Queue(
    maxsize=max(1, config.QUERY_FEEDBACK_QUEUE_SIZE)
)
_START_LOCK = threading.Lock()
_WORKER: threading.Thread | None = None


def enqueue_recall_feedback(story_ids: list[int]) -> bool:
    """入队成功返回 True；队列满时丢弃反馈，不阻塞查询。"""

    unique_ids = tuple(dict.fromkeys(int(story_id) for story_id in story_ids))
    if not unique_ids:
        return True
    _ensure_worker()
    job = _FeedbackJob(str(Path(config.DB_PATH).resolve()), unique_ids)
    try:
        _QUEUE.put_nowait(job)
        return True
    except queue.Full:
        logger.warning("召回反馈队列已满，本次反馈已丢弃")
        return False


def enqueue_recall_event(
    *,
    request_id: str,
    origin: str,
    query: str,
    matches: list[dict],
    workspace_path_hash: str | None = None,
    result_state: str | None = None,
) -> bool:
    """记录一次 Agent 侧召回曝光；与查询读路径解耦，队列满时丢弃。"""

    if not config.RECALL_TELEMETRY_ENABLED:
        return False
    exposures = [
        (int(m["story_id"]), rank, _score(m))
        for rank, m in enumerate(matches, 1)
    ]
    return _put(_EventJob(_db_path(), "event", {
        "request_id": request_id,
        "origin": origin,
        "query": query,
        "exposures": exposures,
        "workspace_path_hash": workspace_path_hash,
        "result_state": result_state,
        "created_at": datetime.now(UTC).isoformat(),
    }))


def enqueue_story_click(story_id: int) -> bool:
    """get_story 视为对最近一次曝光的点击。"""

    if not config.RECALL_TELEMETRY_ENABLED:
        return False
    return _put(_EventJob(_db_path(), "click", {
        "story_id": int(story_id),
        "now": datetime.now(UTC).isoformat(),
    }))


def _score(match: dict) -> float | None:
    value = match.get("score", match.get("similarity"))
    return round(float(value), 6) if isinstance(value, (int, float)) else None


def _db_path() -> str:
    return str(Path(config.DB_PATH).resolve())


def _put(job) -> bool:
    _ensure_worker()
    try:
        _QUEUE.put_nowait(job)
        return True
    except queue.Full:
        logger.warning("召回反馈队列已满，本次反馈已丢弃")
        return False


def flush_feedback(timeout: float = 1.0) -> bool:
    """等待已入队反馈完成；主要供测试和短生命周期 CLI 退出使用。"""

    deadline = time.monotonic() + max(0.0, timeout)
    while _QUEUE.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.002)
    return True


def _ensure_worker() -> None:
    global _WORKER
    if _WORKER is not None and _WORKER.is_alive():
        return
    with _START_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _WORKER = threading.Thread(
            target=_worker_loop,
            name="storybook-recall-feedback",
            daemon=True,
        )
        _WORKER.start()


def _worker_loop() -> None:
    while True:
        job = _QUEUE.get()
        try:
            if isinstance(job, _EventJob):
                _apply_event(job)
            else:
                store.apply_recall_feedback(
                    list(job.story_ids), db_path=job.db_path
                )
        except Exception:  # noqa: BLE001 -- 反馈失败不能污染/阻断查询
            logger.warning("召回反馈写入失败")
        finally:
            _QUEUE.task_done()


def _apply_event(job: _EventJob) -> None:
    if job.kind == "event":
        store.record_recall_event(db_path=job.db_path, **job.payload)
    elif job.kind == "click":
        store.record_story_click(
            job.payload["story_id"],
            now=datetime.fromisoformat(job.payload["now"]),
            db_path=job.db_path,
        )


atexit.register(flush_feedback, 1.0)
