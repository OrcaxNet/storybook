"""非阻塞 recall 反馈队列：查询读路径与 SQLite 写入解耦。"""
from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config, store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FeedbackJob:
    db_path: str
    story_ids: tuple[int, ...]


_QUEUE: queue.Queue[_FeedbackJob] = queue.Queue(
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
            store.apply_recall_feedback(
                list(job.story_ids), db_path=job.db_path
            )
        except Exception:  # noqa: BLE001 -- 反馈失败不能污染/阻断查询
            logger.warning("召回反馈写入失败")
        finally:
            _QUEUE.task_done()


atexit.register(flush_feedback, 1.0)
