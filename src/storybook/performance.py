"""查询性能诊断：隐私安全的本地 JSONL 记录与最近窗口汇总。

只允许写入固定白名单字段。原始 query、Story 内容、绝对路径、主机名和
仓库 URL 均不在接口或落盘 schema 中，避免诊断能力意外变成内容日志。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

LATENCY_STAGES = (
    "cache", "embed", "vector", "lexical", "fusion", "transform", "fallback",
    "graph", "rerank", "serialize", "total"
)
_VALID_MODES = {"cache", "vector", "lexical_fallback", "unavailable", "error"}
_VALID_DEGRADED_REASONS = {
    "embedding_unavailable",
    "embedding_timeout",
    "vector_index_unavailable",
    "lexical_index_unavailable",
    "query_transform_timeout",
    "query_transform_unavailable",
    "transformed_embedding_timeout",
    "transformed_embedding_unavailable",
    "reranker_timeout",
    "reranker_unavailable",
    "reranker_circuit_open",
    "graph_unavailable",
    "write_feedback_failed",
}
_MAX_LOG_BYTES = 2 * 1024 * 1024
_MAX_RETAINED_RECORDS = 1000
_WRITE_LOCK = threading.Lock()


def now() -> float:
    """可注入的单调时钟，供查询阶段计时和确定性测试使用。"""
    return time.perf_counter()


def elapsed_ms(start: float, end: float | None = None) -> float:
    """返回非负毫秒耗时，保留三位小数。"""
    end = now() if end is None else end
    return round(max(0.0, end - start) * 1000, 3)


def empty_latency() -> dict[str, float]:
    return {stage: 0.0 for stage in LATENCY_STAGES}


def record_query_diagnostic(
    *,
    request_id: str,
    mode: str,
    latency_ms: dict[str, float],
    result_count: int,
    cache_hit: bool = False,
    degraded: bool = False,
    degraded_reason: str | None = None,
    model_state: str = "unknown",
) -> bool:
    """追加一条白名单诊断记录；失败时返回 ``False``，绝不影响查询结果。

    注意：函数刻意不接受 query、Story、路径或 URL 参数。即使调用方持有这些
    数据，也无法通过本接口写入诊断文件。
    """
    path = Path(config.PERFORMANCE_LOG_PATH)
    safe_mode = mode if mode in _VALID_MODES else "error"
    safe_reason = (
        degraded_reason if degraded_reason in _VALID_DEGRADED_REASONS else None
    )
    safe_state = model_state if model_state in {"warm", "cold", "unknown"} else "unknown"
    safe_latency = {
        stage: _safe_duration(latency_ms.get(stage, 0.0))
        for stage in LATENCY_STAGES
    }
    record = {
        "schema_version": 2,
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": _safe_request_id(request_id),
        "mode": safe_mode,
        "degraded": bool(degraded),
        "degraded_reason": safe_reason,
        "result_count": max(0, int(result_count)),
        "cache_hit": bool(cache_hit),
        "model_state": safe_state,
        "latency_ms": safe_latency,
    }
    line = json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, path.open("a+", encoding="utf-8") as fh:
            _lock_file(fh, exclusive=True)
            try:
                os.chmod(path, 0o600)
                fh.write(line)
                fh.flush()
                if fh.tell() > _MAX_LOG_BYTES:
                    _truncate_locked(fh)
            finally:
                _unlock_file(fh)
        return True
    except OSError:
        # 不把异常正文写入常规日志：文件系统错误常含绝对路径。
        logger.warning("查询性能诊断写入失败")
        return False


def read_query_diagnostics(limit: int = 100) -> list[dict]:
    """读取最近 ``limit`` 条合法记录；坏行跳过，文件不存在返回空列表。"""
    if limit <= 0:
        return []
    path = Path(config.PERFORMANCE_LOG_PATH)
    if not path.exists():
        return []

    records: deque[dict] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as fh:
            _lock_file(fh, exclusive=False)
            try:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(row, dict) and isinstance(row.get("latency_ms"), dict):
                        records.append(row)
            finally:
                _unlock_file(fh)
    except OSError:
        logger.warning("查询性能诊断读取失败")
        return []
    return list(records)


def summarize_query_performance(limit: int = 100) -> dict:
    """汇总最近查询的阶段 p50/p95、cache/fallback/degraded 比例。"""
    rows = read_query_diagnostics(limit=limit)
    count = len(rows)
    latency = {}
    for stage in LATENCY_STAGES:
        values = [
            _safe_duration(row.get("latency_ms", {}).get(stage, 0.0))
            for row in rows
        ]
        latency[stage] = {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
        }

    divisor = count or 1
    return {
        "sample_size": count,
        "window_size": limit,
        "latency_ms": latency,
        "cache_hit_ratio": round(
            sum(bool(row.get("cache_hit")) for row in rows) / divisor, 4
        ) if count else 0.0,
        "fallback_ratio": round(
            sum(row.get("mode") == "lexical_fallback" for row in rows) / divisor, 4
        ) if count else 0.0,
        "degraded_ratio": round(
            sum(bool(row.get("degraded")) for row in rows) / divisor, 4
        ) if count else 0.0,
    }


def percentile(values: Iterable[float], percent: float) -> float:
    """线性插值百分位；空集合返回 0，结果保留三位小数。"""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * min(100.0, max(0.0, percent)) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(value, 3)


def _safe_duration(value: object) -> float:
    try:
        return round(max(0.0, float(value)), 3)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _safe_request_id(value: object) -> str:
    candidate = str(value)
    if len(candidate) == 32 and all(
        character in "0123456789abcdef" for character in candidate.lower()
    ):
        return candidate.lower()
    # 外部传入的非 UUID 值只保存不可逆摘要，避免字段被滥用来夹带 query/路径。
    return hashlib.sha256(
        candidate.encode("utf-8", errors="replace")
    ).hexdigest()[:32]


def _truncate_locked(fh) -> None:
    fh.seek(0)
    lines = deque(fh, maxlen=_MAX_RETAINED_RECORDS)
    fh.seek(0)
    fh.truncate()
    fh.writelines(lines)
    fh.flush()


def _lock_file(fh, *, exclusive: bool) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows beta；线程锁仍生效
        return
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(fh.fileno(), operation)


def _unlock_file(fh) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows beta
        return
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
