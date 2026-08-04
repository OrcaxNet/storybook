"""按 Profile 数据库与 ``index_version`` 隔离的进程内查询缓存。"""
from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from typing import Generic, TypeVar

from . import config

T = TypeVar("T")


class _TTLCache(Generic[T]):
    def __init__(self, max_size: int, ttl_seconds: float):
        self.max_size = max(0, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._values: OrderedDict[tuple, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: tuple) -> T | None:
        if self.max_size == 0 or self.ttl_seconds == 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._values.pop(key, None)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                return None
            self._values[key] = item
            return copy.deepcopy(value)

    def set(self, key: tuple, value: T) -> None:
        if self.max_size == 0 or self.ttl_seconds == 0:
            return
        with self._lock:
            self._values.pop(key, None)
            self._values[key] = (
                time.monotonic() + self.ttl_seconds,
                copy.deepcopy(value),
            )
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


_vector_cache = _TTLCache[list[float]](
    config.QUERY_VECTOR_CACHE_SIZE,
    config.QUERY_VECTOR_CACHE_TTL_SECONDS,
)
_result_cache = _TTLCache[dict](
    config.QUERY_RESULT_CACHE_SIZE,
    config.QUERY_RESULT_CACHE_TTL_SECONDS,
)


def index_identity(index_version: int) -> tuple:
    """绝对路径仅作为进程内隔离 key，不进入响应或诊断日志。"""

    return (
        str(config.DB_PATH.resolve()),
        int(index_version),
        config.EMBED_TYPE,
        config.EMBED_BASE_URL,
        config.EMBED_ADAPTER,
        config.EMBED_MODEL,
        config.EMBED_VERSION,
    )


def get_query_vector(identity: tuple, query: str) -> list[float] | None:
    return _vector_cache.get((*identity, query))


def set_query_vector(identity: tuple, query: str, vector: list[float]) -> None:
    _vector_cache.set((*identity, query), vector)


def get_result(identity: tuple, query: str, top_k: int) -> dict | None:
    return _result_cache.get((*identity, query, int(top_k)))


def set_result(identity: tuple, query: str, top_k: int, result: dict) -> None:
    _result_cache.set((*identity, query, int(top_k)), result)


def clear() -> None:
    _vector_cache.clear()
    _result_cache.clear()
