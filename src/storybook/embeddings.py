"""
嵌入层 — 封装 Ollama embedding API + 余弦相似度
"""
import logging
import threading
import time
from typing import Optional

import numpy as np
import requests

from . import config
from . import inference_cache

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_LAST_SUCCESS_AT: float | None = None


def embed(
    text: str,
    model: str = None,
    *,
    timeout_seconds: float | None = None,
    keep_alive: str | None = None,
    cache_version: str | None = None,
) -> Optional[list[float]]:
    """调用 Ollama 生成语义向量，返回 L2 归一化后的向量。

    - 维度不匹配 / 零向量时返回 None（上层据此标记 failed 或报错，不再传入坏向量）。
    - 归一化保证 store.search_by_vector 中 ``1 - dist²/2`` 等于 cosine 相似度（精确）。
    """
    if model is None:
        try:
            from . import store
            state = store.get_embedding_index_state()
            model = state.get("active_model") or config.EMBED_MODEL
            cache_version = (
                cache_version or state.get("active_version") or config.EMBED_VERSION
            )
        except Exception:  # schema may not exist during setup health probes
            model = config.EMBED_MODEL
    cache_version = cache_version or config.EMBED_VERSION
    cache_payload = {
        "model": model,
        "version": cache_version,
        "dimension": config.EMBED_DIM,
        "text": text,
    }
    cached = inference_cache.get("embedding-v1", cache_payload)
    if isinstance(cached, list) and len(cached) == config.EMBED_DIM:
        mark_model_used()
        return [float(value) for value in cached]
    timeout_seconds = 30.0 if timeout_seconds is None else max(0.001, timeout_seconds)
    keep_alive = config.EMBED_KEEP_ALIVE if keep_alive is None else keep_alive
    try:
        if config.EMBED_PROVIDER == "api":
            resp = requests.post(
                f"{config.EMBED_BASE_URL.rstrip('/')}/v1/embeddings",
                headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
                json={"model": model, "input": text},
                timeout=(min(1.0, timeout_seconds), timeout_seconds),
            )
        else:
            resp = requests.post(
                f"{config.EMBED_BASE_URL.rstrip('/')}/api/embeddings",
                json={"model": model, "prompt": text, "keep_alive": keep_alive},
                timeout=(min(1.0, timeout_seconds), timeout_seconds),
            )
        resp.raise_for_status()
        data = resp.json()
        if config.EMBED_PROVIDER == "api":
            rows = data.get("data") if isinstance(data, dict) else None
            vec = rows[0].get("embedding") if isinstance(rows, list) and rows else None
        else:
            vec = data.get("embedding")
        if not vec or len(vec) != config.EMBED_DIM:
            logger.warning("向量维度不匹配: got %d, expect %d", len(vec or []), config.EMBED_DIM)
            return None
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            logger.warning("零向量，跳过")
            return None
        result = (arr / norm).tolist()
        inference_cache.set("embedding-v1", cache_payload, result)
        mark_model_used()
        return result
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.error(
            "Embedding failed provider=%s category=%s",
            config.EMBED_PROVIDER,
            f"http_{status}" if status else type(exc).__name__,
        )
        return None


def model_state() -> str:
    """返回本进程观察到的模型 warm/cold 状态。"""

    with _STATE_LOCK:
        last_success = _LAST_SUCCESS_AT
    if last_success is None:
        return "cold"
    if time.monotonic() - last_success > config.EMBED_WARM_WINDOW_SECONDS:
        return "cold"
    return "warm"


def mark_model_used() -> None:
    global _LAST_SUCCESS_AT
    with _STATE_LOCK:
        _LAST_SUCCESS_AT = time.monotonic()


def mark_model_cold() -> None:
    global _LAST_SUCCESS_AT
    with _STATE_LOCK:
        _LAST_SUCCESS_AT = None


def prewarm() -> bool:
    """best-effort 预热 embedding 模型，并用 keep_alive 保持驻留。"""

    return bool(embed(
        "storybook embedding warmup",
        timeout_seconds=config.QUERY_COLD_TIMEOUT_SECONDS,
        keep_alive=config.EMBED_KEEP_ALIVE,
    ))


def backfill(
    *,
    model: str,
    version: str,
    representation: str = "default",
    batch_size: int = 100,
    activate: bool = True,
) -> dict:
    """Incrementally build a shadow embedding index and optionally activate it.

    Ready rows are content-hash checked and skipped on retries.  A failed vector
    remains in the shadow table with an attempt counter; the current serving
    vec0 index is untouched until every live Story is ready and activation can
    commit atomically.
    """

    # Local imports avoid a store -> embeddings -> store import cycle.
    from . import store
    from . import story_v2

    store.begin_embedding_backfill(model, version, representation)
    pending = store.stories_pending_embedding_backfill(
        version,
        representation,
        limit=max(1, batch_size),
    )
    attempted = succeeded = failed = 0
    for story in pending:
        attempted += 1
        vector = embed(
            story_v2.embedding_input(story, representation),
            model=model,
            cache_version=version,
        )
        if vector:
            succeeded += 1
            store.stage_embedding_backfill(
                story["id"],
                model=model,
                version=version,
                representation=representation,
                content_hash=story["target_content_hash"],
                embedding=vector,
            )
        else:
            failed += 1
            store.stage_embedding_backfill(
                story["id"],
                model=model,
                version=version,
                representation=representation,
                content_hash=story["target_content_hash"],
                embedding=None,
                error="embedding unavailable or dimension mismatch",
            )

    progress = store.embedding_backfill_progress(version, representation)
    activation = None
    if activate and progress["pending"] == 0:
        activation = store.activate_embedding_backfill(
            model=model,
            version=version,
            representation=representation,
        )
    elif progress["pending"] == 0:
        store.mark_embedding_backfill_ready()
    return {
        "model": model,
        "version": version,
        "representation": representation,
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "progress": progress,
        "activation": activation,
        "serving_index_unchanged": activation is None,
    }
