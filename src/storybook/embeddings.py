"""嵌入层 — 统一 API 抽象，内部适配 Ollama / OpenAI-compatible 协议。"""
import logging
import os
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


class EmbeddingAPIError(RuntimeError):
    """可稳定诊断的 embedding API 错误；message 不包含凭据。"""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def api_identity(
    model: str | None = None,
    version: str | None = None,
    dimension: int | None = None,
    *,
    base_url: str | None = None,
    adapter: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, object]:
    """返回会影响向量结果的非敏感身份，用于缓存隔离。"""

    return {
        "type": config.EMBED_TYPE,
        "base_url": base_url or config.EMBED_BASE_URL,
        "adapter": adapter or config.EMBED_ADAPTER,
        "credential_env": (
            config.EMBED_API_KEY_ENV if api_key_env is None else api_key_env
        ),
        "model": model or config.EMBED_MODEL,
        "version": version or config.EMBED_VERSION,
        "dimension": dimension or config.EMBED_DIM,
    }


def _request_embedding(
    text: str,
    model: str,
    *,
    timeout_seconds: float,
    keep_alive: str | None,
    base_url: str | None = None,
    adapter: str | None = None,
    api_key_env: str | None = None,
) -> list[float]:
    request_base_url = (base_url or config.EMBED_BASE_URL).rstrip("/")
    request_adapter = adapter or config.EMBED_ADAPTER
    credential_env = config.EMBED_API_KEY_ENV if api_key_env is None else api_key_env
    headers: dict[str, str] = {}
    if credential_env:
        credential = os.getenv(credential_env)
        if not credential:
            raise EmbeddingAPIError(
                "credentials_missing",
                f"credential environment variable {credential_env} is missing",
            )
        headers["Authorization"] = f"Bearer {credential}"

    if request_adapter == "ollama":
        url = f"{request_base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
    else:
        url = f"{request_base_url}/embeddings"
        payload = {"model": model, "input": text}

    try:
        request_kwargs = {
            "json": payload,
            "timeout": (min(1.0, timeout_seconds), timeout_seconds),
        }
        if headers:
            request_kwargs["headers"] = headers
        response = requests.post(url, **request_kwargs)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise EmbeddingAPIError("endpoint_unreachable", "embedding endpoint unavailable") from exc
    except requests.RequestException as exc:
        raise EmbeddingAPIError("endpoint_unreachable", "embedding request failed") from exc

    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        raise EmbeddingAPIError("authentication_failed", "embedding API rejected credentials")
    if status_code in {404, 422}:
        raise EmbeddingAPIError("model_unavailable", "embedding model is unavailable")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise EmbeddingAPIError("model_unavailable", "embedding API rejected the request") from exc
    try:
        data = response.json()
        if request_adapter == "ollama":
            vector = data.get("embedding")
        else:
            rows = data.get("data")
            vector = rows[0].get("embedding") if isinstance(rows, list) and rows else None
        if not isinstance(vector, list) or not vector:
            raise TypeError("missing embedding vector")
        return [float(value) for value in vector]
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise EmbeddingAPIError(
            "response_protocol_incompatible", "embedding response protocol is incompatible"
        ) from exc


def probe() -> dict[str, object]:
    """无密钥泄露的 endpoint/model/protocol/dimension 探测结果。"""

    try:
        vector = _request_embedding(
            "storybook embedding probe",
            config.EMBED_MODEL,
            timeout_seconds=30.0,
            keep_alive=(
                config.EMBED_KEEP_ALIVE
                if config.EMBED_ADAPTER == "ollama"
                else None
            ),
        )
    except EmbeddingAPIError as exc:
        return {"ok": False, "reason": exc.reason, "dimension": 0}
    if config.EMBED_ADAPTER == "ollama":
        # A successful Ollama request loads the model, even if a subsequent
        # configured-dimension check reports an operator error.
        mark_model_used()
    actual = len(vector)
    if actual != config.EMBED_DIM:
        return {"ok": False, "reason": "dimension_mismatch", "dimension": actual}
    return {"ok": True, "reason": None, "dimension": actual}


def embed(
    text: str,
    model: str = None,
    *,
    timeout_seconds: float | None = None,
    keep_alive: str | None = None,
    cache_version: str | None = None,
) -> Optional[list[float]]:
    """调用配置的 API 生成语义向量，返回 L2 归一化后的向量。

    - 维度不匹配 / 零向量时返回 None（上层据此标记 failed 或报错，不再传入坏向量）。
    - 归一化保证 store.search_by_vector 中 ``1 - dist²/2`` 等于 cosine 相似度（精确）。
    """
    serving_request = model is None
    expected_dimension = config.EMBED_DIM
    request_base_url = config.EMBED_BASE_URL
    request_adapter = config.EMBED_ADAPTER
    request_api_key_env = config.EMBED_API_KEY_ENV
    if serving_request:
        try:
            from . import store
            state = store.get_embedding_index_state()
            model = state.get("active_model") or config.EMBED_MODEL
            expected_dimension = (
                store.serving_embedding_dimension() or config.EMBED_DIM
            )
            cache_version = (
                cache_version or state.get("active_version") or config.EMBED_VERSION
            )
            request_base_url = state.get("active_endpoint")
            request_adapter = state.get("active_adapter")
            request_api_key_env = state.get("active_api_key_env")
            if (
                not request_base_url or not request_adapter
                or request_api_key_env is None
            ):
                raise EmbeddingAPIError(
                    "active_index_identity_unknown",
                    "active embedding index endpoint identity is unknown; backfill required",
                )
        except Exception:  # schema may not exist during setup health probes
            if config.DB_PATH.exists():
                logger.error("Active embedding index identity is unavailable")
                return None
            model = config.EMBED_MODEL
            request_base_url = config.EMBED_BASE_URL
            request_adapter = config.EMBED_ADAPTER
            request_api_key_env = config.EMBED_API_KEY_ENV
    cache_version = cache_version or config.EMBED_VERSION
    cache_payload = {
        **api_identity(
            model, cache_version, expected_dimension,
            base_url=request_base_url, adapter=request_adapter,
            api_key_env=request_api_key_env,
        ),
        "text": text,
    }
    cached = inference_cache.get("embedding-v1", cache_payload)
    if isinstance(cached, list) and len(cached) == expected_dimension:
        mark_model_used()
        return [float(value) for value in cached]
    timeout_seconds = 30.0 if timeout_seconds is None else max(0.001, timeout_seconds)
    keep_alive = config.EMBED_KEEP_ALIVE if keep_alive is None else keep_alive
    try:
        vec = _request_embedding(
            text,
            model,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive if request_adapter == "ollama" else None,
            base_url=request_base_url,
            adapter=request_adapter,
            api_key_env=request_api_key_env,
        )
        if not vec or len(vec) != expected_dimension:
            logger.warning(
                "向量维度不匹配: got %d, expect %d",
                len(vec or []), expected_dimension,
            )
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
    except Exception as e:
        logger.error("Embedding 失败: %s", e)
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
