"""
嵌入层 — 封装 Ollama embedding API + 余弦相似度
"""
import logging
from typing import Optional

import numpy as np
import requests

from . import config

logger = logging.getLogger(__name__)


def embed(text: str, model: str = None) -> Optional[list[float]]:
    """调用 Ollama 生成语义向量，返回 L2 归一化后的向量。

    - 维度不匹配 / 零向量时返回 None（上层据此标记 failed 或报错，不再传入坏向量）。
    - 归一化保证 store.search_by_vector 中 ``1 - dist²/2`` 等于 cosine 相似度（精确）。
    """
    model = model or config.EMBED_MODEL
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding")
        if not vec or len(vec) != config.EMBED_DIM:
            logger.warning("向量维度不匹配: got %d, expect %d", len(vec or []), config.EMBED_DIM)
            return None
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            logger.warning("零向量，跳过")
            return None
        return (arr / norm).tolist()
    except Exception as e:
        logger.error("Embedding 失败: %s", e)
        return None
