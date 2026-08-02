"""测试共享工具：向量构造、可控 Ollama 桩、断言辅助。

与 conftest.py 分离，避免 ``from .conftest import`` 的导入歧义；
conftest.py 与各 test_*.py 均从此处导入。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

# 确保 src/ 在路径上，即使未做 editable 安装也能 `import storybook`（一键运行友好）。
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from storybook import config, store  # noqa: E402
from storybook import embeddings as embeddings_mod  # noqa: E402
from storybook import llm as llm_mod  # noqa: E402

DIM = config.EMBED_DIM


# ═══════════════════════════════════════════════
#  向量构造工具
# ═══════════════════════════════════════════════

def basis(index: int = 0) -> list[float]:
    """返回单位基向量 ``e_index``（已是 L2 归一化）。"""
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    return v.tolist()


def with_cos(ref_index: int, cos: float) -> list[float]:
    """返回一个单位向量，其与基向量 ``e_{ref_index}`` 的余弦相似度 == ``cos``。

    构造：``v[ref_index]=cos``，``v[ref_index+1]=sqrt(1-cos²)``，其余为 0。
    对 L2 归一化向量，``store.search_by_vector`` 换算 ``similarity = 1 - dist²/2``
    应精确等于 ``cos``（sqlite-vec 返回真实 L2 距离，非平方）。
    """
    cos = float(np.clip(cos, -1.0, 1.0))
    v = np.zeros(DIM, dtype=np.float32)
    v[ref_index] = cos
    v[ref_index + 1] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return v.tolist()


def hash_unit_vec(text: str) -> list[float]:
    """对任意文本返回确定性的 L2 归一化 1024 维向量（兜底用，保证 embed 永不返回空）。

    用文本哈希作种子驱动 numpy RandomState，避免直接把原始字节解释成 float32
    （那会引入 NaN/inf 破坏归一化）。同一文本永远得到同一向量。
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rng = np.random.RandomState(seed)
    arr = rng.randn(DIM).astype(np.float32)
    n = np.linalg.norm(arr)
    if n == 0 or not np.isfinite(n):
        arr = np.zeros(DIM, dtype=np.float32)
        arr[0] = 1.0
        n = 1.0
    return (arr / n).tolist()


# ═══════════════════════════════════════════════
#  可控桩：embeddings / llm
# ═══════════════════════════════════════════════

class FakeEmbedder:
    """可控的 ``embeddings.embed`` 桩。

    - ``register(text, vector)``：精确匹配某个 embed 输入文本，返回指定向量；传 None 模拟失败。
    - ``set_default(vector)``：未命中注册表时的兜底向量。
    - 默认兜底为按文本哈希的确定性单位向量（永不返回 None / 空）。
    """

    def __init__(self):
        self._overrides: dict[str, list[float] | None] = {}
        self._default: list[float] | None = None
        self.calls: list[str] = []

    def register(self, text: str, vector: list[float] | None) -> None:
        if vector is not None:
            assert len(vector) == DIM, f"向量维度应为 {DIM}，实际 {len(vector)}"
        self._overrides[text] = vector

    def set_default(self, vector: list[float]) -> None:
        assert len(vector) == DIM
        self._default = vector

    def __call__(self, text: str, model: str = None, **kwargs):
        self.calls.append(text)
        if text in self._overrides:
            return self._overrides[text]
        if self._default is not None:
            return self._default
        return hash_unit_vec(text)


class FakeLLM:
    """可控的 ``llm.*`` 桩。默认返回安全值，测试按需覆盖各方法返回。"""

    def __init__(self):
        self.keywords: list[str] = ["测试关键词"]
        self.summary: dict = {"title": "测试标题", "content": "问题：x 步骤：1.x 结果：ok"}
        self.merged: dict = {"title": "合并标题", "content": "合并内容 问题：x 步骤：1.x 结果：ok"}
        self.should_split: bool = False
        self.sub_stories: list[dict] = [
            {"title": "子记忆1", "content": "子内容1"},
            {"title": "子记忆2", "content": "子内容2"},
        ]
        self.stories: list[dict] | None = None
        self.transformation: dict | None = {
            "rewrite": "",
            "queries": [],
            "hypothetical_document": "",
        }
        self.calls: dict[str, int] = {}

    def _tick(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def extract_keywords(self, text):
        self._tick("extract_keywords")
        return list(self.keywords)

    def summarize_session(self, content):
        self._tick("summarize_session")
        return dict(self.summary)

    def form_stories(self, content):
        self._tick("form_stories")
        if self.stories is not None:
            return [dict(story) for story in self.stories]
        return [dict(self.summary)]

    def merge_stories(self, old, new):
        self._tick("merge_stories")
        return dict(self.merged)

    def judge_split(self, merged_text):
        self._tick("judge_split")
        return bool(self.should_split)

    def split_story(self, merged_text):
        self._tick("split_story")
        return [dict(s) for s in self.sub_stories]

    def transform_search_query(self, query, transformations, **kwargs):
        self._tick("transform_search_query")
        return dict(self.transformation) if self.transformation is not None else None


# ═══════════════════════════════════════════════
#  断言辅助：直接读取 vec0 索引中的向量
# ═══════════════════════════════════════════════

def vector_in_index(story_id: int) -> list[float] | None:
    """读取 ``story_vectors`` 虚表中某 story 的向量；不存在返回 None。"""
    db = store.get_db()
    try:
        row = db.execute(
            "SELECT embedding FROM story_vectors WHERE story_id = ?", (story_id,)
        ).fetchone()
        if not row:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32).tolist()
    finally:
        db.close()


# 暴露给 conftest 用于 monkeypatch
__all__ = [
    "DIM", "basis", "with_cos", "hash_unit_vec",
    "FakeEmbedder", "FakeLLM", "vector_in_index",
    "config", "store", "embeddings_mod", "llm_mod",
]
