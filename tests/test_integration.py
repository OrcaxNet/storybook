"""端到端集成测试：用 ``generate_sample_sessions`` 与 ``test_logs/*.json`` 提供固定输入，
mock Ollama 跑通 collector -> store -> processor -> search 全链路。

验收点：
- 无需启动 Ollama 即可跑通（全 mock）。
- 双向量存储一致性在真实写入路径下成立。
- 处理后所有会话状态正确流转。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from storybook import store, processor, collector, search as search_module
from ._helpers import basis, vector_in_index

TEST_LOGS = Path(__file__).resolve().parent.parent / "test_logs"


def _load_test_log_sessions() -> list[dict]:
    """读取 test_logs/*.json，归一化为 session 字典（复刻 cli 的 messages 解析逻辑）。"""
    sessions = []
    for f in sorted(TEST_LOGS.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        msgs = data.get("messages", [])
        raw = "\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in msgs)
        sessions.append({
            "source": data.get("id", f.name),
            "raw_content": raw,
            "problem_desc": data.get("title", (msgs[0]["content"][:200] if msgs else "")),
            "code_snippets": data.get("code_snippets", "[]"),
            "conclusion": data.get("conclusion", ""),
        })
    return sessions


def _register_orthogonal_topics(fake_embedder, sessions) -> dict[str, list[float]]:
    """为每个不同 problem_desc 注册一个正交基向量，使各会话互相 sim≈0 -> 各自 create。"""
    mapping: dict[str, list[float]] = {}
    idx = 0
    for s in sessions:
        pd = s["problem_desc"]
        if pd and pd not in mapping:
            vec = basis(idx)
            mapping[pd] = vec
            # process_session 的 embed 输入 = " ".join(keywords) + " " + problem_desc
            fake_embedder.register("topic " + pd, vec)
            idx += 1
    return mapping


# ═══════════════════════════════════════════════
#  generate_sample_sessions 全链路
# ═══════════════════════════════════════════════

class TestGenerateSampleSessionsFlow:
    def test_end_to_end_create_and_search(self, fake_llm, fake_embedder):
        fake_llm.keywords = ["topic"]
        fake_llm.summary = {"title": "记忆", "content": "问题：x 步骤：1.x 结果：ok"}

        sessions = collector.generate_sample_sessions(4)
        assert len(sessions) == 4
        count = collector.import_sessions(sessions)
        assert count == 4
        assert store.count_sessions() == 4
        assert len(store.get_pending_sessions()) == 4

        mapping = _register_orthogonal_topics(fake_embedder, sessions)

        summary = processor.process_all_pending(verbose=False)
        assert summary == {"total": 4, "success": 4, "failed": 0}

        # 4 条会话各创建 1 条 story（互相正交 -> 无合并）
        assert store.count_stories() == 4
        # 所有会话已 processed，无 pending 残留
        assert len(store.get_pending_sessions()) == 0

        # 双写一致性：每条 story 的向量在 stories.embedding 与 story_vectors 两处同步
        for story in store.get_all_stories():
            idx_vec = vector_in_index(story["id"])
            assert idx_vec is not None, f"story#{story['id']} 向量未入索引"
            assert story["embedding"] == pytest.approx(idx_vec, abs=1e-6)

        # 检索：用第一条会话的问题作为 query，应命中对应 story
        first_pd = sessions[0]["problem_desc"]
        fake_embedder.register(first_pd, mapping[first_pd])
        result = search_module.search(first_pd, top_k=3)
        assert result["top_matches"], "应检索到至少一条记忆"
        top = result["top_matches"][0]
        assert top["similarity"] == pytest.approx(1.0, abs=2e-3)
        # access_count 自增
        assert store.get_story(top["story_id"])["access_count"] == 1


# ═══════════════════════════════════════════════
#  test_logs/*.json 全链路
# ═══════════════════════════════════════════════

class TestTestLogsFlow:
    def test_end_to_end_with_test_logs(self, fake_llm, fake_embedder):
        fake_llm.keywords = ["topic"]
        fake_llm.summary = {"title": "记忆", "content": "问题-步骤-结果"}

        sessions = _load_test_log_sessions()
        assert len(sessions) >= 1, "test_logs 目录应有样例 JSON"
        collector.import_sessions(sessions)
        _register_orthogonal_topics(fake_embedder, sessions)

        summary = processor.process_all_pending(verbose=False)
        assert summary["failed"] == 0
        assert summary["success"] == len(sessions)
        assert store.count_stories() == len(sessions)

        # 每条 story 双写一致
        for story in store.get_all_stories():
            assert vector_in_index(story["id"]) is not None
            assert story["embedding"] == pytest.approx(
                vector_in_index(story["id"]), abs=1e-6)

        # 用最后一条会话的问题检索，命中自身
        last_pd = sessions[-1]["problem_desc"]
        fake_embedder.register(last_pd, basis(len(sessions) - 1))
        result = search_module.search(last_pd, top_k=3)
        assert result["top_matches"]
        assert result["top_matches"][0]["similarity"] == pytest.approx(1.0, abs=2e-3)
