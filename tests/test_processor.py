"""记忆加工层测试：create / merge / update 三分支 + split 路径。

全程 mock ``llm`` / ``embeddings``，通过精确构造余弦相似度驱动分支选择，
并断言每条分支的边建立/权重强化与双向量一致性。

阈值（见 config）：
  SIM_THRESHOLD_LOW=0.75  SIM_THRESHOLD_HIGH=0.85
  SIM_THRESHOLD_UPDATE_ONLY=0.92
分支：
  best sim < 0.85            -> create（与 [0.75,0.85) 弱匹配建边，weight=sim）
  0.85 <= sim < 0.92         -> merge（judge_split 决定是否 split）
  sim >= 0.92                -> update（仅合并关键词/向量/强化边）
"""
from __future__ import annotations

import pytest

from storybook import store, processor, config
from ._helpers import basis, with_cos, vector_in_index


def _add_session(problem_desc: str = "PD", raw: str = "raw content") -> int:
    return store.add_session("test", raw, problem_desc, "[]", "")


def _embed_text(keywords: list[str], problem_desc: str) -> str:
    """复刻 processor.process_session 中构造的 embed 输入文本。"""
    return " ".join(keywords) + " " + (problem_desc or "")


def _seed_story(title: str, vec: list[float], keywords: list[str] | None = None,
                content: str = "旧内容") -> int:
    return store.add_story(title, content, keywords or [], vec, source_session_ids=[])


# ═══════════════════════════════════════════════
#  create 分支
# ═══════════════════════════════════════════════

class TestCreateBranch:
    def test_create_on_no_high_match(self, fake_llm, fake_embedder):
        """best sim < 0.85：新建 story，与 [0.75,0.85) 弱匹配建边，weight=sim。"""
        fake_llm.keywords = ["kw"]
        fake_llm.summary = {"title": "新记忆", "content": "问题：x 步骤：1.x 结果：ok"}

        # 预置一条弱匹配 story（sim 0.80）
        weak = _seed_story("weak", with_cos(0, 0.80), keywords=["weak_kw"])
        # query = basis(0)，与 weak 的 sim=0.80（落在 [0.75,0.85)）
        sid = _add_session("PD_CREATE")
        et = _embed_text(["kw"], "PD_CREATE")
        fake_embedder.register(et, basis(0))

        new_id = processor.process_session(sid)

        # 1) 新建了一条 story
        assert new_id is not None and new_id != weak
        assert store.count_stories() == 2
        new_story = store.get_story(new_id)
        assert new_story["title"] == "新记忆"
        assert new_story["source_session_ids"] == [sid]

        # 2) 与弱匹配建立了无向边，weight=sim≈0.80
        edges = store.get_edges(new_id)
        assert len(edges) == 1
        assert edges[0]["related_id"] == weak
        assert edges[0]["weight"] == pytest.approx(0.80, abs=2e-3)
        assert edges[0]["edge_type"] == "semantic"

        # 3) 双写一致：新 story 向量两处同步
        assert vector_in_index(new_id) is not None
        assert store.get_story(new_id)["embedding"][:3] == pytest.approx(basis(0)[:3], abs=1e-6)

        # 4) 会话已标记 processed；用了 summarize，没用 merge/split
        assert store.get_session(sid)["status"] == "processed"
        assert fake_llm.calls.get("form_stories") == 1
        assert fake_llm.calls.get("merge_stories", 0) == 0
        assert fake_llm.calls.get("split_story", 0) == 0

    def test_create_with_no_matches_no_edges(self, fake_llm, fake_embedder):
        """无任何匹配（sim < 0.75）：新建 story，不建边。"""
        fake_llm.keywords = ["kw"]
        # 不预置任何 story -> 检索为空
        sid = _add_session("PD_ALONE")
        fake_embedder.register(_embed_text(["kw"], "PD_ALONE"), basis(0))

        new_id = processor.process_session(sid)
        assert new_id is not None
        assert store.get_edges(new_id) == []
        assert store.count_stories() == 1


# ═══════════════════════════════════════════════
#  update 分支（sim >= 0.92）
# ═══════════════════════════════════════════════

class TestUpdateBranch:
    def test_update_only_supplements_details(self, fake_llm, fake_embedder):
        """sim >= 0.92：仅合并关键词、更新向量、强化已有边权重；不调用 merge/split。"""
        fake_llm.keywords = ["new_kw"]

        old = _seed_story("old", with_cos(0, 0.95), keywords=["old_kw"])
        other = _seed_story("other", with_cos(0, 0.80), keywords=["o_kw"])
        # 预置 old-other 边（increment 只对已存在边生效）
        store.add_or_update_edge(old, other, 0.3, "semantic")

        sid = _add_session("PD_UPDATE")
        # query=basis(0)：old sim=0.95(>=0.92 -> update)，other sim=0.80(弱匹配)
        fake_embedder.register(_embed_text(["new_kw"], "PD_UPDATE"), basis(0))

        ret = processor.process_session(sid)

        # 返回的是被更新的 old story id
        assert ret == old
        story = store.get_story(old)
        assert story["version"] == 2
        # 关键词合并去重（set 无序，按集合断言）
        assert set(story["keywords"]) == {"old_kw", "new_kw"}
        # 内容不变（update 不改 content）
        assert story["content"] == "旧内容"

        # old-other 边被强化 +0.1（每对仅一次）
        edge = store.get_edges(old)
        assert len(edge) == 1
        assert edge[0]["related_id"] == other
        assert edge[0]["weight"] == pytest.approx(0.4, abs=1e-6)

        # 双写一致
        idx_vec = vector_in_index(old)
        assert idx_vec is not None
        assert store.get_story(old)["embedding"] == pytest.approx(idx_vec, abs=1e-6)

        # 没走 merge/split/summarize；只走了 extract_keywords
        assert fake_llm.calls.get("extract_keywords") == 1
        assert fake_llm.calls.get("form_stories") == 1
        assert fake_llm.calls.get("merge_stories", 0) == 0
        assert fake_llm.calls.get("judge_split", 0) == 0
        assert store.get_session(sid)["status"] == "processed"

    def test_update_edge_weight_clamps_to_max(self, fake_llm, fake_embedder):
        """强化边权重不超过 WEIGHT_MAX=1.0。"""
        fake_llm.keywords = ["new_kw"]
        old = _seed_story("old", with_cos(0, 0.95), keywords=["old_kw"])
        other = _seed_story("other", with_cos(0, 0.80))
        store.add_or_update_edge(old, other, 0.95)   # 0.95+0.1=1.05 -> 封顶 1.0
        sid = _add_session("PD_CLAMP")
        fake_embedder.register(_embed_text(["new_kw"], "PD_CLAMP"), basis(0))

        processor.process_session(sid)
        assert store.get_edges(old)[0]["weight"] == pytest.approx(config.WEIGHT_MAX)


# ═══════════════════════════════════════════════
#  merge 分支（0.85 <= sim < 0.92，不分裂）
# ═══════════════════════════════════════════════

class TestMergeBranch:
    def test_merge_no_split(self, fake_llm, fake_embedder):
        """0.85 <= sim < 0.92 且 judge_split=False：合并内容到旧 story，强化边。"""
        fake_llm.keywords = ["new_kw"]
        fake_llm.merged = {"title": "合并后标题", "content": "合并后内容"}
        fake_llm.should_split = False

        old = _seed_story("old", with_cos(0, 0.88), keywords=["old_kw"])
        other = _seed_story("other", with_cos(0, 0.80))
        store.add_or_update_edge(old, other, 0.3)

        sid = _add_session("PD_MERGE")
        # query=basis(0)：old sim=0.88(->merge)，other sim=0.80
        fake_embedder.register(_embed_text(["new_kw"], "PD_MERGE"), basis(0))

        ret = processor.process_session(sid)

        assert ret == old
        story = store.get_story(old)
        assert story["version"] == 2
        assert story["title"] == "合并后标题"
        assert story["content"] == "合并后内容"
        assert set(story["keywords"]) == {"old_kw", "new_kw"}

        # old-other 边强化
        assert store.get_edges(old)[0]["weight"] == pytest.approx(0.4, abs=1e-6)

        # 调用了 summarize + merge + judge_split，没 split
        assert fake_llm.calls.get("form_stories") == 1
        assert fake_llm.calls.get("merge_stories") == 1
        assert fake_llm.calls.get("judge_split") == 1
        assert fake_llm.calls.get("split_story", 0) == 0
        assert store.get_session(sid)["status"] == "processed"

    def test_merge_idempotent_session_skipped(self, fake_llm, fake_embedder):
        """已 processed 的会话再次处理应被跳过，返回 None。"""
        fake_llm.keywords = ["kw"]
        sid = _add_session("PD_ONCE")
        fake_embedder.register(_embed_text(["kw"], "PD_ONCE"), basis(0))
        assert processor.process_session(sid) is not None
        # 第二次：status 已 processed
        assert processor.process_session(sid) is None


# ═══════════════════════════════════════════════
#  split 分支（merge + judge_split=True）
# ═══════════════════════════════════════════════

class TestSplitBranch:
    def test_split_creates_children_and_removes_parent_vector(self, fake_llm, fake_embedder):
        """0.85 <= sim < 0.92 且 judge_split=True：拆分为子 story，父向量从索引移除。"""
        fake_llm.keywords = ["new_kw"]
        fake_llm.merged = {"title": "合并标题", "content": "合并内容（将分裂）"}
        fake_llm.should_split = True
        fake_llm.sub_stories = [
            {"title": "子1", "content": "子内容1"},
            {"title": "子2", "content": "子内容2"},
        ]

        parent = _seed_story("parent", with_cos(0, 0.88), keywords=["parent_kw"])
        sid = _add_session("PD_SPLIT")
        # query=basis(0)：parent sim=0.88 -> merge -> judge_split=True -> split
        fake_embedder.register(_embed_text(["new_kw"], "PD_SPLIT"), basis(0))

        ret = processor.process_session(sid)

        # 子 story 列表
        all_stories = {s["id"]: s for s in store.get_all_stories()}
        children = [s for s in all_stories.values() if s["parent_id"] == parent]
        assert len(children) == 2
        child_ids = sorted(s["id"] for s in children)
        # _split_and_store 返回首个子 story
        assert ret in child_ids

        # 1) 父向量已从索引移除（验收点4：split 后父 story 向量已从索引移除）
        assert vector_in_index(parent) is None
        parent_story = store.get_story(parent)
        assert parent_story["embedding"] == []          # NULL -> []
        assert parent_story["title"] == "parent"        # 父行保留用于溯源

        # 2) 子 story 向量在索引中（双写一致）
        for c in children:
            assert vector_in_index(c["id"]) is not None
            assert store.get_story(c["id"])["embedding"] == pytest.approx(
                vector_in_index(c["id"]), abs=1e-6)

        # 3) 父子边 weight=1.0，类型 parent_child（双向可查）
        c1, c2 = child_ids
        edges_parent = store.get_edges(parent)
        pc = {e["related_id"]: e for e in edges_parent}
        assert set(pc) == {c1, c2}
        for cid in (c1, c2):
            assert pc[cid]["weight"] == pytest.approx(config.WEIGHT_PARENT_CHILD)
            assert pc[cid]["edge_type"] == "parent_child"

        # 4) 兄弟关系用标准 semantic 边，精确语义留在 provenance
        edges_c1 = store.get_edges(c1)
        sibling = [e for e in edges_c1 if e["related_id"] == c2]
        assert len(sibling) == 1
        assert sibling[0]["weight"] == pytest.approx(0.5)
        assert sibling[0]["edge_type"] == "semantic"
        assert sibling[0]["provenance"]["relationship"] == "sibling"

        # 5) 子 story 关键词 = 父关键词 + 本会话关键词（去重保序）
        for c in children:
            assert c["keywords"] == ["parent_kw", "new_kw"]

        # 6) 父向量移除后不再被检索命中
        assert all(r["story_id"] != parent for r in store.search_by_vector(basis(0), top_k=10))

        # 7) LLM 调用符合预期
        assert fake_llm.calls.get("form_stories") == 1
        assert fake_llm.calls.get("merge_stories") == 1
        assert fake_llm.calls.get("judge_split") == 1
        assert fake_llm.calls.get("split_story") == 1
        assert store.get_session(sid)["status"] == "processed"


# ═══════════════════════════════════════════════
#  边界与异常
# ═══════════════════════════════════════════════

class TestProcessorBoundaries:
    def test_missing_session_returns_none(self, fake_llm, fake_embedder):
        assert processor.process_session(99999) is None

    def test_embedding_failure_marks_session_failed(self, fake_llm, fake_embedder):
        """embed 返回 None 时，会话标记 failed，返回 None（不写入坏向量）。"""
        fake_llm.keywords = ["kw"]
        sid = _add_session("PD_FAIL")
        fake_embedder.register(_embed_text(["kw"], "PD_FAIL"), None)  # 模拟 embedding 失败

        assert processor.process_session(sid) is None
        assert store.get_session(sid)["status"] == "failed"
        assert store.count_stories() == 0   # 未写入任何 story

    def test_keyword_fallback_when_llm_returns_empty(self, fake_llm, fake_embedder):
        """LLM 关键词为空时，回退用 problem_desc 分词。"""
        fake_llm.keywords = []   # 触发 fallback
        sid = _add_session("alpha beta")  # 两个词
        # fallback keywords = problem_desc.split()[:5] = ["alpha","beta"]
        et = _embed_text(["alpha", "beta"], "alpha beta")
        fake_embedder.register(et, basis(0))

        new_id = processor.process_session(sid)
        assert new_id is not None
        # 新 story 用 fallback 关键词落库
        assert store.get_story(new_id)["keywords"] == ["alpha", "beta"]

    def test_process_all_pending_runs_each(self, fake_llm, fake_embedder):
        """批量处理所有 pending 会话：各自走 create 分支（互不相似）。"""
        fake_llm.keywords = ["kw"]
        # 3 个会话用正交基向量，互相 sim=0 -> 各自 create
        for i, pd in enumerate(["A", "B", "C"]):
            sid = _add_session(pd)
            fake_embedder.register(_embed_text(["kw"], pd), basis(i))

        summary = processor.process_all_pending(verbose=False)
        assert summary["total"] == 3
        assert summary["success"] == 3
        assert summary["failed"] == 0
        assert store.count_stories() == 3
