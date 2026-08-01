"""存储层测试：Session/Story CRUD、无向边归一、向量相似度换算、双写一致性。

覆盖验收点：
- store 层 CRUD、``_edge_pair`` 无向归一、``search_by_vector`` 的 ``1 - dist²/2`` 换算、
  双写一致性（``add_story``/``update_story`` 后 ``stories.embedding`` 与 ``story_vectors`` 同步）。
"""
from __future__ import annotations

import os
import json
import sqlite3
import stat
import uuid

import numpy as np
import pytest

from storybook import store, config
from ._helpers import basis, with_cos, vector_in_index


# ═══════════════════════════════════════════════
#  Session CRUD
# ═══════════════════════════════════════════════

class TestSessionCRUD:
    def test_add_and_get_session(self):
        sid = store.add_session("manual", "raw content", "problem", "[]", "conclusion")
        assert sid >= 1
        s = store.get_session(sid)
        assert s["source"] == "manual"
        assert s["raw_content"] == "raw content"
        assert s["status"] == "pending"
        uuid.UUID(s["global_id"])
        assert s["profile_id"] == config.PROFILE_ID
        assert s["sync_state"] == "local_only"

    def test_count_and_pending(self):
        assert store.count_sessions() == 0
        store.add_session("a", "r1", "p1")
        store.add_session("b", "r2", "p2")
        assert store.count_sessions() == 2
        pending = store.get_pending_sessions()
        assert len(pending) == 2
        # 按 id 升序
        assert [r["id"] for r in pending] == sorted(r["id"] for r in pending)

    def test_pending_limit(self):
        for i in range(5):
            store.add_session("a", f"r{i}", f"p{i}")
        assert len(store.get_pending_sessions(limit=2)) == 2

    def test_update_session_status(self):
        sid = store.add_session("a", "r", "p")
        store.update_session_status(sid, "processed")
        s = store.get_session(sid)
        assert s["status"] == "processed"
        assert s["processed_at"] is not None
        # processed 不再出现在 pending
        assert all(r["id"] != sid for r in store.get_pending_sessions())

    def test_get_session_missing(self):
        assert store.get_session(99999) is None


# ═══════════════════════════════════════════════
#  Story CRUD
# ═══════════════════════════════════════════════

class TestStoryCRUD:
    def test_add_and_get_story(self):
        vec = basis(0)
        sid = store.add_story("标题", "内容", ["k1", "k2"], vec, source_session_ids=[10, 11])
        story = store.get_story(sid)
        assert story["title"] == "标题"
        assert story["keywords"] == ["k1", "k2"]
        assert story["source_session_ids"] == [10, 11]
        assert story["parent_id"] is None
        assert story["version"] == 1
        assert story["access_count"] == 0
        uuid.UUID(story["global_id"])
        assert story["profile_id"] == config.PROFILE_ID
        assert story["sync_state"] == "local_only"
        # embedding 应被还原为 list[float]，长度 = EMBED_DIM
        assert len(story["embedding"]) == config.EMBED_DIM

    def test_add_story_default_keywords_and_sources(self):
        sid = store.add_story("t", "c", [], basis(1))
        story = store.get_story(sid)
        assert story["keywords"] == []
        assert story["source_session_ids"] == []

    def test_get_all_stories_ordered(self):
        a = store.add_story("a", "c", [], basis(0))
        b = store.add_story("b", "c", [], basis(1))
        stories = store.get_all_stories()
        assert [s["id"] for s in stories] == [a, b]

    def test_count_stories(self):
        assert store.count_stories() == 0
        store.add_story("a", "c", [], basis(0))
        assert store.count_stories() == 1

    def test_update_story_partial_fields_and_version(self):
        sid = store.add_story("t", "c", ["k"], basis(0))
        store.update_story(sid, title="新标题")  # 仅改 title
        story = store.get_story(sid)
        assert story["title"] == "新标题"
        assert story["content"] == "c"            # 未传 -> 不变
        assert story["keywords"] == ["k"]         # 未传 -> 不变
        assert story["version"] == 2              # 每次更新 version+1

        store.update_story(sid, content="新内容", keywords=["k1", "k2"])
        story = store.get_story(sid)
        assert story["content"] == "新内容"
        assert story["keywords"] == ["k1", "k2"]
        assert story["version"] == 3

    def test_update_story_keywords_serialized_as_json(self):
        sid = store.add_story("t", "c", [], basis(0))
        store.update_story(sid, keywords=["中文", "English", "with space"])
        assert store.get_story(sid)["keywords"] == ["中文", "English", "with space"]

    def test_increment_access_count(self):
        sid = store.add_story("t", "c", [], basis(0))
        assert store.get_story(sid)["access_count"] == 0
        store.increment_access_count(sid)
        store.increment_access_count(sid)
        assert store.get_story(sid)["access_count"] == 2

    def test_update_story_raw_sessions(self):
        sid = store.add_story("t", "c", [], basis(0), source_session_ids=[1])
        store.update_story_raw_sessions(sid, [1, 2, 3])
        assert store.get_story(sid)["source_session_ids"] == [1, 2, 3]

    def test_get_story_missing(self):
        assert store.get_story(99999) is None


# ═══════════════════════════════════════════════
#  无向边归一 _edge_pair / add_or_update_edge / increment_edge_weight
# ═══════════════════════════════════════════════

class TestEdges:
    def test_edge_pair_normalizes_order(self):
        assert store._edge_pair(3, 1) == (1, 3)
        assert store._edge_pair(1, 3) == (1, 3)
        assert store._edge_pair(2, 2) == (2, 2)

    def _two_stories(self):
        """edges 表有外键约束，需先建 story。"""
        a = store.add_story("a", "c", [], basis(0))
        b = store.add_story("b", "c", [], basis(1))
        return a, b

    def test_add_or_update_edge_is_undirected(self):
        a, b = self._two_stories()
        # 正反方向建边应只产生一行
        store.add_or_update_edge(a, b, 0.5)
        store.add_or_update_edge(b, a, 0.7)   # 反向，取 max(0.5, 0.7)=0.7
        edges = store.get_edges(a)
        assert len(edges) == 1
        assert edges[0]["weight"] == pytest.approx(0.7)
        uuid.UUID(edges[0]["global_id"])
        assert edges[0]["profile_id"] == config.PROFILE_ID
        assert edges[0]["sync_state"] == "local_only"
        # 从另一端也能查到同一条
        assert len(store.get_edges(b)) == 1

    def test_add_or_update_edge_takes_max(self):
        a, b = self._two_stories()
        store.add_or_update_edge(a, b, 0.6)
        store.add_or_update_edge(a, b, 0.4)   # 较小，不降权
        assert store.get_edges(a)[0]["weight"] == pytest.approx(0.6)

    def test_add_or_update_edge_clamps_to_max_on_update(self):
        """已存在边再写入超过 WEIGHT_MAX 的权重时，UPDATE 路径封顶到 1.0。

        注：当前实现仅在 UPDATE（已存在）路径做 ``min(..., WEIGHT_MAX)`` 封顶；
        INSERT 新边时不封顶。正常流程不会插入 >1.0 的权重（create 用
        similarity≤1.0、split 用常量），故此处只覆盖已实现的 UPDATE 封顶。
        """
        a, b = self._two_stories()
        store.add_or_update_edge(a, b, 0.6)
        store.add_or_update_edge(a, b, 1.5)   # update 路径：max(0.6,1.5)=1.5 -> 封顶 1.0
        assert store.get_edges(a)[0]["weight"] == pytest.approx(config.WEIGHT_MAX)

    def test_increment_edge_weight_undirected_and_clamped(self):
        a, b = self._two_stories()
        store.add_or_update_edge(a, b, 0.9)
        store.increment_edge_weight(b, a)     # 反向调用，应命中同一条
        assert store.get_edges(a)[0]["weight"] == pytest.approx(1.0)  # 0.9+0.1=1.0（封顶）

    def test_increment_edge_weight_only_existing(self):
        a, b = self._two_stories()
        # increment 不会凭空创建边
        store.increment_edge_weight(a, b)
        assert store.get_edges(a) == []

    def test_increment_edge_weight_custom_delta(self):
        a, b = self._two_stories()
        store.add_or_update_edge(a, b, 0.3)
        store.increment_edge_weight(a, b, delta=0.2)
        assert store.get_edges(a)[0]["weight"] == pytest.approx(0.5)

    def test_get_related_stories_bidirectional(self):
        a = store.add_story("a", "c", [], basis(0))
        b = store.add_story("b", "c", [], basis(1))
        store.add_or_update_edge(a, b, 0.8, "semantic")
        # 从 a 查到 b，从 b 查到 a
        rel_a = store.get_related_stories(a)
        assert [r["id"] for r in rel_a] == [b]
        rel_b = store.get_related_stories(b)
        assert [r["id"] for r in rel_b] == [a]
        assert rel_a[0]["weight"] == pytest.approx(0.8)


# ═══════════════════════════════════════════════
#  向量检索 search_by_vector：1 - dist²/2 换算
# ═══════════════════════════════════════════════

class TestVectorSearch:
    @pytest.mark.parametrize("cos", [0.9, 0.85, 0.6, 0.5])
    def test_similarity_conversion_matches_cosine(self, cos):
        """对 L2 归一化向量，search_by_vector 的 similarity = 1 - dist²/2 应等于余弦相似度。"""
        store.add_story("seed", "c", [], basis(0))
        query = with_cos(0, cos)
        results = store.search_by_vector(query, top_k=5)
        assert len(results) == 1
        assert results[0]["similarity"] == pytest.approx(cos, abs=2e-3)
        # distance 字段是真实 L2 距离（非平方）：sqrt(2 - 2*cos)
        expected_dist = (2 - 2 * cos) ** 0.5
        assert results[0]["distance"] == pytest.approx(expected_dist, abs=2e-3)

    def test_search_orders_by_similarity_desc(self):
        store.add_story("high", "c", [], basis(0))                       # cos 1.0
        store.add_story("mid", "c", [], with_cos(0, 0.7))                # cos 0.7
        store.add_story("low", "c", [], with_cos(0, 0.5))                # cos 0.5
        results = store.search_by_vector(basis(0), top_k=3)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)
        assert results[0]["title"] == "high"
        assert results[1]["title"] == "mid"
        assert results[2]["title"] == "low"

    def test_search_top_k_limit(self):
        for i in range(5):
            store.add_story(f"s{i}", "c", [], basis(i))   # 全部与 query(basis 0) 正交 -> cos 0
        results = store.search_by_vector(basis(0), top_k=2)
        assert len(results) == 2

    def test_search_empty_db(self):
        assert store.search_by_vector(basis(0), top_k=5) == []

    def test_search_numpy_zero_query_returns_empty(self):
        """numpy 路径对零向量 query 直接返回空（边界：永不传入，但守卫需成立）。"""
        store.add_story("a", "c", [], basis(0))
        assert store.search_by_vector_numpy([0.0] * config.EMBED_DIM, top_k=5) == []

    def test_vec0_matches_numpy_fallback(self):
        """vec0 路径与 numpy 暴力余弦路径结果一致（交叉验证 1 - dist²/2 换算正确）。"""
        store.add_story("a", "c", [], basis(0))
        store.add_story("b", "c", [], with_cos(0, 0.8))
        store.add_story("c", "c", [], with_cos(0, 0.55))
        query = with_cos(0, 0.9)
        v0 = {r["story_id"]: r["similarity"] for r in store.search_by_vector(query, top_k=5)}
        np_ = {r["story_id"]: r["similarity"] for r in store.search_by_vector_numpy(query, top_k=5)}
        assert set(v0) == set(np_)
        for sid in v0:
            assert v0[sid] == pytest.approx(np_[sid], abs=3e-3)


# ═══════════════════════════════════════════════
#  双写一致性：stories.embedding 与 story_vectors 同步
# ═══════════════════════════════════════════════

class TestDualWriteConsistency:
    def test_add_story_writes_both_locations(self):
        vec = [0.1, -0.2, 0.3] + [0.0] * (config.EMBED_DIM - 3)
        sid = store.add_story("t", "c", [], vec)
        # stories.embedding
        story_vec = store.get_story(sid)["embedding"]
        # story_vectors（vec0 索引）
        index_vec = vector_in_index(sid)
        assert index_vec is not None, "story_vectors 中应有该向量"
        assert len(story_vec) == config.EMBED_DIM
        assert np.allclose(story_vec, index_vec, atol=1e-6)
        assert np.allclose(story_vec, vec, atol=1e-6)

    def test_update_story_embedding_syncs_both(self):
        sid = store.add_story("t", "c", [], basis(0))
        old_index = vector_in_index(sid)
        new_vec = with_cos(0, 0.6)
        store.update_story(sid, embedding=new_vec)

        story_vec = store.get_story(sid)["embedding"]
        index_vec = vector_in_index(sid)
        assert index_vec is not None
        # 向量确已更新（与旧值不同）
        assert not np.allclose(index_vec, old_index, atol=1e-6)
        # 两处一致且等于新值
        assert np.allclose(story_vec, index_vec, atol=1e-6)
        assert np.allclose(story_vec, new_vec, atol=1e-6)
        assert store.get_story(sid)["version"] == 2

    def test_update_story_without_embedding_keeps_vector(self):
        sid = store.add_story("t", "c", [], basis(0))
        store.update_story(sid, title="只改标题")   # 不传 embedding
        assert vector_in_index(sid) is not None
        assert np.allclose(store.get_story(sid)["embedding"], basis(0), atol=1e-6)

    def test_update_story_replaces_not_duplicates(self):
        """更新向量应 delete+insert，story_vectors 中该 story 始终只有一行。"""
        sid = store.add_story("t", "c", [], basis(0))
        store.update_story(sid, embedding=with_cos(0, 0.5))
        store.update_story(sid, embedding=with_cos(0, 0.9))
        db = store.get_db()
        try:
            n = db.execute(
                "SELECT COUNT(*) FROM story_vectors WHERE story_id = ?", (sid,)
            ).fetchone()[0]
        finally:
            db.close()
        assert n == 1

    def test_delete_story_vector_removes_from_index_keeps_row(self):
        """分裂后父 story 向量从索引移除：vec0 行删除、embedding 置空，但 stories 行保留。"""
        sid = store.add_story("parent", "c", [], basis(0), source_session_ids=[1])
        assert vector_in_index(sid) is not None

        store.delete_story_vector(sid)

        # 1) 向量索引已移除
        assert vector_in_index(sid) is None
        # 2) 不再被 search_by_vector 命中
        assert store.search_by_vector(basis(0), top_k=5) == []
        # 3) 不再被 numpy 路径命中（embedding IS NULL 被跳过）
        assert store.search_by_vector_numpy(basis(0), top_k=5) == []
        # 4) stories 行仍在（保留谱系 / parent_id 引用）
        story = store.get_story(sid)
        assert story is not None
        assert story["title"] == "parent"
        assert story["embedding"] == []   # _row_to_story 把 NULL embedding 转为 []


# ═══════════════════════════════════════════════
#  统计 get_stats
# ═══════════════════════════════════════════════

class TestStats:
    def test_get_stats_counts(self):
        s1 = store.add_session("a", "r1", "p1")
        store.add_session("b", "r2", "p2")
        store.update_session_status(s1, "processed")

        root = store.add_story("root", "c", [], basis(0))
        child = store.add_story("child", "c", [], basis(1), parent_id=root)
        store.add_or_update_edge(root, child, 1.0, "parent_child")

        stats = store.get_stats()
        assert stats["sessions"] == 2
        assert stats["pending"] == 1
        assert stats["processed"] == 1
        assert stats["stories"] == 2
        assert stats["edges"] == 1
        assert stats["root_stories"] == 1   # parent_id IS NULL
        assert stats["child_stories"] == 1  # parent_id IS NOT NULL
        assert stats["profile"]["id"] == config.PROFILE_ID
        assert stats["sync_state"] == "local_only"


class TestProfileIdentityMigration:
    def test_init_db_rebuilds_legacy_fts_with_abstract_column(self):
        db = store.get_db(load_vector_extension=False)
        try:
            db.executescript(
                """DROP TRIGGER story_fts_insert;
                   DROP TRIGGER story_fts_delete;
                   DROP TRIGGER story_fts_update;
                   DROP TABLE story_fts;
                   CREATE VIRTUAL TABLE story_fts USING fts5(
                       title, content, keywords,
                       content='stories', content_rowid='id'
                   );"""
            )
            db.commit()
        finally:
            db.close()

        store.init_db()

        db = store.get_db(load_vector_extension=False)
        try:
            columns = {
                row["name"] for row in db.execute(
                    "PRAGMA table_info(story_fts)"
                ).fetchall()
            }
        finally:
            db.close()
        assert "abstract" in columns

    def test_init_db_backfills_existing_v01_rows(self):
        config.DB_PATH.unlink()
        legacy = sqlite3.connect(config.DB_PATH)
        legacy.executescript(
            """
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                problem_desc TEXT,
                code_snippets TEXT,
                conclusion TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                processed_at TEXT
            );
            CREATE TABLE stories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '[]',
                embedding BLOB,
                parent_id INTEGER,
                source_session_ids TEXT DEFAULT '[]',
                access_count INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                weight REAL DEFAULT 0.0,
                edge_type TEXT DEFAULT 'semantic',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(source_id, target_id)
            );
            INSERT INTO sessions (source, raw_content) VALUES ('legacy', 'raw');
            INSERT INTO stories (title, content) VALUES ('legacy', 'content');
            INSERT INTO edges (source_id, target_id) VALUES (1, 1);
            """
        )
        legacy.commit()
        legacy.close()

        store.init_db()

        db = store.get_db()
        try:
            for table in ("sessions", "stories", "edges"):
                row = db.execute(f"SELECT * FROM {table} WHERE id = 1").fetchone()
                uuid.UUID(row["global_id"])
                assert row["profile_id"] == config.PROFILE_ID
                assert row["sync_state"] == "local_only"
            legacy_session = db.execute(
                "SELECT context_json FROM sessions WHERE id = 1"
            ).fetchone()
            envelope = json.loads(legacy_session["context_json"])
            # A pre-E5 Session must remain explicit unknown; migration must not
            # mislabel the current machine/runtime as historical evidence.
            assert envelope["device"]["id"] is None
            assert envelope["runtime"]["kind"] == "unknown"
            assert envelope["provenance"]["runtime.kind"] == "unknown"
        finally:
            db.close()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_database_file_is_private(self):
        assert stat.S_IMODE(config.DB_PATH.stat().st_mode) == 0o600
