"""
存储层 — SQLite + sqlite-vec 向量存储，所有 CRUD 集中于此
"""
import json
import sqlite3
import logging
import os
import uuid
from typing import Optional

import sqlite_vec
import numpy as np

from . import config

logger = logging.getLogger(__name__)

_SCHEMA = """
-- 会话日志表（原始导入数据）
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id TEXT NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    sync_state TEXT NOT NULL DEFAULT 'local_only'
        CHECK(sync_state IN ('local_only', 'synced', 'pending', 'conflict', 'paused', 'error')),
    source TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    problem_desc TEXT,
    code_snippets TEXT,
    conclusion TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now')),
    processed_at TEXT
);

-- Story 表（结构化记忆单元）
CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id TEXT NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    sync_state TEXT NOT NULL DEFAULT 'local_only'
        CHECK(sync_state IN ('local_only', 'synced', 'pending', 'conflict', 'paused', 'error')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '[]',
    embedding BLOB,
    parent_id INTEGER,
    source_session_ids TEXT DEFAULT '[]',
    access_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (parent_id) REFERENCES stories(id)
);

-- 关联边表（带权重的关联网络）
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id TEXT NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    sync_state TEXT NOT NULL DEFAULT 'local_only'
        CHECK(sync_state IN ('local_only', 'synced', 'pending', 'conflict', 'paused', 'error')),
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    weight REAL DEFAULT 0.0,
    edge_type TEXT DEFAULT 'semantic',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES stories(id),
    FOREIGN KEY (target_id) REFERENCES stories(id),
    UNIQUE(source_id, target_id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_stories_parent ON stories(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
"""


def get_db() -> sqlite3.Connection:
    """获取数据库连接（每次调用创建新连接，用完即关）"""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(config.DB_PATH))
    if os.name != "nt":
        config.DB_PATH.chmod(0o600)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    # MCP server（agent 运行时 recall，会写 access_count/边权）与做梦周期 process
    # 可能并发写同一库；设 busy_timeout 让后到的写等待而非立刻报 database is locked。
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA foreign_keys=ON")
    # 加载 sqlite-vec 扩展
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def init_db():
    """初始化数据库 schema，并补齐 v0.2 Profile/同步预留字段。"""
    config.ensure_profile()
    db = get_db()
    try:
        db.executescript(_SCHEMA)
        _ensure_identity_columns(db)
        # 创建 sqlite-vec 虚拟表
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS story_vectors USING vec0(
                story_id INTEGER PRIMARY KEY,
                embedding FLOAT[{config.EMBED_DIM}]
            )
        """)
        db.commit()
        logger.info("数据库初始化完成: %s", config.DB_PATH)
    finally:
        db.close()


def _new_global_id() -> str:
    """生成不依赖路径、hostname 或数据库自增键的全局对象 ID。"""

    return str(uuid.uuid4())


def _ensure_identity_columns(db: sqlite3.Connection) -> None:
    """为已有 v0.1 数据库原地补齐 Profile、global_id 与 sync_state。

    完整的旧库迁移/备份由后续迁移流程负责；这里仅做幂等、无损的兼容列扩展，
    确保当前代码可以继续打开已有数据库。
    """

    for table in ("sessions", "stories", "edges"):
        columns = {
            row["name"] for row in db.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if "global_id" not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN global_id TEXT")
        if "profile_id" not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN profile_id TEXT")
        if "sync_state" not in columns:
            db.execute(
                f"ALTER TABLE {table} "
                "ADD COLUMN sync_state TEXT NOT NULL DEFAULT 'local_only'"
            )

        missing = db.execute(
            f"SELECT id FROM {table} WHERE global_id IS NULL OR global_id = ''"
        ).fetchall()
        for row in missing:
            db.execute(
                f"UPDATE {table} SET global_id = ? WHERE id = ?",
                (_new_global_id(), row["id"]),
            )
        db.execute(
            f"UPDATE {table} SET profile_id = ? "
            "WHERE profile_id IS NULL OR profile_id = ''",
            (config.PROFILE_ID,),
        )
        db.execute(
            f"UPDATE {table} SET sync_state = 'local_only' "
            "WHERE sync_state IS NULL OR sync_state = ''"
        )
        db.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_global_id "
            f"ON {table}(global_id)"
        )


# ═══════════════════════════════════════════════
#  健康检查辅助（供 doctor 使用）
# ═══════════════════════════════════════════════

def check_vec_extension() -> bool:
    """检测 sqlite-vec 扩展能否加载。

    用内存库探测，不触碰数据文件、无副作用。
    """
    try:
        probe = sqlite3.connect(":memory:")
        probe.enable_load_extension(True)
        sqlite_vec.load(probe)
        probe.close()
        return True
    except Exception:
        return False


def stories_table_exists() -> bool:
    """stories 表是否存在（判断 schema 是否已初始化）。"""
    if not config.DB_PATH.exists():
        return False
    db = sqlite3.connect(str(config.DB_PATH))
    try:
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stories'"
        ).fetchone()
        return row is not None
    finally:
        db.close()


def story_vectors_table_exists() -> bool:
    """story_vectors vec0 虚表是否存在。"""
    if not config.DB_PATH.exists():
        return False
    db = sqlite3.connect(str(config.DB_PATH))
    try:
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='story_vectors'"
        ).fetchone()
        return row is not None
    finally:
        db.close()


def vector_consistency() -> dict:
    """检查 stories.embedding 与 story_vectors 的双写一致性。

    embedding 双写约束：stories.embedding（BLOB）与 story_vectors（vec0 行）
    必须同有同无。返回::

        {
          "blob_count": int,     # stories.embedding 非空行数
          "vec_count": int,      # story_vectors 行数
          "missing_vec": [ids],  # 有 BLOB 但缺 vec0 行（需重建）
          "orphan_vec": [ids],   # 有 vec0 行但 BLOB 缺失/Story 不存在（需清除）
        }
    """
    db = get_db()
    try:
        blob_count = db.execute(
            "SELECT COUNT(*) FROM stories WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        vec_count = db.execute("SELECT COUNT(*) FROM story_vectors").fetchone()[0]
        missing_vec = [r[0] for r in db.execute(
            """SELECT s.id FROM stories s
               WHERE s.embedding IS NOT NULL
                 AND s.id NOT IN (SELECT story_id FROM story_vectors)"""
        ).fetchall()]
        orphan_vec = [r[0] for r in db.execute(
            """SELECT v.story_id FROM story_vectors v
               LEFT JOIN stories s ON s.id = v.story_id
               WHERE s.id IS NULL OR s.embedding IS NULL"""
        ).fetchall()]
        return {
            "blob_count": blob_count,
            "vec_count": vec_count,
            "missing_vec": missing_vec,
            "orphan_vec": orphan_vec,
        }
    finally:
        db.close()


def repair_vector_consistency() -> dict:
    """修复向量双写不一致，返回 ``{"rebuilt", "cleared", "failed"}``。

    - missing_vec（有 BLOB 无 vec0 行）：用现有 BLOB 重建 story_vectors 行；
    - orphan_vec（有 vec0 行但 BLOB 缺失/Story 不存在）：删除孤立的 vec0 行。
    """
    report = vector_consistency()
    rebuilt = 0
    cleared = 0
    failed = []
    db = get_db()
    try:
        for sid in report["missing_vec"]:
            row = db.execute(
                "SELECT embedding FROM stories WHERE id = ?", (sid,)
            ).fetchone()
            if row is None or row["embedding"] is None:
                continue
            try:
                db.execute(
                    "INSERT INTO story_vectors (story_id, embedding) VALUES (?, ?)",
                    (sid, row["embedding"]),
                )
                rebuilt += 1
            except Exception as e:
                failed.append((sid, f"rebuild: {e}"))
        for sid in report["orphan_vec"]:
            try:
                db.execute("DELETE FROM story_vectors WHERE story_id = ?", (sid,))
                cleared += 1
            except Exception as e:
                failed.append((sid, f"clear: {e}"))
        db.commit()
    finally:
        db.close()
    return {"rebuilt": rebuilt, "cleared": cleared, "failed": failed}


# ═══════════════════════════════════════════════
#  Session CRUD
# ═══════════════════════════════════════════════

def add_session(source: str, raw_content: str, problem_desc: str = "",
                code_snippets: str = "[]", conclusion: str = "") -> int:
    """插入一条会话日志，返回 session_id"""
    db = get_db()
    try:
        cur = db.execute(
            """INSERT INTO sessions (
                   global_id, profile_id, sync_state,
                   source, raw_content, problem_desc, code_snippets, conclusion
               ) VALUES (?, ?, 'local_only', ?, ?, ?, ?, ?)""",
            (
                _new_global_id(), config.PROFILE_ID,
                source, raw_content, problem_desc, code_snippets, conclusion,
            )
        )
        db.commit()
        return cur.lastrowid
    finally:
        db.close()


def get_pending_sessions(limit: int = 0) -> list[sqlite3.Row]:
    """获取所有 pending 状态的会话"""
    db = get_db()
    try:
        sql = "SELECT * FROM sessions WHERE status = 'pending' ORDER BY id"
        if limit > 0:
            sql += f" LIMIT {limit}"
        return db.execute(sql).fetchall()
    finally:
        db.close()


def update_session_status(session_id: int, status: str):
    """更新会话状态"""
    db = get_db()
    try:
        db.execute(
            "UPDATE sessions SET status = ?, processed_at = datetime('now') WHERE id = ?",
            (status, session_id)
        )
        db.commit()
    finally:
        db.close()


def get_session(session_id: int) -> Optional[sqlite3.Row]:
    db = get_db()
    try:
        return db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    finally:
        db.close()


def count_sessions() -> int:
    db = get_db()
    try:
        return db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        db.close()


# ═══════════════════════════════════════════════
#  Story CRUD
# ═══════════════════════════════════════════════

def add_story(title: str, content: str, keywords: list[str],
              embedding: list[float], parent_id: int = None,
              source_session_ids: list[int] = None) -> int:
    """新建 story，同时写入向量表，返回 story_id"""
    db = get_db()
    try:
        emb_blob = np.array(embedding, dtype=np.float32).tobytes()
        cur = db.execute(
            """INSERT INTO stories (
                   global_id, profile_id, sync_state,
                   title, content, keywords, embedding, parent_id, source_session_ids
               ) VALUES (?, ?, 'local_only', ?, ?, ?, ?, ?, ?)""",
            (_new_global_id(), config.PROFILE_ID,
             title, content, json.dumps(keywords, ensure_ascii=False),
             emb_blob, parent_id, json.dumps(source_session_ids or []))
        )
        story_id = cur.lastrowid
        # 写入向量虚拟表
        db.execute(
            "INSERT INTO story_vectors (story_id, embedding) VALUES (?, ?)",
            (story_id, emb_blob)
        )
        db.commit()
        logger.info("新建 story #%d: %s", story_id, title)
        return story_id
    finally:
        db.close()


def update_story(story_id: int, title: str = None, content: str = None,
                 keywords: list[str] = None, embedding: list[float] = None):
    """更新 story（传 None 的字段不更新），同时更新向量表"""
    db = get_db()
    try:
        sets = []
        params = []
        if title is not None:
            sets.append("title = ?")
            params.append(title)
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if keywords is not None:
            sets.append("keywords = ?")
            params.append(json.dumps(keywords, ensure_ascii=False))
        if embedding is not None:
            emb_blob = np.array(embedding, dtype=np.float32).tobytes()
            sets.append("embedding = ?")
            params.append(emb_blob)
        sets.append("version = version + 1")
        sets.append("updated_at = datetime('now')")
        params.append(story_id)

        db.execute(f"UPDATE stories SET {', '.join(sets)} WHERE id = ?", params)

        # 如果向量更新了，同步到虚拟表
        if embedding is not None:
            db.execute(
                "DELETE FROM story_vectors WHERE story_id = ?", (story_id,)
            )
            db.execute(
                "INSERT INTO story_vectors (story_id, embedding) VALUES (?, ?)",
                (story_id, emb_blob)
            )
        db.commit()
        logger.info("更新 story #%d", story_id)
    finally:
        db.close()


def delete_story_vector(story_id: int):
    """从向量索引移除 story 的向量，使其不再参与检索。

    分裂后父 story 被拆分为子 story 时调用：保留 stories 行（谱系/parent_id），
    但清空 embedding BLOB 并删除 story_vectors 行，使 vec0 与 numpy 两条检索路径都跳过它。
    """
    db = get_db()
    try:
        db.execute("DELETE FROM story_vectors WHERE story_id = ?", (story_id,))
        db.execute("UPDATE stories SET embedding = NULL WHERE id = ?", (story_id,))
        db.commit()
        logger.info("已从检索索引移除 story #%d 的向量", story_id)
    finally:
        db.close()


def get_story(story_id: int) -> Optional[dict]:
    db = get_db()
    try:
        row = db.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        if row:
            return _row_to_story(row)
        return None
    finally:
        db.close()


def get_all_stories() -> list[dict]:
    db = get_db()
    try:
        rows = db.execute("SELECT * FROM stories ORDER BY id").fetchall()
        return [_row_to_story(r) for r in rows]
    finally:
        db.close()


def update_story_raw_sessions(story_id: int, session_ids: list[int]):
    """更新 story 的 source_session_ids"""
    db = get_db()
    try:
        db.execute(
            "UPDATE stories SET source_session_ids = ?, updated_at = datetime('now') WHERE id = ?",
            (json.dumps(session_ids), story_id)
        )
        db.commit()
    finally:
        db.close()


def increment_access_count(story_id: int):
    """检索命中时调用"""
    db = get_db()
    try:
        db.execute("UPDATE stories SET access_count = access_count + 1 WHERE id = ?", (story_id,))
        db.commit()
    finally:
        db.close()


def count_stories() -> int:
    db = get_db()
    try:
        return db.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
    finally:
        db.close()


# ═══════════════════════════════════════════════
#  Edge CRUD（关联网络）
# ═══════════════════════════════════════════════

def _edge_pair(a: int, b: int) -> tuple[int, int]:
    """规范化无向边端点顺序为 (min, max)，避免 (A,B) 与 (B,A) 产生重复行。"""
    return (a, b) if a <= b else (b, a)


def add_or_update_edge(source_id: int, target_id: int, weight: float,
                       edge_type: str = "semantic"):
    """添加或更新一条关联边（无向，方向无关）。已存在则取 max(weight)。"""
    source_id, target_id = _edge_pair(source_id, target_id)
    db = get_db()
    try:
        existing = db.execute(
            "SELECT * FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id)
        ).fetchone()
        if existing:
            new_weight = min(max(existing["weight"], weight), config.WEIGHT_MAX)
            db.execute(
                "UPDATE edges SET weight = ? WHERE source_id = ? AND target_id = ?",
                (new_weight, source_id, target_id)
            )
        else:
            db.execute(
                """INSERT INTO edges (
                       global_id, profile_id, sync_state,
                       source_id, target_id, weight, edge_type
                   ) VALUES (?, ?, 'local_only', ?, ?, ?, ?)""",
                (
                    _new_global_id(), config.PROFILE_ID,
                    source_id, target_id, weight, edge_type,
                )
            )
        db.commit()
    finally:
        db.close()


def increment_edge_weight(source_id: int, target_id: int, delta: float = None):
    """共同调用时提升权重（无向，方向无关）。"""
    if delta is None:
        delta = config.WEIGHT_INCREMENT
    source_id, target_id = _edge_pair(source_id, target_id)
    db = get_db()
    try:
        row = db.execute(
            "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id)
        ).fetchone()
        if row:
            new_weight = min(row["weight"] + delta, config.WEIGHT_MAX)
            db.execute(
                "UPDATE edges SET weight = ? WHERE source_id = ? AND target_id = ?",
                (new_weight, source_id, target_id)
            )
            db.commit()
    finally:
        db.close()


def get_edges(story_id: int) -> list[dict]:
    """获取 story 的所有关联边（双向），按权重降序"""
    db = get_db()
    try:
        rows = db.execute(
            """SELECT e.*, CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END AS related_id
               FROM edges e
               WHERE e.source_id = ? OR e.target_id = ?
               ORDER BY e.weight DESC""",
            (story_id, story_id, story_id)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


def get_related_stories(story_id: int, limit: int = 5) -> list[dict]:
    """获取关联 story 列表（含 story 详情），按权重降序"""
    db = get_db()
    try:
        rows = db.execute(
            """SELECT s.*, e.weight, e.edge_type
               FROM edges e
               JOIN stories s ON s.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
               WHERE e.source_id = ? OR e.target_id = ?
               ORDER BY e.weight DESC
               LIMIT ?""",
            (story_id, story_id, story_id, limit)
        ).fetchall()
        return [_row_to_story(r) for r in rows]
    finally:
        db.close()


# ═══════════════════════════════════════════════
#  向量检索
# ═══════════════════════════════════════════════

def search_by_vector(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """用 sqlite-vec 做向量近邻搜索，返回 [{story_id, distance, similarity, ...story}]"""
    db = get_db()
    try:
        emb_blob = np.array(query_embedding, dtype=np.float32).tobytes()
        rows = db.execute(
            """SELECT v.story_id, v.distance, s.title, s.content, s.keywords
               FROM story_vectors v
               JOIN stories s ON s.id = v.story_id
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (emb_blob, top_k)
        ).fetchall()
        results = []
        for r in rows:
            # sqlite-vec distance 是实际 L2 距离（非平方）。
            # 对归一化向量：L2² = 2 - 2*cosine_sim，故 cosine_sim = 1 - L2²/2（精确）。
            # embeddings.embed 已做 L2 归一化，此换算成立。
            sim = max(0.0, 1 - (r["distance"] ** 2) / 2)
            results.append({
                "story_id": r["story_id"],
                "title": r["title"],
                "content": r["content"],
                "keywords": json.loads(r["keywords"]),
                "similarity": round(sim, 4),
                "distance": r["distance"],
            })
        return results
    finally:
        db.close()


def search_by_vector_numpy(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """numpy 暴力余弦相似度搜索（fallback）"""
    db = get_db()
    try:
        rows = db.execute(
            "SELECT id, title, content, keywords, embedding FROM stories WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        results = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
            results.append({
                "story_id": r["id"],
                "title": r["title"],
                "content": r["content"],
                "keywords": json.loads(r["keywords"]),
                "similarity": round(sim, 4),
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    finally:
        db.close()


# ═══════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════

def _row_to_story(row: sqlite3.Row) -> dict:
    """将 sqlite3.Row 转为 story dict，embedding 转回 list"""
    d = dict(row)
    if d.get("keywords"):
        d["keywords"] = json.loads(d["keywords"])
    if d.get("source_session_ids"):
        d["source_session_ids"] = json.loads(d["source_session_ids"])
    if d.get("embedding"):
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).tolist()
    else:
        d["embedding"] = []
    return d


def get_stats() -> dict:
    """获取系统统计信息"""
    db = get_db()
    try:
        stats = {
            "sessions": db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "pending": db.execute("SELECT COUNT(*) FROM sessions WHERE status='pending'").fetchone()[0],
            "processed": db.execute("SELECT COUNT(*) FROM sessions WHERE status='processed'").fetchone()[0],
            "stories": db.execute("SELECT COUNT(*) FROM stories").fetchone()[0],
            "edges": db.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
            "root_stories": db.execute("SELECT COUNT(*) FROM stories WHERE parent_id IS NULL").fetchone()[0],
            "child_stories": db.execute("SELECT COUNT(*) FROM stories WHERE parent_id IS NOT NULL").fetchone()[0],
            "profile": {
                "id": config.PROFILE_ID,
                "display_name": config.ACTIVE_PROFILE.display_name,
                "mode": config.PROFILE_MODE,
            },
            "sync_state": config.SYNC_STATE,
        }
        return stats
    finally:
        db.close()
