"""
存储层 — SQLite + sqlite-vec 向量存储，所有 CRUD 集中于此
"""
import json
import sqlite3
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import sqlite_vec
import numpy as np

from . import config
from . import context as context_module
from . import story_v2

logger = logging.getLogger(__name__)

LEGACY_EMBED_VERSION = "story-v1-unversioned"
LEGACY_EMBED_REPRESENTATION = "legacy"

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
    device_id TEXT,
    agent_installation_id TEXT,
    workspace_id TEXT,
    runtime_json TEXT NOT NULL DEFAULT '{}',
    external_session_hash TEXT,
    context_json TEXT NOT NULL DEFAULT '{}',
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
    abstract TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    sources_json TEXT NOT NULL DEFAULT '[]',
    keywords TEXT NOT NULL DEFAULT '[]',
    embedding BLOB,
    embedding_model TEXT,
    embedding_version TEXT,
    embedding_content_hash TEXT,
    embedding_status TEXT NOT NULL DEFAULT 'active'
        CHECK(embedding_status IN ('active', 'stale', 'pending', 'failed', 'archived')),
    parent_id INTEGER,
    source_session_ids TEXT DEFAULT '[]',
    applicability_json TEXT NOT NULL DEFAULT '{}',
    environment_summary_json TEXT NOT NULL DEFAULT '[]',
    access_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (parent_id) REFERENCES stories(id)
);

-- ContextEnvelope identity dimensions. Sensitive local details stay in JSON
-- hashes/aliases; absolute paths, hostnames and repository URLs are never keys.
CREATE TABLE IF NOT EXISTS devices (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    os_family TEXT,
    os_version TEXT,
    arch TEXT,
    display_name TEXT,
    local_metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS agent_installations (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    tool_type TEXT NOT NULL,
    tool_version TEXT,
    integration_mode TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    repo_fingerprint TEXT,
    label TEXT,
    local_path_alias TEXT,
    created_at TEXT DEFAULT (datetime('now'))
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
    directed INTEGER NOT NULL DEFAULT 0 CHECK(directed IN (0, 1)),
    provenance_json TEXT NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1,
    observations INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_reinforced_at TEXT,
    deleted_at TEXT,
    FOREIGN KEY (source_id) REFERENCES stories(id),
    FOREIGN KEY (target_id) REFERENCES stories(id),
    UNIQUE(source_id, target_id, edge_type)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_stories_parent ON stories(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_devices_profile ON devices(profile_id);
CREATE INDEX IF NOT EXISTS idx_agent_installations_profile ON agent_installations(profile_id);
CREATE INDEX IF NOT EXISTS idx_workspaces_profile ON workspaces(profile_id);

-- Immutable Story version/event snapshots.  A merge, split or update never
-- destroys the previous material; provenance can be audited without replaying
-- mutable rows.
CREATE TABLE IF NOT EXISTS story_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id INTEGER NOT NULL,
    story_version INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    source_session_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (story_id) REFERENCES stories(id),
    UNIQUE(story_id, story_version)
);
CREATE INDEX IF NOT EXISTS idx_story_revisions_story
    ON story_revisions(story_id, story_version);

-- ``story_vectors`` is the active serving index.  A new model/version is built
-- here first; activation copies a complete shadow set into the active index in
-- one transaction, so failed or partial backfills never disturb recall.
CREATE TABLE IF NOT EXISTS story_embedding_backfill (
    story_id INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    representation TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding BLOB,
    status TEXT NOT NULL CHECK(status IN ('ready', 'failed')),
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY(story_id, embedding_version),
    FOREIGN KEY (story_id) REFERENCES stories(id)
);

CREATE TABLE IF NOT EXISTS embedding_index_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    active_model TEXT NOT NULL,
    active_version TEXT NOT NULL,
    active_representation TEXT NOT NULL,
    target_model TEXT,
    target_version TEXT,
    target_representation TEXT,
    backfill_status TEXT NOT NULL DEFAULT 'idle'
        CHECK(backfill_status IN ('idle', 'running', 'failed', 'ready')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- 查询缓存只依赖可检索内容版本；访问计数与共同召回反馈不会推进该版本。
CREATE TABLE IF NOT EXISTS query_state (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO query_state(key, value) VALUES ('index_version', 1);
"""


def get_db(
    db_path: str | Path | None = None, *, load_vector_extension: bool = True
) -> sqlite3.Connection:
    """获取数据库连接（每次调用创建新连接，用完即关）"""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    if os.name != "nt":
        path.chmod(0o600)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    # MCP server（agent 运行时 recall，会写 access_count/边权）与做梦周期 process
    # 可能并发写同一库；设 busy_timeout 让后到的写等待而非立刻报 database is locked。
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA foreign_keys=ON")
    if load_vector_extension:
        # 加载 sqlite-vec 扩展
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    return db


def init_db():
    """初始化数据库 schema，并幂等补齐 Profile 与 ContextEnvelope 字段。"""
    config.ensure_profile()
    db = get_db()
    try:
        db.executescript(_SCHEMA)
        _ensure_identity_columns(db)
        _ensure_memory_graph_schema(db)
        _ensure_context_columns(db)
        _ensure_story_v2_columns(db)
        _ensure_fts_index(db)
        # 创建 sqlite-vec 虚拟表
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS story_vectors USING vec0(
                story_id INTEGER PRIMARY KEY,
                embedding FLOAT[{config.EMBED_DIM}]
            )
        """)
        legacy_vector = db.execute(
            """SELECT 1 FROM stories
               WHERE embedding IS NOT NULL AND embedding_version IS NULL
               LIMIT 1"""
        ).fetchone()
        initial_version = (
            LEGACY_EMBED_VERSION if legacy_vector else config.EMBED_VERSION
        )
        initial_representation = (
            LEGACY_EMBED_REPRESENTATION
            if legacy_vector else config.EMBED_REPRESENTATION
        )
        db.execute(
            """INSERT OR IGNORE INTO embedding_index_state (
                   id, active_model, active_version, active_representation
               ) VALUES (1, ?, ?, ?)""",
            (config.EMBED_MODEL, initial_version, initial_representation),
        )
        db.commit()
        logger.info("数据库初始化完成: %s", config.DB_PATH)
    finally:
        db.close()


def _ensure_fts_index(db: sqlite3.Connection) -> None:
    """创建与 ``stories`` 外部内容表同步的 FTS5 索引。

    旧数据库首次升级时执行一次 ``rebuild``；后续由触发器增量同步，避免每次
    ``init_db`` 都扫描全部 Story。
    """

    existed = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='story_fts'"
    ).fetchone() is not None
    if existed:
        fts_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(story_fts)").fetchall()
        }
        if "abstract" not in fts_columns:
            db.executescript(
                """DROP TRIGGER IF EXISTS story_fts_insert;
                   DROP TRIGGER IF EXISTS story_fts_delete;
                   DROP TRIGGER IF EXISTS story_fts_update;
                   DROP TABLE story_fts;"""
            )
            existed = False
    try:
        db.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS story_fts USING fts5(
                   title, abstract, content, keywords,
                   content='stories', content_rowid='id', tokenize='unicode61'
               )"""
        )
    except sqlite3.OperationalError:
        # 少数 SQLite 构建未启用 FTS5；关键词 LIKE fallback 仍可工作。
        logger.warning("SQLite FTS5 不可用，将使用关键词 fallback")
        return
    db.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS story_fts_insert AFTER INSERT ON stories BEGIN
            INSERT INTO story_fts(rowid, title, abstract, content, keywords)
            VALUES (new.id, new.title, new.abstract, new.content, new.keywords);
        END;
        CREATE TRIGGER IF NOT EXISTS story_fts_delete AFTER DELETE ON stories BEGIN
            INSERT INTO story_fts(story_fts, rowid, title, abstract, content, keywords)
            VALUES ('delete', old.id, old.title, old.abstract, old.content, old.keywords);
        END;
        CREATE TRIGGER IF NOT EXISTS story_fts_update
        AFTER UPDATE OF title, abstract, content, keywords ON stories BEGIN
            INSERT INTO story_fts(story_fts, rowid, title, abstract, content, keywords)
            VALUES ('delete', old.id, old.title, old.abstract, old.content, old.keywords);
            INSERT INTO story_fts(rowid, title, abstract, content, keywords)
            VALUES (new.id, new.title, new.abstract, new.content, new.keywords);
        END;
        """
    )
    if not existed:
        db.execute("INSERT INTO story_fts(story_fts) VALUES ('rebuild')")


def get_index_version(db_path: str | Path | None = None) -> int:
    """返回当前可检索内容版本，供跨进程缓存 key 使用。"""

    path = Path(db_path) if db_path is not None else config.DB_PATH
    db = sqlite3.connect(str(path), timeout=0.05)
    try:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT value FROM query_state WHERE key = 'index_version'"
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        db.close()


def _bump_index_version(db: sqlite3.Connection) -> int:
    db.execute(
        "UPDATE query_state SET value = value + 1 WHERE key = 'index_version'"
    )
    row = db.execute(
        "SELECT value FROM query_state WHERE key = 'index_version'"
    ).fetchone()
    return int(row[0])


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


_MEMORY_EDGE_TABLE_SQL = """
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id TEXT NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    sync_state TEXT NOT NULL DEFAULT 'local_only'
        CHECK(sync_state IN ('local_only', 'synced', 'pending', 'conflict', 'paused', 'error')),
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    edge_type TEXT NOT NULL DEFAULT 'semantic',
    directed INTEGER NOT NULL DEFAULT 0 CHECK(directed IN (0, 1)),
    provenance_json TEXT NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1,
    observations INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_reinforced_at TEXT,
    deleted_at TEXT,
    FOREIGN KEY (source_id) REFERENCES stories(id),
    FOREIGN KEY (target_id) REFERENCES stories(id),
    UNIQUE(source_id, target_id, edge_type)
)
"""


def _ensure_memory_graph_schema(db: sqlite3.Connection) -> None:
    """Idempotently migrate the v0.1 undirected edge table in place.

    SQLite cannot replace ``UNIQUE(source_id, target_id)`` with a typed unique
    key using ``ALTER TABLE``.  Rebuilding the small edge table is therefore the
    only safe way to allow more than one memory relation between the same pair.
    Story rows and edge global IDs are retained; legacy provenance is made
    explicit instead of being guessed from the current machine or session.
    """

    columns = {
        row["name"] for row in db.execute("PRAGMA table_info(edges)").fetchall()
    }
    required = {
        "directed", "provenance_json", "version", "observations", "updated_at",
        "last_reinforced_at", "deleted_at",
    }
    table_row = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='edges'"
    ).fetchone()
    compact_sql = re.sub(r"\s+", "", (table_row["sql"] or "").upper())
    typed_unique = "UNIQUE(SOURCE_ID,TARGET_ID,EDGE_TYPE)" in compact_sql
    if required.issubset(columns) and typed_unique:
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_active_source "
            "ON edges(source_id, deleted_at, weight DESC)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_active_target "
            "ON edges(target_id, deleted_at, weight DESC)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_type_active "
            "ON edges(edge_type, deleted_at)"
        )
        return

    legacy_table = "edges_graph_legacy_v1"
    if db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (legacy_table,),
    ).fetchone():
        raise RuntimeError("检测到未完成的 Memory Graph 边迁移，请先恢复数据库备份")

    rows = db.execute(
        """SELECT e.*, source.parent_id AS source_parent_id,
                         target.parent_id AS target_parent_id
           FROM edges e
           LEFT JOIN stories source ON source.id = e.source_id
           LEFT JOIN stories target ON target.id = e.target_id
           ORDER BY e.id"""
    ).fetchall()
    db.execute("DROP INDEX IF EXISTS idx_edges_source")
    db.execute("DROP INDEX IF EXISTS idx_edges_target")
    db.execute("DROP INDEX IF EXISTS idx_edges_active_source")
    db.execute("DROP INDEX IF EXISTS idx_edges_active_target")
    db.execute("DROP INDEX IF EXISTS idx_edges_type_active")
    db.execute(f"ALTER TABLE edges RENAME TO {legacy_table}")
    db.execute(_MEMORY_EDGE_TABLE_SQL)

    for row in rows:
        keys = set(row.keys())
        edge_type = _normalize_edge_type(row["edge_type"])
        source_id, target_id = int(row["source_id"]), int(row["target_id"])
        directed = (
            bool(row["directed"])
            if "directed" in keys
            else edge_type in config.DIRECTED_EDGE_TYPES
        )
        # v0.1 normalised every relation as an undirected pair.  Parent/child is
        # the one directed relation whose intended orientation can be recovered
        # losslessly from ``stories.parent_id``.
        if edge_type == "parent_child" and row["source_parent_id"] == target_id:
            source_id, target_id = target_id, source_id
        if not directed:
            source_id, target_id = _edge_pair(source_id, target_id)

        provenance = _json_load(
            row["provenance_json"] if "provenance_json" in keys else None,
            {},
        )
        if not isinstance(provenance, dict) or not provenance:
            provenance = {
                "source": "legacy_migration",
                "original_edge_type": row["edge_type"] or "semantic",
            }
        created_at = (
            row["created_at"] if "created_at" in keys and row["created_at"]
            else datetime.now(UTC).isoformat()
        )
        db.execute(
            """INSERT INTO edges (
                   id, global_id, profile_id, sync_state, source_id, target_id,
                   weight, edge_type, directed, provenance_json, version,
                   observations, created_at, updated_at, last_reinforced_at,
                   deleted_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["id"], row["global_id"], row["profile_id"],
                row["sync_state"], source_id, target_id,
                max(0.0, min(float(row["weight"] or 0.0), config.WEIGHT_MAX)),
                edge_type, int(directed),
                json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                int(row["version"] if "version" in keys else 1),
                int(row["observations"] if "observations" in keys else 0),
                created_at,
                row["updated_at"] if "updated_at" in keys else created_at,
                row["last_reinforced_at"] if "last_reinforced_at" in keys else None,
                row["deleted_at"] if "deleted_at" in keys else None,
            ),
        )

    db.execute(f"DROP TABLE {legacy_table}")
    db.execute("CREATE INDEX idx_edges_source ON edges(source_id)")
    db.execute("CREATE INDEX idx_edges_target ON edges(target_id)")
    db.execute(
        "CREATE INDEX idx_edges_active_source "
        "ON edges(source_id, deleted_at, weight DESC)"
    )
    db.execute(
        "CREATE INDEX idx_edges_active_target "
        "ON edges(target_id, deleted_at, weight DESC)"
    )
    db.execute(
        "CREATE INDEX idx_edges_type_active ON edges(edge_type, deleted_at)"
    )


def _ensure_context_columns(db: sqlite3.Connection) -> None:
    """Backfill v0.1/v0.2 databases with canonical, explicit ContextEnvelope data."""

    additions = {
        "sessions": {
            "device_id": "TEXT",
            "agent_installation_id": "TEXT",
            "workspace_id": "TEXT",
            "runtime_json": "TEXT NOT NULL DEFAULT '{}'",
            "external_session_hash": "TEXT",
            "context_json": "TEXT NOT NULL DEFAULT '{}'",
        },
        "stories": {
            "applicability_json": "TEXT NOT NULL DEFAULT '{}'",
            "environment_summary_json": "TEXT NOT NULL DEFAULT '[]'",
        },
    }
    for table, columns_to_add in additions.items():
        columns = {
            row["name"] for row in db.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, declaration in columns_to_add.items():
            if name not in columns:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
    db.execute(
        """CREATE INDEX IF NOT EXISTS idx_sessions_context ON sessions(
               profile_id, device_id, agent_installation_id, workspace_id
           )"""
    )

    sessions = db.execute(
        """SELECT * FROM sessions
           WHERE context_json IS NULL OR trim(context_json) IN ('', '{}')
           ORDER BY id"""
    ).fetchall()
    for row in sessions:
        try:
            raw_context = json.loads(row["context_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            raw_context = {}
        if raw_context:
            envelope = context_module.normalize_envelope(
                raw_context,
                profile_id=row["profile_id"],
                session_id=row["global_id"],
                source=row["source"],
                captured_at=row["created_at"],
            )
        else:
            envelope = context_module.unknown_envelope(
                profile_id=row["profile_id"],
                session_id=row["global_id"],
                source=row["source"],
                captured_at=row["created_at"],
            )
        _upsert_context_dimensions(db, envelope)
        db.execute(
            """UPDATE sessions
               SET device_id = ?, agent_installation_id = ?, workspace_id = ?,
                   runtime_json = ?, external_session_hash = ?, context_json = ?
               WHERE id = ?""",
            (
                envelope["device"]["id"],
                envelope["tool"]["installation_id"],
                envelope["workspace"]["id"],
                json.dumps(envelope["runtime"], ensure_ascii=False, sort_keys=True),
                envelope["session"]["external_session_hash"],
                json.dumps(envelope, ensure_ascii=False, sort_keys=True),
                row["id"],
            ),
        )

    stories = db.execute(
        """SELECT id, source_session_ids, applicability_json, environment_summary_json
           FROM stories
           WHERE applicability_json IS NULL OR trim(applicability_json) IN ('', '{}')
              OR (trim(COALESCE(environment_summary_json, '')) IN ('', '[]')
                  AND trim(COALESCE(source_session_ids, '')) NOT IN ('', '[]'))"""
    ).fetchall()
    for row in stories:
        applicability = context_module.normalize_applicability(row["applicability_json"])
        try:
            source_ids = json.loads(row["source_session_ids"] or "[]")
        except (json.JSONDecodeError, TypeError):
            source_ids = []
        environments = _environments_for_sessions(db, source_ids)
        try:
            existing = json.loads(row["environment_summary_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            existing = []
        environments = context_module.merge_environments(existing, environments)
        db.execute(
            """UPDATE stories
               SET applicability_json = ?, environment_summary_json = ?
               WHERE id = ?""",
            (
                json.dumps(applicability, ensure_ascii=False, sort_keys=True),
                json.dumps(environments, ensure_ascii=False, sort_keys=True),
                row["id"],
            ),
        )


def _json_load(value, fallback):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value) if value not in (None, "") else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


def _ensure_story_v2_columns(db: sqlite3.Connection) -> None:
    """Idempotently upgrade v0.1/v0.2 Story rows to the v2 contract."""

    additions = {
        "abstract": "TEXT NOT NULL DEFAULT ''",
        "detail_json": "TEXT NOT NULL DEFAULT '{}'",
        "sources_json": "TEXT NOT NULL DEFAULT '[]'",
        "embedding_model": "TEXT",
        "embedding_version": "TEXT",
        "embedding_content_hash": "TEXT",
        "embedding_status": "TEXT NOT NULL DEFAULT 'active'",
    }
    original_columns = {
        row["name"] for row in db.execute("PRAGMA table_info(stories)").fetchall()
    }
    for name, declaration in additions.items():
        if name not in original_columns:
            db.execute(f"ALTER TABLE stories ADD COLUMN {name} {declaration}")

    structure_was_legacy = not {
        "abstract", "detail_json", "sources_json"
    }.issubset(original_columns)
    metadata_was_legacy = not {
        "embedding_model", "embedding_version", "embedding_content_hash",
        "embedding_status",
    }.issubset(original_columns)

    rows = db.execute("SELECT * FROM stories ORDER BY id").fetchall()
    for row in rows:
        source_ids = _json_load(row["source_session_ids"], [])
        payload = story_v2.normalize_story_payload(
            {
                "title": row["title"],
                "abstract": row["abstract"],
                "content": row["content"],
                "detail": _json_load(row["detail_json"], {}),
                "sources": _json_load(row["sources_json"], []),
                "applicability": _json_load(row["applicability_json"], {}),
                "keywords": _json_load(row["keywords"], []),
            },
            source_session_ids=source_ids,
        )
        if structure_was_legacy:
            db.execute(
                """UPDATE stories SET abstract = ?, detail_json = ?, sources_json = ?
                   WHERE id = ?""",
                (
                    payload["abstract"],
                    json.dumps(
                        payload["detail"], ensure_ascii=False, sort_keys=True
                    ),
                    json.dumps(
                        payload["sources"], ensure_ascii=False, sort_keys=True
                    ),
                    row["id"],
                ),
            )
        if metadata_was_legacy:
            # An unversioned v0.1 BLOB has no trustworthy model/input/hash
            # provenance.  Keep it available for an explicit legacy serving
            # window, but require shadow backfill before claiming v2 metadata.
            existing_model = (
                row["embedding_model"]
                if "embedding_model" in original_columns else None
            )
            existing_version = (
                row["embedding_version"]
                if "embedding_version" in original_columns else None
            )
            existing_hash = (
                row["embedding_content_hash"]
                if "embedding_content_hash" in original_columns else None
            )
            existing_status = (
                row["embedding_status"]
                if "embedding_status" in original_columns else None
            )
            provenance_complete = all(
                (existing_model, existing_version, existing_hash)
            )
            if row["embedding"] is None:
                migrated_status = "archived"
            elif existing_status in {"stale", "failed", "archived"}:
                migrated_status = existing_status
            elif provenance_complete:
                migrated_status = "active"
            else:
                migrated_status = "stale"
            db.execute(
                """UPDATE stories SET embedding_model = ?,
                          embedding_version = ?,
                          embedding_content_hash = ?,
                          embedding_status = ?
                   WHERE id = ?""",
                (
                    existing_model, existing_version, existing_hash,
                    migrated_status, row["id"],
                ),
            )
        migrated = db.execute(
            "SELECT * FROM stories WHERE id = ?", (row["id"],)
        ).fetchone()
        revision_exists = db.execute(
            "SELECT 1 FROM story_revisions WHERE story_id = ? AND story_version = ?",
            (row["id"], row["version"]),
        ).fetchone()
        if revision_exists is None:
            db.execute(
                """INSERT INTO story_revisions (
                       story_id, story_version, event_type, snapshot_json,
                       source_session_ids
                   ) VALUES (?, ?, 'migrate', ?, ?)""",
                (
                    row["id"], row["version"],
                    json.dumps(
                        _revision_snapshot(migrated),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    json.dumps(source_ids),
                ),
            )


def _revision_snapshot(row: sqlite3.Row | dict) -> dict:
    data = dict(row)
    return {
        "title": data.get("title"),
        "abstract": data.get("abstract") or "",
        "content": data.get("content") or "",
        "detail": _json_load(data.get("detail_json"), {}),
        "sources": _json_load(data.get("sources_json"), []),
        "keywords": _json_load(data.get("keywords"), []),
        "applicability": context_module.normalize_applicability(
            data.get("applicability_json")
        ),
        "embedding_model": data.get("embedding_model"),
        "embedding_version": data.get("embedding_version"),
        "embedding_content_hash": data.get("embedding_content_hash"),
        "embedding_status": data.get("embedding_status"),
        "parent_id": data.get("parent_id"),
    }


def _record_revision(
    db: sqlite3.Connection,
    story_id: int,
    event_type: str,
) -> None:
    row = db.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
    if row is None:
        return
    db.execute(
        """INSERT OR REPLACE INTO story_revisions (
               story_id, story_version, event_type, snapshot_json,
               source_session_ids
           ) VALUES (?, ?, ?, ?, ?)""",
        (
            story_id,
            row["version"],
            event_type,
            json.dumps(_revision_snapshot(row), ensure_ascii=False, sort_keys=True),
            row["source_session_ids"] or "[]",
        ),
    )


def _upsert_context_dimensions(db: sqlite3.Connection, envelope: dict) -> None:
    device = envelope["device"]
    if device.get("id"):
        db.execute(
            """INSERT INTO devices (
                   id, profile_id, os_family, os_version, arch, display_name
               ) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   os_family = COALESCE(excluded.os_family, devices.os_family),
                   os_version = COALESCE(excluded.os_version, devices.os_version),
                   arch = COALESCE(excluded.arch, devices.arch),
                   display_name = COALESCE(excluded.display_name, devices.display_name)""",
            (
                device["id"], envelope["profile_id"], device.get("os_family"),
                device.get("os_version"), device.get("arch"), device.get("display_name"),
            ),
        )

    tool = envelope["tool"]
    if tool.get("installation_id") and device.get("id"):
        db.execute(
            """INSERT INTO agent_installations (
                   id, profile_id, device_id, tool_type, tool_version, integration_mode
               ) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   tool_version = COALESCE(excluded.tool_version, agent_installations.tool_version)""",
            (
                tool["installation_id"], envelope["profile_id"], device["id"],
                tool.get("type") or "other", tool.get("version"),
                tool.get("integration_mode") or "manual",
            ),
        )

    workspace = envelope["workspace"]
    if workspace.get("id"):
        db.execute(
            """INSERT INTO workspaces (
                   id, profile_id, repo_fingerprint, label, local_path_alias
               ) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   repo_fingerprint = COALESCE(excluded.repo_fingerprint, workspaces.repo_fingerprint),
                   label = COALESCE(excluded.label, workspaces.label),
                   local_path_alias = COALESCE(excluded.local_path_alias, workspaces.local_path_alias)""",
            (
                workspace["id"], envelope["profile_id"],
                workspace.get("repo_fingerprint"), workspace.get("project_label"),
                workspace.get("cwd_alias"),
            ),
        )


def _environments_for_sessions(db: sqlite3.Connection, session_ids: list[int]) -> list[dict]:
    ids = [int(value) for value in session_ids if isinstance(value, int) or str(value).isdigit()]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = db.execute(
        f"SELECT context_json FROM sessions WHERE id IN ({placeholders}) ORDER BY id",
        ids,
    ).fetchall()
    environments = []
    for row in rows:
        try:
            environments.append(json.loads(row["context_json"]))
        except (json.JSONDecodeError, TypeError):
            continue
    return environments


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
        if rebuilt or cleared:
            _bump_index_version(db)
        db.commit()
    finally:
        db.close()
    return {"rebuilt": rebuilt, "cleared": cleared, "failed": failed}


# ═══════════════════════════════════════════════
#  Embedding model/version backfill + atomic index switch
# ═══════════════════════════════════════════════

def get_embedding_index_state() -> dict:
    db = get_db(load_vector_extension=False)
    try:
        row = db.execute(
            "SELECT * FROM embedding_index_state WHERE id = 1"
        ).fetchone()
        return dict(row) if row else {}
    finally:
        db.close()


def _active_embedding_spec(db: sqlite3.Connection) -> dict:
    row = db.execute(
        "SELECT * FROM embedding_index_state WHERE id = 1"
    ).fetchone()
    return dict(row) if row else {
        "active_model": config.EMBED_MODEL,
        "active_version": config.EMBED_VERSION,
        "active_representation": config.EMBED_REPRESENTATION,
    }


def begin_embedding_backfill(
    model: str,
    version: str,
    representation: str,
) -> None:
    """Declare a shadow target without changing the serving index."""

    db = get_db(load_vector_extension=False)
    try:
        existing = db.execute(
            """SELECT DISTINCT embedding_model, representation
               FROM story_embedding_backfill
               WHERE embedding_version = ?""",
            (version,),
        ).fetchall()
        if any(
            row["embedding_model"] != model
            or row["representation"] != representation
            for row in existing
        ):
            raise ValueError(
                f"embedding version {version!r} is immutable and already bound"
            )
        active = db.execute(
            "SELECT * FROM embedding_index_state WHERE id = 1"
        ).fetchone()
        if (
            active and active["active_version"] == version
            and (
                active["active_model"] != model
                or active["active_representation"] != representation
            )
        ):
            raise ValueError(
                f"active embedding version {version!r} is bound to another spec"
            )
        db.execute(
            """UPDATE embedding_index_state
               SET target_model = ?, target_version = ?,
                   target_representation = ?, backfill_status = 'running',
                   updated_at = datetime('now')
               WHERE id = 1""",
            (model, version, representation),
        )
        db.commit()
    finally:
        db.close()


def stories_pending_embedding_backfill(
    version: str,
    representation: str,
    *,
    limit: int = 100,
) -> list[dict]:
    """Return resumable work whose ready shadow vector is missing or stale."""

    db = get_db()
    try:
        rows = db.execute(
            """SELECT s.* FROM stories s
               WHERE s.embedding_status != 'archived'
               ORDER BY s.id"""
        ).fetchall()
        pending = []
        for row in rows:
            story = _row_to_story(row)
            expected_hash = story_v2.content_hash(story, representation)
            shadow = db.execute(
                """SELECT status, content_hash
                   FROM story_embedding_backfill
                   WHERE story_id = ? AND embedding_version = ?""",
                (story["id"], version),
            ).fetchone()
            if (
                shadow is not None
                and shadow["status"] == "ready"
                and shadow["content_hash"] == expected_hash
            ):
                continue
            story["target_content_hash"] = expected_hash
            pending.append(story)
            if limit > 0 and len(pending) >= limit:
                break
        return pending
    finally:
        db.close()


def stage_embedding_backfill(
    story_id: int,
    *,
    model: str,
    version: str,
    representation: str,
    content_hash: str,
    embedding: list[float] | None,
    error: str | None = None,
) -> None:
    """Persist one shadow result; failed rows are retryable on the next run."""

    blob = (
        np.asarray(embedding, dtype=np.float32).tobytes()
        if embedding is not None else None
    )
    status = "ready" if embedding is not None else "failed"
    db = get_db(load_vector_extension=False)
    try:
        db.execute(
            """INSERT INTO story_embedding_backfill (
                   story_id, embedding_model, embedding_version,
                   representation, content_hash, embedding, status, error
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(story_id, embedding_version) DO UPDATE SET
                   embedding_model = excluded.embedding_model,
                   representation = excluded.representation,
                   content_hash = excluded.content_hash,
                   embedding = excluded.embedding,
                   status = excluded.status,
                   error = excluded.error,
                   attempts = story_embedding_backfill.attempts + 1,
                   updated_at = datetime('now')""",
            (
                story_id, model, version, representation, content_hash,
                blob, status, (error or "")[:500] or None,
            ),
        )
        if status == "failed":
            db.execute(
                """UPDATE embedding_index_state
                   SET backfill_status = 'failed', updated_at = datetime('now')
                   WHERE id = 1"""
            )
        db.commit()
    finally:
        db.close()


def embedding_backfill_progress(version: str, representation: str) -> dict:
    db = get_db()
    try:
        stories = db.execute(
            """SELECT * FROM stories
               WHERE embedding_status != 'archived' ORDER BY id"""
        ).fetchall()
        ready = failed = 0
        stale_ids = []
        for row in stories:
            story = _row_to_story(row)
            expected_hash = story_v2.content_hash(story, representation)
            shadow = db.execute(
                """SELECT status, content_hash
                   FROM story_embedding_backfill
                   WHERE story_id = ? AND embedding_version = ?""",
                (story["id"], version),
            ).fetchone()
            if shadow and shadow["status"] == "ready" and shadow["content_hash"] == expected_hash:
                ready += 1
            elif shadow and shadow["status"] == "failed":
                failed += 1
                stale_ids.append(story["id"])
            else:
                stale_ids.append(story["id"])
        return {
            "total": len(stories),
            "ready": ready,
            "failed": failed,
            "pending": len(stories) - ready,
            "pending_story_ids": stale_ids,
        }
    finally:
        db.close()


def activate_embedding_backfill(
    *,
    model: str,
    version: str,
    representation: str,
) -> dict:
    """Atomically replace the active vec0 rows after a complete shadow build."""

    progress = embedding_backfill_progress(version, representation)
    if progress["pending"]:
        raise ValueError(
            f"embedding backfill incomplete: {progress['pending']} pending"
        )
    db = get_db()
    try:
        db.execute("BEGIN IMMEDIATE")
        stories = db.execute(
            """SELECT * FROM stories
               WHERE embedding_status != 'archived' ORDER BY id"""
        ).fetchall()
        rows = []
        for story_row in stories:
            story = _row_to_story(story_row)
            expected_hash = story_v2.content_hash(story, representation)
            shadow = db.execute(
                """SELECT embedding, content_hash, embedding_model, representation,
                          status
                   FROM story_embedding_backfill
                   WHERE story_id = ? AND embedding_version = ?""",
                (story["id"], version),
            ).fetchone()
            if (
                shadow is None or shadow["status"] != "ready"
                or shadow["content_hash"] != expected_hash
                or shadow["embedding_model"] != model
                or shadow["representation"] != representation
            ):
                raise ValueError(
                    f"embedding backfill changed before activation: story {story['id']}"
                )
            rows.append({
                "id": story["id"], "embedding": shadow["embedding"],
                "content_hash": shadow["content_hash"],
            })
        db.execute("DELETE FROM story_vectors")
        for row in rows:
            db.execute(
                "INSERT INTO story_vectors (story_id, embedding) VALUES (?, ?)",
                (row["id"], row["embedding"]),
            )
            db.execute(
                """UPDATE stories SET embedding = ?, embedding_model = ?,
                          embedding_version = ?, embedding_content_hash = ?,
                          embedding_status = 'active', version = version + 1,
                          updated_at = datetime('now')
                   WHERE id = ?""",
                (
                    row["embedding"], model, version, row["content_hash"],
                    row["id"],
                ),
            )
            _record_revision(db, row["id"], "embedding_switch")
        db.execute(
            """UPDATE embedding_index_state
               SET active_model = ?, active_version = ?,
                   active_representation = ?, target_model = NULL,
                   target_version = NULL, target_representation = NULL,
                   backfill_status = 'idle', updated_at = datetime('now')
               WHERE id = 1""",
            (model, version, representation),
        )
        _bump_index_version(db)
        db.commit()
        return {"activated": len(rows), "active_version": version}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def mark_embedding_backfill_ready() -> None:
    db = get_db(load_vector_extension=False)
    try:
        db.execute(
            """UPDATE embedding_index_state
               SET backfill_status = 'ready', updated_at = datetime('now')
               WHERE id = 1"""
        )
        db.commit()
    finally:
        db.close()


# ═══════════════════════════════════════════════
#  Session CRUD
# ═══════════════════════════════════════════════

def add_session(source: str, raw_content: str, problem_desc: str = "",
                code_snippets: str = "[]", conclusion: str = "",
                context: dict | None = None) -> int:
    """插入一条会话日志；每条新 Session 都持久化完整 ContextEnvelope。"""
    db = get_db()
    try:
        global_id = _new_global_id()
        envelope = context_module.normalize_envelope(
            context,
            profile_id=config.PROFILE_ID,
            session_id=global_id,
            source=source,
        )
        _upsert_context_dimensions(db, envelope)
        cur = db.execute(
            """INSERT INTO sessions (
                   global_id, profile_id, sync_state,
                   source, raw_content, problem_desc, code_snippets, conclusion,
                   device_id, agent_installation_id, workspace_id, runtime_json,
                   external_session_hash, context_json
               ) VALUES (?, ?, 'local_only', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                global_id, config.PROFILE_ID,
                source, raw_content, problem_desc, code_snippets, conclusion,
                envelope["device"]["id"],
                envelope["tool"]["installation_id"],
                envelope["workspace"]["id"],
                json.dumps(envelope["runtime"], ensure_ascii=False, sort_keys=True),
                envelope["session"]["external_session_hash"],
                json.dumps(envelope, ensure_ascii=False, sort_keys=True),
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


def get_session_context(session_id: int) -> Optional[dict]:
    """Return a parsed canonical ContextEnvelope for one Session."""

    row = get_session(session_id)
    if row is None:
        return None
    return context_module.normalize_envelope(
        row["context_json"],
        profile_id=row["profile_id"],
        session_id=row["global_id"],
        source=row["source"],
        captured_at=row["created_at"],
    )


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
              source_session_ids: list[int] = None,
              applicability: dict | None = None, *,
              abstract: str | None = None,
              detail: dict | None = None,
              sources: list[dict] | None = None,
              embedding_model: str | None = None,
              embedding_version: str | None = None,
              embedding_content_hash: str | None = None,
              event_type: str = "create") -> int:
    """Create a lossless Story v2 and atomically publish its active vector."""
    db = get_db()
    try:
        emb_blob = np.array(embedding, dtype=np.float32).tobytes()
        source_session_ids = list(dict.fromkeys(source_session_ids or []))
        environments = _environments_for_sessions(db, source_session_ids)
        payload = story_v2.normalize_story_payload(
            {
                "title": title,
                "abstract": abstract,
                "content": content,
                "detail": detail,
                "sources": sources,
                "applicability": applicability,
                "keywords": keywords,
            },
            fallback_content=content,
            source_session_ids=source_session_ids,
        )
        applicability = payload["applicability"]
        active_spec = _active_embedding_spec(db)
        embedding_model = embedding_model or active_spec["active_model"]
        embedding_version = embedding_version or active_spec["active_version"]
        expected_hash = story_v2.content_hash(
            payload, active_spec["active_representation"]
        )
        if (
            embedding_content_hash is not None
            and embedding_content_hash != expected_hash
        ):
            raise ValueError("embedding_content_hash does not match persisted Story")
        embedding_content_hash = expected_hash
        cur = db.execute(
            """INSERT INTO stories (
                   global_id, profile_id, sync_state,
                   title, abstract, content, detail_json, sources_json,
                   keywords, embedding, embedding_model, embedding_version,
                   embedding_content_hash, embedding_status,
                   parent_id, source_session_ids,
                   applicability_json, environment_summary_json
               ) VALUES (?, ?, 'local_only', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active',
                         ?, ?, ?, ?)""",
            (_new_global_id(), config.PROFILE_ID,
             payload["title"], payload["abstract"], payload["content"],
             json.dumps(payload["detail"], ensure_ascii=False, sort_keys=True),
             json.dumps(payload["sources"], ensure_ascii=False, sort_keys=True),
             json.dumps(keywords, ensure_ascii=False), emb_blob,
             embedding_model, embedding_version, embedding_content_hash,
             parent_id, json.dumps(source_session_ids),
             json.dumps(applicability, ensure_ascii=False, sort_keys=True),
             json.dumps(environments, ensure_ascii=False, sort_keys=True))
        )
        story_id = cur.lastrowid
        # 写入向量虚拟表
        db.execute(
            "INSERT INTO story_vectors (story_id, embedding) VALUES (?, ?)",
            (story_id, emb_blob)
        )
        _record_revision(db, story_id, event_type)
        _bump_index_version(db)
        db.commit()
        logger.info("新建 story #%d: %s", story_id, title)
        return story_id
    finally:
        db.close()


def update_story(story_id: int, title: str = None, content: str = None,
                 keywords: list[str] = None, embedding: list[float] = None,
                 applicability: dict | None = None, *,
                 abstract: str | None = None,
                 detail: dict | None = None,
                 sources: list[dict] | None = None,
                 source_session_ids: list[int] | None = None,
                 embedding_model: str | None = None,
                 embedding_version: str | None = None,
                 embedding_content_hash: str | None = None,
                 event_type: str = "update"):
    """Update a Story and append an immutable version event."""
    db = get_db()
    try:
        current = db.execute(
            "SELECT * FROM stories WHERE id = ?", (story_id,)
        ).fetchone()
        if current is None:
            return
        sets = []
        params = []

        if title is not None:
            sets.append("title = ?")
            params.append(title)
        if abstract is not None:
            bounded, _ = story_v2.bound_abstract(abstract)
            sets.append("abstract = ?")
            params.append(bounded)

        current_applicability = context_module.normalize_applicability(
            current["applicability_json"]
        )
        effective_applicability = (
            context_module.normalize_applicability(applicability)
            if applicability is not None else current_applicability
        )
        if detail is not None or applicability is not None:
            raw_detail = dict(
                detail
                if isinstance(detail, dict)
                else _json_load(current["detail_json"], {})
            )
            if content is not None and detail is None:
                raw_detail["problem"] = content
            if applicability is not None:
                raw_detail["applicability"] = effective_applicability
            normalized_detail = story_v2.normalize_detail(
                raw_detail,
                legacy_content=content if content is not None else current["content"],
                applicability=effective_applicability,
            )
            effective_applicability = normalized_detail["applicability"]
            sets.append("detail_json = ?")
            params.append(json.dumps(
                normalized_detail, ensure_ascii=False, sort_keys=True
            ))
            sets.extend(("content = ?", "applicability_json = ?"))
            params.extend((
                (
                    content
                    if content is not None
                    else story_v2.render_detail(normalized_detail)
                ),
                json.dumps(
                    effective_applicability, ensure_ascii=False, sort_keys=True
                ),
            ))
        elif content is not None:
            sets.append("content = ?")
            params.append(content)

        if sources is not None or source_session_ids is not None:
            effective_ids = list(dict.fromkeys(
                source_session_ids
                if source_session_ids is not None
                else _json_load(current["source_session_ids"], [])
            ))
            normalized_sources = story_v2.normalize_sources(
                sources if sources is not None else _json_load(current["sources_json"], []),
                effective_ids,
            )
            sets.extend(("sources_json = ?", "source_session_ids = ?"))
            params.extend((
                json.dumps(normalized_sources, ensure_ascii=False, sort_keys=True),
                json.dumps(effective_ids),
            ))
            environments = context_module.merge_environments(
                current["environment_summary_json"],
                _environments_for_sessions(db, effective_ids),
            )
            sets.append("environment_summary_json = ?")
            params.append(json.dumps(
                environments, ensure_ascii=False, sort_keys=True
            ))
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

        persisted = db.execute(
            "SELECT * FROM stories WHERE id = ?", (story_id,)
        ).fetchone()
        active_spec = _active_embedding_spec(db)
        expected_hash = story_v2.content_hash(
            _row_to_story(persisted), active_spec["active_representation"]
        )
        if embedding is not None:
            if (
                embedding_content_hash is not None
                and embedding_content_hash != expected_hash
            ):
                raise ValueError(
                    "embedding_content_hash does not match persisted Story"
                )
            db.execute(
                """UPDATE stories SET embedding_model = ?, embedding_version = ?,
                          embedding_content_hash = ?, embedding_status = 'active'
                   WHERE id = ?""",
                (
                    embedding_model or active_spec["active_model"],
                    embedding_version or active_spec["active_version"],
                    expected_hash,
                    story_id,
                ),
            )
        elif (
            persisted["embedding"] is not None
            and persisted["embedding_status"] != "archived"
            and persisted["embedding_content_hash"] != expected_hash
        ):
            db.execute(
                "UPDATE stories SET embedding_status = 'stale' WHERE id = ?",
                (story_id,),
            )

        # 如果向量更新了，同步到虚拟表
        if embedding is not None:
            db.execute(
                "DELETE FROM story_vectors WHERE story_id = ?", (story_id,)
            )
            db.execute(
                "INSERT INTO story_vectors (story_id, embedding) VALUES (?, ?)",
                (story_id, emb_blob)
            )
        _record_revision(db, story_id, event_type)
        if any(value is not None for value in (
            title, content, keywords, embedding, applicability, abstract,
            detail, sources, source_session_ids,
        )):
            _bump_index_version(db)
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
        cur = db.execute("DELETE FROM story_vectors WHERE story_id = ?", (story_id,))
        updated = db.execute(
            """UPDATE stories
               SET embedding = NULL, embedding_status = 'archived',
                   version = version + 1, updated_at = datetime('now')
               WHERE id = ? AND embedding IS NOT NULL""",
            (story_id,),
        )
        if cur.rowcount or updated.rowcount:
            _record_revision(db, story_id, "split_parent")
            _bump_index_version(db)
        db.commit()
        logger.info("已从检索索引移除 story #%d 的向量", story_id)
    finally:
        db.close()


def get_story(story_id: int) -> Optional[dict]:
    db = get_db()
    try:
        row = db.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        if row:
            story = _row_to_story(row)
            hydrated_sources = []
            for source in story.get("sources", []):
                item = dict(source)
                session_id = item.get("session_id")
                if session_id is not None:
                    session = db.execute(
                        """SELECT global_id, source, created_at
                           FROM sessions WHERE id = ?""",
                        (session_id,),
                    ).fetchone()
                    if session is not None:
                        item.update({
                            "session_global_id": session["global_id"],
                            "source": session["source"],
                            "captured_at": session["created_at"],
                        })
                hydrated_sources.append(item)
            story["sources"] = hydrated_sources
            return story
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


def get_story_revisions(story_id: int) -> list[dict]:
    """Return the immutable create/update/merge/split chain for one Story."""

    db = get_db(load_vector_extension=False)
    try:
        rows = db.execute(
            """SELECT story_version, event_type, snapshot_json,
                      source_session_ids, created_at
               FROM story_revisions
               WHERE story_id = ? ORDER BY story_version""",
            (story_id,),
        ).fetchall()
        return [
            {
                "version": row["story_version"],
                "event_type": row["event_type"],
                "snapshot": _json_load(row["snapshot_json"], {}),
                "source_session_ids": _json_load(row["source_session_ids"], []),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        db.close()


def update_story_raw_sessions(story_id: int, session_ids: list[int]):
    """更新来源 Session，并合并环境集合而非覆盖最后一次来源。"""
    db = get_db()
    try:
        session_ids = list(dict.fromkeys(session_ids))
        row = db.execute(
            """SELECT environment_summary_json, sources_json
               FROM stories WHERE id = ?""", (story_id,)
        ).fetchone()
        if row is None:
            return
        environments = context_module.merge_environments(
            row["environment_summary_json"],
            _environments_for_sessions(db, session_ids),
        )
        db.execute(
            """UPDATE stories
               SET source_session_ids = ?, sources_json = ?,
                   environment_summary_json = ?, version = version + 1,
                   updated_at = datetime('now')
               WHERE id = ?""",
            (
                json.dumps(session_ids),
                json.dumps(
                    story_v2.normalize_sources(
                        _json_load(row["sources_json"], []), session_ids
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(environments, ensure_ascii=False, sort_keys=True),
                story_id,
            )
        )
        _record_revision(db, story_id, "source_update")
        _bump_index_version(db)
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


def apply_recall_feedback(
    story_ids: list[int], *, db_path: str | Path | None = None
) -> None:
    """单事务写入一次召回的访问计数与共同召回边权反馈。

    查询线程通过后台队列调用本函数；这里刻意不推进 ``index_version``，避免
    纯反馈写使查询结果缓存每次都失效。

    新 schema 会为每对 Story 维护独立 ``co_recall`` 边，并继续强化已有
    语义/层级边以保持 v0.1 的学习语义。写失败由上层队列吞掉，不会
    变成查询失败。
    """

    unique_ids = list(dict.fromkeys(int(story_id) for story_id in story_ids))
    if not unique_ids:
        return
    db = get_db(db_path, load_vector_extension=False)
    try:
        db.executemany(
            "UPDATE stories SET access_count = access_count + 1 WHERE id = ?",
            [(story_id,) for story_id in unique_ids],
        )
        for source_id, target_id in combinations(sorted(unique_ids), 2):
            now = datetime.now(UTC).isoformat()
            db.execute(
                """UPDATE edges
                   SET weight = MIN(weight + ?, ?),
                       observations = observations + 1,
                       version = version + 1,
                       last_reinforced_at = ?, updated_at = ?
                   WHERE source_id = ? AND target_id = ?
                     AND edge_type != 'co_recall' AND deleted_at IS NULL""",
                (
                    config.WEIGHT_INCREMENT,
                    config.WEIGHT_MAX,
                    now,
                    now,
                    source_id,
                    target_id,
                ),
            )
            db.execute(
                """INSERT INTO edges (
                       global_id, profile_id, sync_state, source_id, target_id,
                       weight, edge_type, directed, provenance_json, version,
                       observations, last_reinforced_at, updated_at
                   ) VALUES (?, ?, 'local_only', ?, ?, ?, 'co_recall', 0, ?, 1, 1, ?, ?)
                   ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                       weight = MIN(edges.weight + excluded.weight, ?),
                       observations = edges.observations + 1,
                       version = edges.version + 1,
                       last_reinforced_at = excluded.last_reinforced_at,
                       updated_at = excluded.updated_at,
                       deleted_at = NULL""",
                (
                    _new_global_id(), config.PROFILE_ID, source_id, target_id,
                    min(config.WEIGHT_INCREMENT, config.WEIGHT_MAX),
                    json.dumps(
                        {"source": "recall_feedback", "method": "co_occurrence"},
                        sort_keys=True,
                    ),
                    now,
                    now,
                    config.WEIGHT_MAX,
                ),
            )
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


def _normalize_edge_type(edge_type: str | None, *, strict: bool = False) -> str:
    normalized = (edge_type or "semantic").strip().lower()
    allowed = set(config.MEMORY_EDGE_TYPES) | {"sibling"}
    if normalized in allowed:
        return normalized
    if strict:
        raise ValueError(
            f"不支持的 edge_type: {edge_type}; "
            f"允许值为 {', '.join(config.MEMORY_EDGE_TYPES)}"
        )
    return "semantic"


def _edge_is_directed(edge_type: str) -> bool:
    return edge_type in config.DIRECTED_EDGE_TYPES


def _canonical_edge_endpoints(
    source_id: int, target_id: int, *, directed: bool
) -> tuple[int, int]:
    source_id, target_id = int(source_id), int(target_id)
    return (source_id, target_id) if directed else _edge_pair(source_id, target_id)


def _edge_provenance(value: dict | None, *, edge_type: str) -> dict:
    if value is not None and not isinstance(value, dict):
        raise ValueError("provenance 必须是 JSON object")
    provenance = dict(value or {})
    provenance.setdefault("source", "storybook")
    provenance.setdefault("relation", edge_type)
    return provenance


def _edge_row_to_dict(row: sqlite3.Row | dict) -> dict:
    result = dict(row)
    result["directed"] = bool(result.get("directed"))
    result["provenance"] = _json_load(result.pop("provenance_json", None), {})
    return result


def add_or_update_edge(
    source_id: int,
    target_id: int,
    weight: float,
    edge_type: str = "semantic",
    *,
    provenance: dict | None = None,
    directed: bool | None = None,
) -> int:
    """添加或更新一条有类型、可解释的 Memory Graph 边。

    标准方向语义：``temporal`` 旧→新，``causal`` 因→果，
    ``parent_child`` 父→子，``supersedes`` 新→旧。其余类型无向并对
    端点排序。同一 Story 对可同时拥有多种边；同类型重复写取较大
    权重，且保留 provenance/version 审计信息。
    """

    edge_type = _normalize_edge_type(edge_type, strict=True)
    expected_direction = _edge_is_directed(edge_type)
    if directed is not None and bool(directed) != expected_direction:
        raise ValueError(f"{edge_type} 的 directed 必须为 {expected_direction}")
    directed = expected_direction
    source_id, target_id = _canonical_edge_endpoints(
        source_id, target_id, directed=directed
    )
    weight = max(0.0, min(float(weight), config.WEIGHT_MAX))
    new_provenance = _edge_provenance(provenance, edge_type=edge_type)
    db = get_db()
    try:
        existing = db.execute(
            """SELECT * FROM edges
               WHERE source_id = ? AND target_id = ? AND edge_type = ?""",
            (source_id, target_id, edge_type),
        ).fetchone()
        changed = False
        if existing:
            new_weight = min(max(float(existing["weight"]), weight), config.WEIGHT_MAX)
            old_provenance = _json_load(existing["provenance_json"], {})
            merged_provenance = dict(old_provenance)
            if provenance is not None:
                merged_provenance.update(new_provenance)
            revived = existing["deleted_at"] is not None
            changed = (
                new_weight != existing["weight"]
                or merged_provenance != old_provenance
                or revived
            )
            if changed:
                db.execute(
                    """UPDATE edges
                       SET weight = ?, provenance_json = ?, directed = ?,
                           version = version + 1, updated_at = datetime('now'),
                           deleted_at = NULL
                       WHERE id = ?""",
                    (
                        new_weight,
                        json.dumps(
                            merged_provenance, ensure_ascii=False, sort_keys=True
                        ),
                        int(directed),
                        existing["id"],
                    ),
                )
            edge_id = int(existing["id"])
        else:
            cursor = db.execute(
                """INSERT INTO edges (
                       global_id, profile_id, sync_state,
                       source_id, target_id, weight, edge_type, directed,
                       provenance_json
                   ) VALUES (?, ?, 'local_only', ?, ?, ?, ?, ?, ?)""",
                (
                    _new_global_id(), config.PROFILE_ID,
                    source_id, target_id, weight, edge_type, int(directed),
                    json.dumps(new_provenance, ensure_ascii=False, sort_keys=True),
                ),
            )
            edge_id = int(cursor.lastrowid)
            changed = True
        if changed:
            _bump_index_version(db)
        db.commit()
        return edge_id
    finally:
        db.close()


def increment_edge_weight(
    source_id: int,
    target_id: int,
    delta: float = None,
    *,
    edge_type: str | None = None,
) -> None:
    """提升一条已有边权重，不凭空创建关系。

    未指定类型时优先兼容旧的 ``semantic`` 边，否则选取权重最高
    的现有关系。共现反馈的专用语义由 ``apply_recall_feedback``
    维护。
    """
    if delta is None:
        delta = config.WEIGHT_INCREMENT
    pair = _edge_pair(int(source_id), int(target_id))
    normalized_type = (
        _normalize_edge_type(edge_type, strict=True) if edge_type else None
    )
    db = get_db()
    try:
        row = db.execute(
            """SELECT * FROM edges
               WHERE deleted_at IS NULL
                 AND ((source_id = ? AND target_id = ?)
                   OR (source_id = ? AND target_id = ?))
                 AND (? IS NULL OR edge_type = ?)
               ORDER BY CASE WHEN edge_type = 'semantic' THEN 0 ELSE 1 END,
                        weight DESC, id
               LIMIT 1""",
            (
                pair[0], pair[1], pair[1], pair[0],
                normalized_type, normalized_type,
            ),
        ).fetchone()
        if row:
            new_weight = max(
                0.0, min(float(row["weight"]) + float(delta), config.WEIGHT_MAX)
            )
            db.execute(
                """UPDATE edges SET weight = ?, version = version + 1,
                                     updated_at = datetime('now')
                   WHERE id = ?""",
                (new_weight, row["id"]),
            )
            db.commit()
    finally:
        db.close()


def delete_edge(
    source_id: int, target_id: int, edge_type: str | None = None
) -> int:
    """软删除匹配边并返回受影响行数；历史仍可审计。"""

    pair = _edge_pair(int(source_id), int(target_id))
    normalized_type = (
        _normalize_edge_type(edge_type, strict=True) if edge_type else None
    )
    db = get_db()
    try:
        cursor = db.execute(
            """UPDATE edges
               SET deleted_at = datetime('now'), updated_at = datetime('now'),
                   version = version + 1
               WHERE deleted_at IS NULL
                 AND ((source_id = ? AND target_id = ?)
                   OR (source_id = ? AND target_id = ?))
                 AND (? IS NULL OR edge_type = ?)""",
            (
                pair[0], pair[1], pair[1], pair[0],
                normalized_type, normalized_type,
            ),
        )
        if cursor.rowcount:
            _bump_index_version(db)
        db.commit()
        return max(0, cursor.rowcount)
    finally:
        db.close()


def get_edges(story_id: int, *, include_deleted: bool = False) -> list[dict]:
    """获取 Story 的入/出/无向边，按权重降序。"""

    db = get_db()
    try:
        rows = db.execute(
            """SELECT e.*,
                      CASE WHEN e.source_id = ? THEN e.target_id
                           ELSE e.source_id END AS related_id,
                      CASE WHEN e.directed = 0 THEN 'undirected'
                           WHEN e.source_id = ? THEN 'outbound'
                           ELSE 'inbound' END AS direction
               FROM edges e
               WHERE (e.source_id = ? OR e.target_id = ?)
                 AND (? OR e.deleted_at IS NULL)
               ORDER BY e.weight DESC, e.id""",
            (story_id, story_id, story_id, story_id, int(include_deleted)),
        ).fetchall()
        return [_edge_row_to_dict(row) for row in rows]
    finally:
        db.close()


def get_related_stories(story_id: int, limit: int = 5) -> list[dict]:
    """获取去重后的相关 Story，保留最强边的解释。"""

    return get_related_stories_batch([story_id], limit=limit).get(story_id, [])


def get_related_stories_batch(
    story_ids: list[int], limit: int = 5
) -> dict[int, list[dict]]:
    """在一个只读连接中批量读取多条 Story 的关联项。

    同一 Story 对可同时有多种边；兼容的 ``related`` 视图按目标
    Story 去重，仅展示最强的一条。完整多边路径由 Graph RAG 解释
    字段返回。
    """

    unique_ids = list(dict.fromkeys(int(story_id) for story_id in story_ids))
    related_by_story = {story_id: [] for story_id in unique_ids}
    if not unique_ids:
        return related_by_story
    db = get_db(load_vector_extension=False)
    try:
        for story_id in unique_ids:
            rows = db.execute(
                """SELECT s.*, e.weight, e.edge_type, e.directed,
                          e.provenance_json, e.version AS edge_version,
                          CASE WHEN e.directed = 0 THEN 'undirected'
                               WHEN e.source_id = ? THEN 'outbound'
                               ELSE 'inbound' END AS edge_direction
                   FROM edges e
                   JOIN stories s ON s.id = CASE
                       WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
                   WHERE (e.source_id = ? OR e.target_id = ?)
                     AND e.deleted_at IS NULL
                     AND s.embedding_status != 'archived'
                   ORDER BY e.weight DESC, e.id""",
                (story_id, story_id, story_id, story_id),
            ).fetchall()
            seen: set[int] = set()
            related = []
            for row in rows:
                item = _row_to_story(row)
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                item["directed"] = bool(item.get("directed"))
                item["edge_provenance"] = _json_load(
                    item.pop("provenance_json", None), {}
                )
                related.append(item)
                if len(related) >= max(0, int(limit)):
                    break
            related_by_story[story_id] = related
        return related_by_story
    finally:
        db.close()


def get_graph_neighbors_batch(
    story_ids: list[int],
    *,
    fan_out: int,
    allowed_edge_types: set[str] | frozenset[str] | None = None,
) -> dict[int, dict]:
    """为一层图扩散读取已通过方向/路径策略的有界邻接快照。

    fan-out 必须在方向和路径类型过滤之后执行：否则不可遍历的
    高权入边会占满 LIMIT，使合法出边饥饿。返回值为
    ``{story_id: {items, policy_suppressed, fan_out_truncated}}``。
    """

    unique_ids = list(dict.fromkeys(int(story_id) for story_id in story_ids))
    output = {
        story_id: {
            "items": [],
            "policy_suppressed": 0,
            "fan_out_truncated": False,
        }
        for story_id in unique_ids
    }
    fan_out = max(0, int(fan_out))
    if not unique_ids or fan_out == 0:
        return output
    allowed = sorted(
        set(config.GRAPH_EDGE_TYPE_FACTORS)
        if allowed_edge_types is None else set(allowed_edge_types)
    )
    db = get_db(load_vector_extension=False)
    try:
        for story_id in unique_ids:
            type_clause = (
                f"e.edge_type IN ({','.join('?' for _ in allowed)})"
                if allowed else "0"
            )
            direction_clause = """(
                e.directed = 0
                OR e.edge_type = 'parent_child'
                OR (e.edge_type = 'supersedes' AND e.target_id = ?)
                OR (
                    e.directed = 1
                    AND e.edge_type NOT IN ('parent_child', 'supersedes')
                    AND e.source_id = ?
                )
            )"""
            counts = db.execute(
                f"""SELECT COUNT(*) AS total,
                           COALESCE(SUM(CASE
                               WHEN {direction_clause} AND {type_clause}
                               THEN 1 ELSE 0 END), 0) AS eligible
                    FROM edges e
                    JOIN stories s ON s.id = CASE
                        WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
                    WHERE (e.source_id = ? OR e.target_id = ?)
                      AND e.deleted_at IS NULL
                      AND s.embedding_status != 'archived'""",
                (
                    story_id, story_id, *allowed,
                    story_id, story_id, story_id,
                ),
            ).fetchone()
            eligible_count = int(counts["eligible"] or 0)
            output[story_id]["policy_suppressed"] = max(
                0, int(counts["total"] or 0) - eligible_count
            )
            output[story_id]["fan_out_truncated"] = eligible_count > fan_out
            if eligible_count == 0:
                continue
            rows = db.execute(
                f"""SELECT e.*, s.id AS story_id, s.title, s.abstract, s.content,
                          s.keywords, s.applicability_json,
                          s.environment_summary_json,
                          (SELECT COUNT(*) FROM edges degree
                           WHERE degree.deleted_at IS NULL
                             AND (degree.source_id = s.id OR degree.target_id = s.id)
                          ) AS degree
                   FROM edges e
                   JOIN stories s ON s.id = CASE
                       WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
                   WHERE (e.source_id = ? OR e.target_id = ?)
                     AND e.deleted_at IS NULL
                     AND s.embedding_status != 'archived'
                     AND {direction_clause}
                     AND {type_clause}
                   ORDER BY e.weight DESC, e.id
                   LIMIT ?""",
                (
                    story_id, story_id, story_id,
                    story_id, story_id,
                    *allowed,
                    fan_out,
                ),
            ).fetchall()
            items = []
            for row in rows:
                direction = (
                    "undirected" if not row["directed"]
                    else "outbound" if row["source_id"] == story_id
                    else "inbound"
                )
                summary, truncated = story_v2.recall_summary(dict(row))
                items.append({
                    "story_id": int(row["story_id"]),
                    "title": row["title"],
                    "abstract": summary,
                    "content": summary,
                    "truncated": truncated,
                    "keywords": _json_load(row["keywords"], []),
                    "applicability": context_module.normalize_applicability(
                        row["applicability_json"]
                    ),
                    "environments": context_module.merge_environments(
                        row["environment_summary_json"], []
                    ),
                    "degree": int(row["degree"] or 0),
                    "edge": {
                        "id": int(row["id"]),
                        "global_id": row["global_id"],
                        "source_id": int(row["source_id"]),
                        "target_id": int(row["target_id"]),
                        "edge_type": row["edge_type"],
                        "weight": float(row["weight"]),
                        "directed": bool(row["directed"]),
                        "traversal": direction,
                        "provenance": _json_load(row["provenance_json"], {}),
                        "version": int(row["version"]),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    },
                })
            output[story_id]["items"] = items
        return output
    finally:
        db.close()


def get_superseded_story_ids(story_ids: list[int]) -> set[int]:
    """返回默认应被新版 Story 抑制的旧 Story ID。"""

    unique_ids = list(dict.fromkeys(int(story_id) for story_id in story_ids))
    if not unique_ids:
        return set()
    placeholders = ",".join("?" for _ in unique_ids)
    db = get_db(load_vector_extension=False)
    try:
        rows = db.execute(
            f"""SELECT e.target_id
                 FROM edges e
                 JOIN stories replacement ON replacement.id = e.source_id
                 WHERE e.edge_type = 'supersedes' AND e.deleted_at IS NULL
                   AND e.target_id IN ({placeholders})
                   AND replacement.embedding_status != 'archived'""",
            unique_ids,
        ).fetchall()
        return {int(row["target_id"]) for row in rows}
    finally:
        db.close()


def decay_co_recall_edges(
    *,
    now: datetime | None = None,
    half_life_days: float | None = None,
    min_weight: float | None = None,
    db_path: str | Path | None = None,
) -> dict:
    """对共现边做半衰期衰减，低于下限时软删除。"""

    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    half_life_days = (
        config.GRAPH_CO_RECALL_HALF_LIFE_DAYS
        if half_life_days is None else float(half_life_days)
    )
    min_weight = (
        config.GRAPH_CO_RECALL_MIN_WEIGHT
        if min_weight is None else float(min_weight)
    )
    if half_life_days <= 0:
        raise ValueError("half_life_days 必须大于 0")
    db = get_db(db_path, load_vector_extension=False)
    decayed = deleted = 0
    try:
        rows = db.execute(
            """SELECT * FROM edges
               WHERE edge_type = 'co_recall' AND deleted_at IS NULL"""
        ).fetchall()
        for row in rows:
            raw_timestamp = (
                row["last_reinforced_at"] or row["updated_at"] or row["created_at"]
            )
            try:
                timestamp = datetime.fromisoformat(
                    str(raw_timestamp).replace("Z", "+00:00")
                )
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
            except (TypeError, ValueError):
                timestamp = now
            age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
            new_weight = float(row["weight"]) * (0.5 ** (age_days / half_life_days))
            if new_weight >= float(row["weight"]) - 1e-12:
                continue
            if new_weight < min_weight:
                db.execute(
                    """UPDATE edges SET deleted_at = ?, updated_at = ?,
                                         version = version + 1
                       WHERE id = ?""",
                    (now.isoformat(), now.isoformat(), row["id"]),
                )
                deleted += 1
            else:
                db.execute(
                    """UPDATE edges SET weight = ?, updated_at = ?,
                                         version = version + 1
                       WHERE id = ?""",
                    (new_weight, now.isoformat(), row["id"]),
                )
                decayed += 1
        db.commit()
        return {"decayed": decayed, "deleted": deleted, "examined": len(rows)}
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
            """SELECT v.story_id, v.distance, s.title, s.abstract, s.content, s.keywords,
                      s.applicability_json, s.environment_summary_json
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
            summary, truncated = story_v2.recall_summary(dict(r))
            results.append({
                "story_id": r["story_id"],
                "title": r["title"],
                "abstract": summary,
                "content": summary,
                "truncated": truncated,
                "keywords": json.loads(r["keywords"]),
                "applicability": context_module.normalize_applicability(
                    r["applicability_json"]
                ),
                "environments": context_module.merge_environments(
                    r["environment_summary_json"], []
                ),
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
            """SELECT id, title, abstract, content, keywords, embedding,
                      applicability_json, environment_summary_json
               FROM stories WHERE embedding IS NOT NULL"""
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
            summary, truncated = story_v2.recall_summary(dict(r))
            results.append({
                "story_id": r["id"],
                "title": r["title"],
                "abstract": summary,
                "content": summary,
                "truncated": truncated,
                "keywords": json.loads(r["keywords"]),
                "applicability": context_module.normalize_applicability(
                    r["applicability_json"]
                ),
                "environments": context_module.merge_environments(
                    r["environment_summary_json"], []
                ),
                "similarity": round(sim, 4),
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    finally:
        db.close()


def search_by_lexical(
    query: str, top_k: int = 5, *, timeout_seconds: float = 0.5
) -> list[dict]:
    """FTS5 + 参数化关键词子串的低时延降级检索。

    FTS5 负责常规词法命中；LIKE 分支补齐中文连续文本和关键词 JSON 中 FTS
    tokenizer 难以稳定切分的情况。SQLite progress handler 与 busy_timeout 共同
    保证调用方给定的 deadline。
    """

    terms = _lexical_terms(query)
    if not terms or top_k <= 0:
        return []
    deadline = time.perf_counter() + max(0.001, timeout_seconds)
    db = get_db(load_vector_extension=False)
    busy_timeout_ms = max(1, int(timeout_seconds * 1000))
    db.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    db.set_progress_handler(
        lambda: 1 if time.perf_counter() >= deadline else 0,
        500,
    )
    try:
        candidates: dict[int, sqlite3.Row] = {}
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        try:
            rows = db.execute(
                """SELECT s.id, s.title, s.abstract, s.content, s.keywords,
                          s.applicability_json, s.environment_summary_json
                   FROM story_fts
                   JOIN stories s ON s.id = story_fts.rowid
                   WHERE story_fts MATCH ? AND s.embedding IS NOT NULL
                   ORDER BY bm25(story_fts, 8.0, 5.0, 2.0, 5.0)
                   LIMIT ?""",
                (fts_query, max(top_k * 8, 16)),
            ).fetchall()
            candidates.update({int(row["id"]): row for row in rows})
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise TimeoutError("lexical fallback timed out") from exc
            logger.warning("FTS 查询失败，继续关键词 fallback")

        clauses = []
        params: list[object] = []
        for term in terms:
            clauses.append(
                "(instr(lower(s.title), ?) > 0 OR "
                "instr(lower(s.abstract), ?) > 0 OR "
                "instr(lower(s.content), ?) > 0 OR "
                "instr(lower(s.keywords), ?) > 0)"
            )
            params.extend((term, term, term, term))
        rows = db.execute(
            f"""SELECT s.id, s.title, s.abstract, s.content, s.keywords,
                       s.applicability_json, s.environment_summary_json
                FROM stories s
                WHERE s.embedding IS NOT NULL AND ({' OR '.join(clauses)})
                LIMIT ?""",
            (*params, max(top_k * 32, 256)),
        ).fetchall()
        candidates.update({int(row["id"]): row for row in rows})

        exact = query.strip().casefold()
        ranked = []
        for row in candidates.values():
            title = row["title"].casefold()
            abstract = row["abstract"].casefold()
            content = row["content"].casefold()
            keywords_text = row["keywords"].casefold()
            score = 0.0
            if exact:
                score += 6.0 if exact in title else 0.0
                score += 4.0 if exact in abstract else 0.0
                score += 3.0 if exact in content else 0.0
                score += 4.0 if exact in keywords_text else 0.0
            for term in terms:
                score += 3.0 if term in title else 0.0
                score += 2.0 if term in abstract else 0.0
                score += 1.0 if term in content else 0.0
                score += 2.0 if term in keywords_text else 0.0
            if score <= 0:
                # 仅 FTS tokenizer 命中的结果仍保留一个稳定的最低分。
                score = 0.5
            max_score = 17.0 + 8.0 * len(terms)
            summary, truncated = story_v2.recall_summary(dict(row))
            ranked.append({
                "story_id": row["id"],
                "title": row["title"],
                "abstract": summary,
                "content": summary,
                "truncated": truncated,
                "keywords": json.loads(row["keywords"]),
                "applicability": context_module.normalize_applicability(
                    row["applicability_json"]
                ),
                "environments": context_module.merge_environments(
                    row["environment_summary_json"], []
                ),
                "similarity": round(min(1.0, score / max_score), 4),
                "lexical_score": round(score, 4),
            })
        ranked.sort(
            key=lambda item: (-item["lexical_score"], item["story_id"])
        )
        return ranked[:top_k]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise TimeoutError("lexical fallback timed out") from exc
        raise
    finally:
        db.set_progress_handler(None, 0)
        db.close()


def _lexical_terms(query: str) -> list[str]:
    raw_terms = re.findall(
        r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+",
        query.casefold(),
    )
    terms: list[str] = []
    for token in raw_terms:
        if token not in terms:
            terms.append(token)
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", token) and len(token) > 2:
            for index in range(len(token) - 1):
                bigram = token[index:index + 2]
                if bigram not in terms:
                    terms.append(bigram)
                if len(terms) >= 8:
                    break
        if len(terms) >= 8:
            break
    return terms


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
    else:
        d["source_session_ids"] = []
    d["detail"] = story_v2.normalize_detail(
        _json_load(d.pop("detail_json", None), {}),
        legacy_content=d.get("content", ""),
        applicability=d.get("applicability_json"),
    )
    d["sources"] = story_v2.normalize_sources(
        _json_load(d.pop("sources_json", None), []),
        d["source_session_ids"],
    )
    d["applicability"] = context_module.normalize_applicability(
        d.pop("applicability_json", None)
    )
    d["environments"] = context_module.merge_environments(
        d.pop("environment_summary_json", None), []
    )
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
            "edges": db.execute(
                "SELECT COUNT(*) FROM edges WHERE deleted_at IS NULL"
            ).fetchone()[0],
            "root_stories": db.execute("SELECT COUNT(*) FROM stories WHERE parent_id IS NULL").fetchone()[0],
            "child_stories": db.execute("SELECT COUNT(*) FROM stories WHERE parent_id IS NOT NULL").fetchone()[0],
            "profile": {
                "id": config.PROFILE_ID,
                "display_name": config.ACTIVE_PROFILE.display_name,
                "mode": config.PROFILE_MODE,
            },
            "sync_state": config.SYNC_STATE,
        }
        stats["edge_types"] = {
            row["edge_type"]: row["count"]
            for row in db.execute(
                """SELECT edge_type, COUNT(*) AS count FROM edges
                   WHERE deleted_at IS NULL GROUP BY edge_type ORDER BY edge_type"""
            ).fetchall()
        }
        return stats
    finally:
        db.close()
