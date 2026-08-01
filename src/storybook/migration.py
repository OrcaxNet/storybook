"""Safe v1 project database migration into a Profile-owned v2 generation.

The source database is always opened read-only.  Conversion happens in an
isolated generation and the only cut-over is an atomic Profile registry write.
This avoids replacing a WAL database underneath another process and gives
rollback a single, auditable pointer change.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import sqlite_vec

from . import config, store
from .profiles import Profile, ProfileRegistry

try:  # Unix/macOS
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - Unix/macOS
    msvcrt = None


MIGRATION_NAMESPACE = uuid.UUID("6d7f36c9-cc7a-4d8a-b35b-642dc97c75b7")
BACKUP_RETENTION_DAYS = 30
CORE_TABLES = ("sessions", "stories", "edges")
REQUIRED_COLUMNS = {
    "sessions": {"id", "source", "raw_content"},
    "stories": {
        "id", "title", "content", "keywords", "embedding", "parent_id",
        "source_session_ids",
    },
    "edges": {"id", "source_id", "target_id", "weight", "edge_type"},
}


class MigrationError(RuntimeError):
    """Stable migration failure suitable for CLI and automation."""

    def __init__(self, code: str, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "hint": self.hint}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)


def _private_file(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o600)


def _readonly_backup(path: Path) -> None:
    """Make the retained source snapshot immutable to normal app writes."""

    if os.name != "nt":
        path.chmod(0o400)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    _private_dir(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        _private_file(path)
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - Windows
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)


def _read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    db = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA busy_timeout=5000")
    return db


def _columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [str(row["name"]) for row in db.execute(f"PRAGMA table_info({table})")]


def _validate_v1_schema(db: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for table, required in REQUIRED_COLUMNS.items():
        if table not in tables:
            raise MigrationError(
                "SB_MIGRATION_SCHEMA_INVALID",
                f"旧数据库缺少 {table} 表",
                hint="确认 --source 指向 Storybook v1 memory.db",
            )
        missing = required - set(_columns(db, table))
        if missing:
            raise MigrationError(
                "SB_MIGRATION_SCHEMA_INVALID",
                f"旧数据库 {table} 表缺少字段: {', '.join(sorted(missing))}",
                hint="不要手工修补源库；先从原始备份恢复",
            )
    story_columns = set(_columns(db, "stories"))
    if {"detail_json", "sources_json", "embedding_version"}.issubset(
        story_columns
    ):
        raise MigrationError(
            "SB_MIGRATION_SOURCE_ALREADY_V2",
            "源数据库已经是 Story v2，不需要执行 v1 迁移",
        )
    _validate_source_relations(db)


def _validate_source_relations(db: sqlite3.Connection) -> None:
    session_ids = {
        int(row[0]) for row in db.execute("SELECT id FROM sessions")
    }
    story_ids = {int(row[0]) for row in db.execute("SELECT id FROM stories")}
    for row in db.execute(
        "SELECT id, parent_id, source_session_ids FROM stories ORDER BY id"
    ):
        if row["parent_id"] is not None and int(row["parent_id"]) not in story_ids:
            raise MigrationError(
                "SB_MIGRATION_SOURCE_RELATION_INVALID",
                f"Story #{row['id']} 指向不存在的 parent #{row['parent_id']}",
            )
        try:
            source_ids = json.loads(row["source_session_ids"] or "[]")
        except (json.JSONDecodeError, TypeError) as exc:
            raise MigrationError(
                "SB_MIGRATION_SOURCE_RELATION_INVALID",
                f"Story #{row['id']} 的 source_session_ids 不是有效 JSON",
            ) from exc
        if not isinstance(source_ids, list) or any(
            not isinstance(value, int) or value not in session_ids
            for value in source_ids
        ):
            raise MigrationError(
                "SB_MIGRATION_SOURCE_RELATION_INVALID",
                f"Story #{row['id']} 引用了不存在或无效的 Session",
            )
    for row in db.execute(
        "SELECT id, source_id, target_id FROM edges ORDER BY id"
    ):
        if int(row["source_id"]) not in story_ids or int(row["target_id"]) not in story_ids:
            raise MigrationError(
                "SB_MIGRATION_SOURCE_RELATION_INVALID",
                f"Edge #{row['id']} 引用了不存在的 Story",
            )


def _hash_item(digest: "hashlib._Hash", value: object) -> None:
    if value is None:
        payload = b"null"
    elif isinstance(value, bytes):
        payload = b"bytes:" + value
    elif isinstance(value, float):
        payload = b"float:" + value.hex().encode("ascii")
    elif isinstance(value, int):
        payload = b"int:" + str(value).encode("ascii")
    else:
        payload = b"text:" + str(value).encode("utf-8", errors="surrogatepass")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _logical_hash(db: sqlite3.Connection) -> str:
    """Hash the v1 logical content, including schema and vector BLOBs."""

    digest = hashlib.sha256()
    for table in CORE_TABLES:
        columns = _columns(db, table)
        _hash_item(digest, table)
        for row in db.execute(f"PRAGMA table_info({table})"):
            for key in ("name", "type", "notnull", "dflt_value", "pk"):
                _hash_item(digest, row[key])
        order = "id" if "id" in columns else "rowid"
        for row in db.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            for column in columns:
                _hash_item(digest, row[column])
    return digest.hexdigest()


def _counts(db: sqlite3.Connection) -> dict[str, int]:
    counts = {
        table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in CORE_TABLES
    }
    counts["vectors"] = int(
        db.execute(
            "SELECT COUNT(*) FROM stories WHERE embedding IS NOT NULL"
        ).fetchone()[0]
    )
    return counts


def _relation_snapshot(db: sqlite3.Connection) -> dict:
    return {
        "sessions": [
            row[0] for row in db.execute("SELECT id FROM sessions ORDER BY id")
        ],
        "stories": [
            tuple(row)
            for row in db.execute(
                """SELECT id, parent_id, source_session_ids
                   FROM stories ORDER BY id"""
            )
        ],
        "edges": [
            tuple(row)
            for row in db.execute(
                """SELECT id, source_id, target_id, weight, edge_type
                   FROM edges ORDER BY id"""
            )
        ],
    }


def _existing_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {table: 0 for table in CORE_TABLES}
    try:
        db = _read_only(path)
    except (OSError, sqlite3.Error):
        return {table: 0 for table in CORE_TABLES}
    try:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        return {
            table: (
                int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                if table in tables else 0
            )
            for table in CORE_TABLES
        }
    finally:
        db.close()


def discover_legacy_databases(
    *,
    cwd: Path | None = None,
    current_database: Path | None = None,
) -> list[Path]:
    """Discover conventional project-level v1 database locations read-only."""

    root = (cwd or Path.cwd()).resolve(strict=False)
    current = (current_database or config.DB_PATH).resolve(strict=False)
    candidates = {
        root / "data" / "memory.db",
        root / "memory.db",
        config.BASE_DIR / "data" / "memory.db",
    }
    found: list[Path] = []
    for candidate in sorted(candidates, key=str):
        if not candidate.is_file() or candidate.resolve(strict=False) == current:
            continue
        try:
            db = _read_only(candidate)
            try:
                _validate_v1_schema(db)
            finally:
                db.close()
        except (MigrationError, OSError, sqlite3.Error):
            continue
        found.append(candidate.resolve())
    return found


def _atomic_json(path: Path, payload: dict) -> None:
    _private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _private_file(path)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _read_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(
            "SB_MIGRATION_MANIFEST_INVALID",
            f"迁移清单损坏: {exc}",
            hint="从备份恢复 manifest.json 后重试",
        ) from exc
    if not isinstance(payload, dict):
        raise MigrationError(
            "SB_MIGRATION_MANIFEST_INVALID", "迁移清单根节点必须是 object"
        )
    return payload


def _copy_read_only_database(source: Path, destination: Path) -> None:
    source_db = _read_only(source)
    destination_db = sqlite3.connect(str(destination))
    try:
        source_db.backup(destination_db)
        destination_db.commit()
    finally:
        destination_db.close()
        source_db.close()
    _private_file(destination)


def _seal_sqlite(path: Path) -> None:
    db = store.get_db(path)
    try:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("PRAGMA journal_mode=DELETE")
        db.commit()
    finally:
        db.close()
    path.with_name(path.name + "-wal").unlink(missing_ok=True)
    path.with_name(path.name + "-shm").unlink(missing_ok=True)


def _validate_transformed(source: Path, target: Path, profile_id: str) -> dict:
    source_db = _read_only(source)
    target_db = sqlite3.connect(str(target))
    target_db.row_factory = sqlite3.Row
    target_db.enable_load_extension(True)
    sqlite_vec.load(target_db)
    target_db.enable_load_extension(False)
    try:
        source_counts = _counts(source_db)
        target_counts = {
            table: int(
                target_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in CORE_TABLES
        }
        vector_count = int(
            target_db.execute("SELECT COUNT(*) FROM story_vectors").fetchone()[0]
        )
        target_counts["vectors"] = vector_count
        if source_counts != target_counts:
            raise MigrationError(
                "SB_MIGRATION_COUNT_MISMATCH",
                f"迁移数量不一致: source={source_counts}, target={target_counts}",
                hint="旧库仍未切换；检查磁盘空间后重试",
            )
        if _relation_snapshot(source_db) != _relation_snapshot(target_db):
            raise MigrationError(
                "SB_MIGRATION_RELATION_MISMATCH",
                "Session/Story/edge 关系未被完整保留",
            )
        source_content = {
            int(row["id"]): row["content"]
            for row in source_db.execute("SELECT id, content FROM stories")
        }
        target_content = {
            int(row["id"]): row["legacy_raw"]
            for row in target_db.execute("SELECT id, legacy_raw FROM stories")
        }
        if source_content != target_content:
            raise MigrationError(
                "SB_MIGRATION_LEGACY_RAW_MISMATCH",
                "v1 content 未完整保留到 legacy_raw",
            )
        source_embeddings = {
            int(row["id"]): row["embedding"]
            for row in source_db.execute("SELECT id, embedding FROM stories")
        }
        target_embeddings = {
            int(row["id"]): row["embedding"]
            for row in target_db.execute("SELECT id, embedding FROM stories")
        }
        if source_embeddings != target_embeddings:
            raise MigrationError(
                "SB_MIGRATION_VECTOR_MISMATCH",
                "v1 embedding BLOB 未被逐字节保留",
            )
        bad_profiles = sum(
            int(target_db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE profile_id != ?",
                (profile_id,),
            ).fetchone()[0])
            for table in CORE_TABLES
        )
        if bad_profiles:
            raise MigrationError(
                "SB_MIGRATION_PROFILE_MISMATCH",
                "迁移对象未全部归属目标 Profile",
            )
        pending_abstracts = int(target_db.execute(
            "SELECT COUNT(*) FROM stories WHERE abstract_status = 'pending'"
        ).fetchone()[0])
        if pending_abstracts != source_counts["stories"]:
            raise MigrationError(
                "SB_MIGRATION_ABSTRACT_STATE_INVALID",
                "legacy Story 未全部标记为待异步补 abstract",
            )
        first_vector = target_db.execute(
            "SELECT id, embedding FROM stories WHERE embedding IS NOT NULL ORDER BY id LIMIT 1"
        ).fetchone()
        if first_vector is not None:
            hit = target_db.execute(
                """SELECT story_id FROM story_vectors
                   WHERE embedding MATCH ? AND k = 1""",
                (first_vector["embedding"],),
            ).fetchone()
            if hit is None:
                raise MigrationError(
                    "SB_MIGRATION_VECTOR_UNSEARCHABLE",
                    "保留向量无法从 sqlite-vec 检索",
                )
        integrity = str(target_db.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = target_db.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok" or foreign_keys:
            raise MigrationError(
                "SB_MIGRATION_INTEGRITY_FAILED",
                f"目标库完整性校验失败: integrity={integrity}, fk={len(foreign_keys)}",
            )
        return {
            "counts": target_counts,
            "integrity": integrity,
            "foreign_key_violations": 0,
            "vector_searchable": True,
        }
    finally:
        target_db.close()
        source_db.close()


class MigrationManager:
    def __init__(self, registry: ProfileRegistry | None = None) -> None:
        self.registry = registry or config.PROFILE_REGISTRY

    def _profile(self, *, create: bool) -> Profile:
        profile = (
            self.registry.active_profile()
            if create else self.registry.peek_active_profile()
        )
        if profile is None:
            raise MigrationError(
                "SB_MIGRATION_PROFILE_MISSING",
                "Profile registry 尚未初始化",
                hint="先运行 storybook setup，再执行迁移",
            )
        return profile

    def _paths(self, profile: Profile) -> tuple[Path, Path]:
        root = self.registry.paths_for(profile).root / "migrations"
        return root, root / ".migration.lock"

    def plan(self, source: str | Path) -> dict:
        """Build a zero-write migration plan."""

        profile = self._profile(create=False)
        try:
            source_path = Path(source).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise MigrationError(
                "SB_MIGRATION_SOURCE_NOT_FOUND", f"旧数据库不存在: {source}"
            ) from exc
        active_path = self.registry.paths_for(profile).database.resolve(strict=False)
        if source_path == active_path:
            raise MigrationError(
                "SB_MIGRATION_SOURCE_ACTIVE",
                "不能原地迁移当前活动数据库",
                hint="请选择旧项目下的 data/memory.db",
            )
        try:
            source_db = _read_only(source_path)
            try:
                _validate_v1_schema(source_db)
                source_hash = _logical_hash(source_db)
                counts = _counts(source_db)
            finally:
                source_db.close()
        except sqlite3.Error as exc:
            raise MigrationError(
                "SB_MIGRATION_SOURCE_UNREADABLE", f"无法只读打开旧数据库: {exc}"
            ) from exc

        migration_id = str(uuid.uuid5(
            MIGRATION_NAMESPACE, f"{profile.id}:{source_hash}"
        ))
        generation_ref = f"migrations/{migration_id}/v2.db"
        backup_ref = f"migrations/{migration_id}/v1-backup.db"
        rollback_ref = f"migrations/{migration_id}/rollback-v1.db"
        root, _ = self._paths(profile)
        manifest = _read_manifest(root / migration_id / "manifest.json")
        target_counts = _existing_counts(active_path)
        already_applied = bool(
            manifest
            and manifest.get("source_hash") == source_hash
            and profile.database_ref == generation_ref
            and (root.parent / generation_ref).is_file()
        )
        if not already_applied and manifest is None and any(target_counts.values()):
            raise MigrationError(
                "SB_MIGRATION_TARGET_NOT_EMPTY",
                f"当前 Profile 已含数据: {target_counts}",
                hint="迁移到新的空 Profile，避免覆盖现有用户级记忆",
            )
        return {
            "migration_id": migration_id,
            "source": str(source_path),
            "source_hash": source_hash,
            "source_schema": "v1",
            "target_schema": "v2",
            "counts": counts,
            "profile_id": profile.id,
            "current_database_ref": profile.database_ref,
            "generation_ref": generation_ref,
            "backup_ref": backup_ref,
            "rollback_ref": rollback_ref,
            "already_applied": already_applied,
            "dry_run": True,
            "writes": [] if already_applied else [
                backup_ref, generation_ref, "profile registry database_ref"
            ],
        }

    def run(self, source: str | Path) -> dict:
        profile = self._profile(create=True)
        root, lock_path = self._paths(profile)
        with _locked(lock_path):
            plan = self.plan(source)
            migration_id = plan["migration_id"]
            generation_dir = root / migration_id
            manifest_path = generation_dir / "manifest.json"
            destination = root.parent / plan["generation_ref"]
            backup = root.parent / plan["backup_ref"]
            existing = _read_manifest(manifest_path)

            if plan["already_applied"]:
                if existing and existing.get("status") != "activated":
                    existing.update({
                        "status": "activated",
                        "activated_at": _timestamp(_utc_now()),
                    })
                    _atomic_json(manifest_path, existing)
                return {**plan, "dry_run": False, "status": "already_applied"}
            if existing and destination.is_file():
                validation = _validate_transformed(
                    backup, destination, profile.id
                )
                self.registry.set_profile_database(
                    profile.id, plan["generation_ref"]
                )
                if self.registry is config.PROFILE_REGISTRY:
                    config.refresh_profile(profile.id)
                existing.update({
                    "status": "activated",
                    "activated_at": _timestamp(_utc_now()),
                })
                _atomic_json(manifest_path, existing)
                return {
                    **plan, "dry_run": False, "status": "reapplied",
                    "validation": validation,
                }

            _private_dir(generation_dir)
            source_path = Path(plan["source"])
            backup_tmp = generation_dir / f".v1-backup.{uuid.uuid4().hex}.tmp.db"
            stage = generation_dir / f".v2.{uuid.uuid4().hex}.tmp.db"
            try:
                _copy_read_only_database(source_path, backup_tmp)
                backup_db = _read_only(backup_tmp)
                try:
                    _validate_v1_schema(backup_db)
                    snapshot_hash = _logical_hash(backup_db)
                finally:
                    backup_db.close()
                if snapshot_hash != plan["source_hash"]:
                    raise MigrationError(
                        "SB_MIGRATION_SOURCE_CHANGED",
                        "旧数据库在计划与备份之间发生变化",
                        hint="未切换 Profile；重新运行 dry-run 后再迁移",
                    )
                os.replace(backup_tmp, backup)
                _readonly_backup(backup)
                shutil.copy2(backup, stage)
                _private_file(stage)
                store.init_db(
                    stage,
                    profile_id=profile.id,
                    identity_namespace=migration_id,
                )
                retain_until = _timestamp(
                    _utc_now() + timedelta(days=BACKUP_RETENTION_DAYS)
                )
                history_db = store.get_db(stage)
                try:
                    history_db.execute(
                        """INSERT OR REPLACE INTO migration_history (
                               migration_id, source_hash, source_schema,
                               target_schema, source_counts_json, backup_ref,
                               retain_until
                           ) VALUES (?, ?, 'v1', 'v2', ?, ?, ?)""",
                        (
                            migration_id,
                            plan["source_hash"],
                            json.dumps(plan["counts"], sort_keys=True),
                            plan["backup_ref"],
                            retain_until,
                        ),
                    )
                    history_db.commit()
                finally:
                    history_db.close()
                validation = _validate_transformed(backup, stage, profile.id)
                _seal_sqlite(stage)
                os.replace(stage, destination)
                _private_file(destination)
                now = _timestamp(_utc_now())
                manifest = {
                    "migration_id": migration_id,
                    "source_hash": plan["source_hash"],
                    "source_schema": "v1",
                    "target_schema": "v2",
                    "profile_id": profile.id,
                    "previous_database_ref": profile.database_ref,
                    "generation_ref": plan["generation_ref"],
                    "backup_ref": plan["backup_ref"],
                    "rollback_ref": plan["rollback_ref"],
                    "counts": plan["counts"],
                    "validation": validation,
                    "retain_until": retain_until,
                    "status": "validated",
                    "created_at": now,
                }
                _atomic_json(manifest_path, manifest)
                self.registry.set_profile_database(
                    profile.id, plan["generation_ref"]
                )
                if self.registry is config.PROFILE_REGISTRY:
                    config.refresh_profile(profile.id)
                manifest.update({"status": "activated", "activated_at": now})
                _atomic_json(manifest_path, manifest)
                return {
                    **plan,
                    "dry_run": False,
                    "status": "applied",
                    "retain_until": retain_until,
                    "validation": validation,
                }
            finally:
                backup_tmp.unlink(missing_ok=True)
                stage.unlink(missing_ok=True)
                stage.with_name(stage.name + "-wal").unlink(missing_ok=True)
                stage.with_name(stage.name + "-shm").unlink(missing_ok=True)

    def rollback(self, migration_id: str) -> dict:
        profile = self._profile(create=True)
        root, lock_path = self._paths(profile)
        generation_dir = root / migration_id
        manifest_path = generation_dir / "manifest.json"
        with _locked(lock_path):
            manifest = _read_manifest(manifest_path)
            if manifest is None or manifest.get("profile_id") != profile.id:
                raise MigrationError(
                    "SB_MIGRATION_NOT_FOUND", f"迁移记录不存在: {migration_id}"
                )
            backup = root.parent / str(manifest["backup_ref"])
            rollback_ref = str(manifest["rollback_ref"])
            rollback_db = root.parent / rollback_ref
            if profile.database_ref == rollback_ref and rollback_db.is_file():
                return {
                    "migration_id": migration_id,
                    "status": "already_rolled_back",
                    "database_ref": rollback_ref,
                }
            if profile.database_ref != manifest["generation_ref"]:
                raise MigrationError(
                    "SB_MIGRATION_ACTIVE_GENERATION_CHANGED",
                    "当前 Profile 已不再指向该迁移生成库",
                    hint="先查看 storybook migration status，避免覆盖后续切换",
                )
            if not backup.is_file():
                raise MigrationError(
                    "SB_MIGRATION_BACKUP_MISSING",
                    "保留的 v1 备份不存在，无法回滚",
                )
            backup_db = _read_only(backup)
            try:
                if _logical_hash(backup_db) != manifest["source_hash"]:
                    raise MigrationError(
                        "SB_MIGRATION_BACKUP_HASH_MISMATCH",
                        "v1 备份 hash 与迁移记录不一致",
                    )
            finally:
                backup_db.close()
            rollback_tmp = generation_dir / f".rollback.{uuid.uuid4().hex}.tmp.db"
            try:
                _copy_read_only_database(backup, rollback_tmp)
                os.replace(rollback_tmp, rollback_db)
                _private_file(rollback_db)
            finally:
                rollback_tmp.unlink(missing_ok=True)
            self.registry.set_profile_database(profile.id, rollback_ref)
            if self.registry is config.PROFILE_REGISTRY:
                config.refresh_profile(profile.id)
            manifest.update({
                "status": "rolled_back",
                "rolled_back_at": _timestamp(_utc_now()),
            })
            _atomic_json(manifest_path, manifest)
            return {
                "migration_id": migration_id,
                "status": "rolled_back",
                "database_ref": rollback_ref,
                "counts": manifest["counts"],
            }

    def status(self) -> dict:
        profile = self._profile(create=False)
        root, _ = self._paths(profile)
        migrations: list[dict] = []
        if root.is_dir():
            for manifest_path in sorted(root.glob("*/manifest.json")):
                manifest = _read_manifest(manifest_path)
                if manifest:
                    migrations.append(manifest)
        return {
            "profile_id": profile.id,
            "database_ref": profile.database_ref,
            "migrations": migrations,
        }

    def delete_backup(self, migration_id: str) -> dict:
        """Delete a retained backup only after an explicit user command."""

        profile = self._profile(create=True)
        root, lock_path = self._paths(profile)
        manifest_path = root / migration_id / "manifest.json"
        with _locked(lock_path):
            manifest = _read_manifest(manifest_path)
            if manifest is None or manifest.get("profile_id") != profile.id:
                raise MigrationError(
                    "SB_MIGRATION_NOT_FOUND", f"迁移记录不存在: {migration_id}"
                )
            if profile.database_ref == manifest.get("rollback_ref"):
                raise MigrationError(
                    "SB_MIGRATION_BACKUP_ACTIVE",
                    "当前 Profile 正使用该迁移的 rollback 数据库，不能删除备份",
                )
            backup = root.parent / str(manifest["backup_ref"])
            existed = backup.is_file()
            backup.unlink(missing_ok=True)
            manifest.update({
                "backup_deleted": True,
                "backup_deleted_at": _timestamp(_utc_now()),
            })
            _atomic_json(manifest_path, manifest)
            return {
                "migration_id": migration_id,
                "status": "backup_deleted" if existed else "backup_already_missing",
            }
