"""v1 -> v2 migration safety, idempotency, integrity and rollback tests."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from storybook import config, migration, store
from storybook.cli import cli
from storybook.profiles import PlatformRoots, ProfileRegistry
from ._helpers import basis


def _roots(tmp_path: Path) -> PlatformRoots:
    return PlatformRoots(
        config=tmp_path / "config",
        data=tmp_path / "data",
        cache=tmp_path / "cache",
        state=tmp_path / "state",
        logs=tmp_path / "logs",
    )


@pytest.fixture
def migration_profile(tmp_path):
    old_registry = config.PROFILE_REGISTRY
    old_profile_id = config.PROFILE_ID
    registry = ProfileRegistry(
        tmp_path / "config" / "profiles.json", roots=_roots(tmp_path)
    )
    config.PROFILE_REGISTRY = registry
    config.refresh_profile()
    try:
        yield registry
    finally:
        config.PROFILE_REGISTRY = old_registry
        config.refresh_profile(old_profile_id)


def _legacy_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(
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
            UNIQUE(source_id, target_id)
        );
        """
    )
    db.executemany(
        """INSERT INTO sessions (
               source, raw_content, problem_desc, status
           ) VALUES (?, ?, ?, ?)""",
        [
            ("claude_code", "raw-1", "problem-1", "processed"),
            ("mystery-import", "raw-2", "problem-2", "processed"),
        ],
    )
    db.executemany(
        """INSERT INTO stories (
               title, content, keywords, embedding, parent_id,
               source_session_ids
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (
                "root", "legacy root", '["root"]',
                np.asarray(basis(0), dtype=np.float32).tobytes(), None, "[1]",
            ),
            (
                "child", "legacy child", '["child"]',
                np.asarray(basis(1), dtype=np.float32).tobytes(), 1, "[1, 2]",
            ),
            ("no-vector", "legacy plain", "[]", None, None, "[2]"),
        ],
    )
    db.executemany(
        """INSERT INTO edges (source_id, target_id, weight, edge_type)
           VALUES (?, ?, ?, ?)""",
        [(1, 2, 0.9, "parent_child"), (1, 3, 0.4, "semantic")],
    )
    db.commit()
    db.close()
    return path


def _inject_commit_failure(
    monkeypatch,
    *,
    failure_index: int,
    failure_mode: str,
) -> None:
    real_commit = migration._commit_guard
    calls = 0

    def fail_selected_commit(db: sqlite3.Connection) -> None:
        nonlocal calls
        calls += 1
        should_fail = calls == failure_index
        if should_fail and failure_mode == "before":
            raise OSError("injected commit failure before persistence")
        real_commit(db)
        if should_fail and failure_mode == "after":
            raise OSError("injected commit failure after persistence")

    monkeypatch.setattr(migration, "_commit_guard", fail_selected_commit)


def test_dry_run_is_zero_write_and_reports_stable_plan(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    registry_before = migration_profile.path.read_bytes()
    source_before = source.read_bytes()

    manager = migration.MigrationManager()
    first = manager.plan(source)
    second = manager.plan(source)

    assert first == second
    assert first["dry_run"] is True
    assert first["counts"] == {
        "sessions": 2, "stories": 3, "edges": 2, "vectors": 2,
    }
    assert migration_profile.path.read_bytes() == registry_before
    assert source.read_bytes() == source_before
    assert not (migration_profile.active_paths().root / "migrations").exists()


def test_run_preserves_objects_relations_vectors_and_is_idempotent(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    source_hash_before = migration._hash_database(source)
    manager = migration.MigrationManager()

    result = manager.run(source)

    assert result["status"] == "applied"
    assert result["validation"] == {
        "counts": {
            "sessions": 2, "stories": 3, "edges": 2, "vectors": 2,
        },
        "integrity": "ok",
        "foreign_key_violations": 0,
        "vector_searchable": True,
    }
    active = migration_profile.active_profile()
    assert active.database_ref == result["generation_ref"]
    assert config.DB_PATH == migration_profile.paths_for(active).database
    assert migration._hash_database(source) == source_hash_before
    retired = sqlite3.connect(source)
    try:
        with pytest.raises(
            sqlite3.DatabaseError, match="SB_MIGRATION_GENERATION_FENCED"
        ):
            retired.execute(
                """INSERT INTO sessions (source, raw_content)
                   VALUES ('retired-writer', 'must fail')"""
            )
    finally:
        retired.rollback()
        retired.close()

    migrated = store.get_db()
    try:
        assert migrated.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert migrated.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 3
        assert migrated.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2
        for table in ("sessions", "stories", "edges"):
            assert all(
                uuid.UUID(row[0]).version == 7
                for row in migrated.execute(
                    f"SELECT global_id FROM {table} ORDER BY id"
                )
            )
        assert [
            tuple(row) for row in migrated.execute(
                "SELECT id, parent_id, source_session_ids FROM stories ORDER BY id"
            )
        ] == [(1, None, "[1]"), (2, 1, "[1, 2]"), (3, None, "[2]")]
        stories = migrated.execute(
            """SELECT legacy_raw, abstract, abstract_status,
                      embedding_status FROM stories ORDER BY id"""
        ).fetchall()
        assert [row["legacy_raw"] for row in stories] == [
            "legacy root", "legacy child", "legacy plain",
        ]
        assert {row["abstract_status"] for row in stories} == {"pending"}
        assert {row["abstract"] for row in stories} == {""}
        contexts = [
            json.loads(row[0])
            for row in migrated.execute("SELECT context_json FROM sessions ORDER BY id")
        ]
        assert contexts[0]["tool"]["type"] == "claude_code"
        assert contexts[0]["provenance"]["tool.type"] == "reported"
        assert contexts[1]["tool"]["type"] is None
        assert all(item["runtime"]["kind"] == "unknown" for item in contexts)
        assert migrated.execute(
            "SELECT COUNT(*) FROM migration_history"
        ).fetchone()[0] == 1
    finally:
        migrated.close()

    repeated = manager.run(source)
    assert repeated["status"] == "already_applied"
    assert store.count_sessions() == 2
    assert store.count_stories() == 3


def test_failure_before_cutover_keeps_registry_and_source_authoritative(
    tmp_path, migration_profile, monkeypatch
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    source_before = source.read_bytes()
    original_ref = migration_profile.active_profile().database_ref

    def fail_validation(*args, **kwargs):
        raise migration.MigrationError(
            "SB_MIGRATION_TEST_FAILURE", "injected validation failure"
        )

    monkeypatch.setattr(migration, "_validate_transformed", fail_validation)
    with pytest.raises(migration.MigrationError, match="injected"):
        migration.MigrationManager().run(source)

    assert migration_profile.active_profile().database_ref == original_ref
    assert source.read_bytes() == source_before
    db = sqlite3.connect(source)
    try:
        assert db.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 3
    finally:
        db.close()


def test_wal_commit_after_backup_aborts_before_cutover(
    tmp_path, migration_profile, monkeypatch
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    original_ref = migration_profile.active_profile().database_ref
    original_copy = migration._copy_read_only_database
    committed = False

    def copy_then_commit(source_path, destination):
        nonlocal committed
        original_copy(source_path, destination)
        if Path(source_path).resolve() != source.resolve() or committed:
            return
        writer = sqlite3.connect(source)
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute(
                """INSERT INTO sessions (source, raw_content, problem_desc)
                   VALUES ('late-writer', 'raw-3', 'problem-3')"""
            )
            writer.commit()
            committed = True
        finally:
            writer.close()

    monkeypatch.setattr(
        migration, "_copy_read_only_database", copy_then_commit
    )

    with pytest.raises(migration.MigrationError) as exc_info:
        migration.MigrationManager().run(source)

    assert exc_info.value.code == "SB_MIGRATION_SOURCE_CHANGED"
    assert migration_profile.active_profile().database_ref == original_ref
    source_db = sqlite3.connect(source)
    try:
        assert source_db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
    finally:
        source_db.close()


def test_active_source_writer_blocks_cutover_without_switching(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    original_ref = migration_profile.active_profile().database_ref
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        """INSERT INTO sessions (source, raw_content, problem_desc)
           VALUES ('uncommitted-writer', 'raw-3', 'problem-3')"""
    )
    try:
        with pytest.raises(migration.MigrationError) as exc_info:
            migration.MigrationManager().run(source)
    finally:
        writer.rollback()
        writer.close()

    assert exc_info.value.code == "SB_MIGRATION_SOURCE_BUSY"
    assert migration_profile.active_profile().database_ref == original_ref


def test_waiting_source_writer_is_fenced_after_successful_cutover(
    tmp_path, migration_profile, monkeypatch
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    wal = sqlite3.connect(source)
    wal.execute("PRAGMA journal_mode=WAL")
    wal.close()
    real_switch = migration_profile.set_profile_database
    writer_started = threading.Event()
    writer_result: dict[str, object] = {}
    writer_thread: threading.Thread | None = None

    def write_to_open_source() -> None:
        writer = sqlite3.connect(source, timeout=5)
        try:
            writer_started.set()
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                """INSERT INTO sessions (source, raw_content, problem_desc)
                   VALUES ('waiting-writer', 'raw-3', 'problem-3')"""
            )
            writer.commit()
            writer_result["committed"] = True
        except sqlite3.DatabaseError as exc:
            writer.rollback()
            writer_result["error"] = str(exc)
        finally:
            writer.close()

    def switch_with_waiting_writer(*args, **kwargs):
        nonlocal writer_thread
        writer_thread = threading.Thread(target=write_to_open_source)
        writer_thread.start()
        assert writer_started.wait(timeout=5)
        return real_switch(*args, **kwargs)

    monkeypatch.setattr(
        migration_profile, "set_profile_database", switch_with_waiting_writer
    )

    applied = migration.MigrationManager().run(source)
    assert writer_thread is not None
    writer_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert writer_result.get("committed") is not True
    assert "SB_MIGRATION_GENERATION_FENCED" in str(writer_result.get("error"))
    assert migration_profile.active_profile().database_ref == applied["generation_ref"]
    source_db = sqlite3.connect(source)
    try:
        assert source_db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    finally:
        source_db.close()


def test_fence_staging_rolls_back_when_registry_switch_fails(
    tmp_path, migration_profile, monkeypatch
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    original_ref = migration_profile.active_profile().database_ref

    def fail_switch(*args, **kwargs):
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(migration_profile, "set_profile_database", fail_switch)

    with pytest.raises(migration.MigrationError) as exc_info:
        migration.MigrationManager().run(source)

    assert exc_info.value.code == "SB_MIGRATION_SWITCH_PREPARE_FAILED"
    assert migration_profile.active_profile().database_ref == original_ref
    writer = sqlite3.connect(source)
    try:
        writer.execute(
            """INSERT INTO sessions (source, raw_content, problem_desc)
               VALUES ('retryable-writer', 'raw-3', 'problem-3')"""
        )
        writer.commit()
        assert writer.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
    finally:
        writer.close()


def test_registry_error_after_durable_pointer_is_recovered_as_success(
    tmp_path, migration_profile, monkeypatch
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    real_switch = migration_profile.set_profile_database

    def switch_then_report_failure(*args, **kwargs):
        real_switch(*args, **kwargs)
        raise OSError("injected error after atomic registry replace")

    monkeypatch.setattr(
        migration_profile, "set_profile_database", switch_then_report_failure
    )

    applied = migration.MigrationManager().run(source)

    assert applied["status"] == "applied"
    assert migration_profile.active_profile().database_ref == applied["generation_ref"]
    assert config.DB_PATH == migration_profile.paths_for(
        migration_profile.active_profile()
    ).database
    assert migration.MigrationManager().status()["migrations"][0][
        "status"
    ] == "activated"
    retired = sqlite3.connect(source)
    try:
        with pytest.raises(
            sqlite3.DatabaseError, match="SB_MIGRATION_GENERATION_FENCED"
        ):
            retired.execute(
                """INSERT INTO sessions (source, raw_content)
                   VALUES ('ambiguous-registry', 'must fail')"""
            )
    finally:
        retired.rollback()
        retired.close()


@pytest.mark.parametrize("failure_mode", ["before", "after"])
def test_cutover_commit_failure_restores_old_authority(
    tmp_path, migration_profile, monkeypatch, failure_mode
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    original = migration_profile.active_profile()
    original_path = migration_profile.paths_for(original).database
    _inject_commit_failure(
        monkeypatch,
        failure_index=1,
        failure_mode=failure_mode,
    )

    with pytest.raises(migration.MigrationError) as exc_info:
        manager.run(source)

    assert exc_info.value.code == "SB_MIGRATION_SWITCH_PREPARE_FAILED"
    assert migration_profile.active_profile().database_ref == original.database_ref
    assert config.DB_PATH == original_path
    assert manager.status()["migrations"][0]["status"] == "validated"
    writer = sqlite3.connect(source)
    try:
        writer.execute(
            """INSERT INTO sessions (source, raw_content, problem_desc)
               VALUES ('cutover-recovery', 'raw-3', 'problem-3')"""
        )
        writer.commit()
        assert writer.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
    finally:
        writer.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX read-only permission bits")
def test_read_only_source_uses_hash_cas_without_mutation(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    source_before = source.read_bytes()
    source.chmod(0o400)

    result = migration.MigrationManager().run(source)

    assert result["status"] == "applied"
    assert source.read_bytes() == source_before


def test_rollback_atomically_points_to_independent_v1_copy(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    applied = manager.run(source)

    rolled_back = manager.rollback(applied["migration_id"])

    assert rolled_back["status"] == "rolled_back"
    active = migration_profile.active_profile()
    assert active.database_ref == applied["rollback_ref"]
    assert config.DB_PATH == migration_profile.paths_for(active).database
    db = sqlite3.connect(config.DB_PATH)
    try:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2
        assert "abstract" not in {
            row[1] for row in db.execute("PRAGMA table_info(stories)")
        }
    finally:
        db.close()
    assert manager.rollback(applied["migration_id"])["status"] == (
        "already_rolled_back"
    )


def test_rollback_rejects_active_v2_writer_without_switching(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    applied = manager.run(source)
    writer = sqlite3.connect(config.DB_PATH)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "UPDATE stories SET access_count = access_count + 1 WHERE id = 1"
    )
    try:
        with pytest.raises(migration.MigrationError) as exc_info:
            manager.rollback(applied["migration_id"])
        assert exc_info.value.code == "SB_MIGRATION_AUTHORITY_BUSY"
        assert (
            migration_profile.active_profile().database_ref
            == applied["generation_ref"]
        )
        writer.commit()
    finally:
        writer.close()

    active = sqlite3.connect(config.DB_PATH)
    try:
        assert active.execute(
            "SELECT access_count FROM stories WHERE id = 1"
        ).fetchone()[0] == 1
    finally:
        active.close()


def test_waiting_v2_writer_is_fenced_after_successful_rollback(
    tmp_path, migration_profile, monkeypatch
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    applied = manager.run(source)
    old_v2 = config.DB_PATH
    wal = sqlite3.connect(old_v2)
    wal.execute("PRAGMA journal_mode=WAL")
    wal.close()
    real_switch = migration_profile.set_profile_database
    writer_started = threading.Event()
    writer_result: dict[str, object] = {}
    writer_thread: threading.Thread | None = None

    def write_to_open_v2() -> None:
        writer = sqlite3.connect(old_v2, timeout=5)
        try:
            writer_started.set()
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "UPDATE stories SET access_count = access_count + 1 WHERE id = 1"
            )
            writer.commit()
            writer_result["committed"] = True
        except sqlite3.DatabaseError as exc:
            writer.rollback()
            writer_result["error"] = str(exc)
        finally:
            writer.close()

    def switch_with_waiting_writer(*args, **kwargs):
        nonlocal writer_thread
        writer_thread = threading.Thread(target=write_to_open_v2)
        writer_thread.start()
        assert writer_started.wait(timeout=5)
        return real_switch(*args, **kwargs)

    monkeypatch.setattr(
        migration_profile, "set_profile_database", switch_with_waiting_writer
    )

    rolled_back = manager.rollback(applied["migration_id"])
    assert writer_thread is not None
    writer_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert writer_result.get("committed") is not True
    assert "SB_MIGRATION_GENERATION_FENCED" in str(writer_result.get("error"))
    assert migration_profile.active_profile().database_ref == rolled_back["database_ref"]
    old_db = sqlite3.connect(old_v2)
    try:
        assert old_db.execute(
            "SELECT access_count FROM stories WHERE id = 1"
        ).fetchone()[0] == 0
    finally:
        old_db.close()


@pytest.mark.parametrize("failure_index", [1, 2])
@pytest.mark.parametrize("failure_mode", ["before", "after"])
def test_rollback_commit_failure_restores_active_v2(
    tmp_path,
    migration_profile,
    monkeypatch,
    failure_index,
    failure_mode,
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    applied = manager.run(source)
    active_v2 = config.DB_PATH
    _inject_commit_failure(
        monkeypatch,
        failure_index=failure_index,
        failure_mode=failure_mode,
    )

    with pytest.raises(migration.MigrationError) as exc_info:
        manager.rollback(applied["migration_id"])

    assert exc_info.value.code == "SB_MIGRATION_SWITCH_PREPARE_FAILED"
    assert (
        migration_profile.active_profile().database_ref
        == applied["generation_ref"]
    )
    assert config.DB_PATH == active_v2
    assert manager.status()["migrations"][0]["status"] == "activated"
    writer = sqlite3.connect(active_v2)
    try:
        writer.execute(
            "UPDATE stories SET access_count = access_count + 1 WHERE id = 1"
        )
        writer.commit()
        assert writer.execute(
            "SELECT access_count FROM stories WHERE id = 1"
        ).fetchone()[0] == 1
    finally:
        writer.close()


def test_reapply_rejects_changes_written_after_rollback(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    applied = manager.run(source)
    manager.rollback(applied["migration_id"])
    rollback_ref = migration_profile.active_profile().database_ref

    authoritative = sqlite3.connect(config.DB_PATH)
    try:
        authoritative.execute(
            """INSERT INTO sessions (source, raw_content, problem_desc)
               VALUES ('rollback-writer', 'raw-3', 'problem-3')"""
        )
        authoritative.commit()
    finally:
        authoritative.close()

    with pytest.raises(migration.MigrationError) as exc_info:
        manager.run(source)

    assert exc_info.value.code == "SB_MIGRATION_AUTHORITY_CHANGED"
    assert migration_profile.active_profile().database_ref == rollback_ref
    current = sqlite3.connect(config.DB_PATH)
    try:
        assert current.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
    finally:
        current.close()


def test_reapply_allows_unchanged_rollback_authority(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    applied = manager.run(source)
    manager.rollback(applied["migration_id"])

    reapplied = manager.run(source)

    assert reapplied["status"] == "reapplied"
    assert migration_profile.active_profile().database_ref == applied["generation_ref"]
    assert store.count_sessions() == 2
    store.add_session("reapplied-writer", "new active v2 data")
    assert store.count_sessions() == 3


@pytest.mark.parametrize("failure_index", [1, 2, 3])
@pytest.mark.parametrize("failure_mode", ["before", "after"])
def test_reapply_commit_failure_restores_rollback_authority(
    tmp_path,
    migration_profile,
    monkeypatch,
    failure_index,
    failure_mode,
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    manager = migration.MigrationManager()
    applied = manager.run(source)
    manager.rollback(applied["migration_id"])
    rollback_ref = migration_profile.active_profile().database_ref
    rollback_db = config.DB_PATH
    target_v2 = (
        migration_profile.paths_for(migration_profile.active_profile()).root
        / applied["generation_ref"]
    )
    _inject_commit_failure(
        monkeypatch,
        failure_index=failure_index,
        failure_mode=failure_mode,
    )

    with pytest.raises(migration.MigrationError) as exc_info:
        manager.run(source)

    assert exc_info.value.code == "SB_MIGRATION_SWITCH_PREPARE_FAILED"
    assert migration_profile.active_profile().database_ref == rollback_ref
    assert config.DB_PATH == rollback_db
    assert manager.status()["migrations"][0]["status"] == "rolled_back"
    writer = sqlite3.connect(rollback_db)
    try:
        writer.execute(
            """INSERT INTO sessions (source, raw_content, problem_desc)
               VALUES ('reapply-recovery', 'raw-3', 'problem-3')"""
        )
        writer.commit()
        assert writer.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3
    finally:
        writer.close()
    retired_target = sqlite3.connect(target_v2)
    try:
        with pytest.raises(
            sqlite3.DatabaseError, match="SB_MIGRATION_GENERATION_FENCED"
        ):
            retired_target.execute(
                "UPDATE stories SET access_count = access_count + 1 WHERE id = 1"
            )
    finally:
        retired_target.rollback()
        retired_target.close()


def test_cli_dry_run_and_status_json(tmp_path, migration_profile):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    runner = CliRunner()

    dry_run = runner.invoke(
        cli, ["migration", "run", str(source), "--dry-run", "--json"]
    )

    assert dry_run.exit_code == 0, dry_run.output
    payload = json.loads(dry_run.output)
    assert payload["dry_run"] is True
    assert payload["counts"]["stories"] == 3
    assert not (migration_profile.active_paths().root / "migrations").exists()


def test_nonempty_profile_is_rejected_without_modification(
    tmp_path, migration_profile
):
    source = _legacy_db(tmp_path / "repo" / "data" / "memory.db")
    store.init_db()
    store.add_session("manual", "new profile memory")

    with pytest.raises(migration.MigrationError) as exc_info:
        migration.MigrationManager().plan(source)

    assert exc_info.value.code == "SB_MIGRATION_TARGET_NOT_EMPTY"
    assert store.count_sessions() == 1
