"""v1 -> v2 migration safety, idempotency, integrity and rollback tests."""
from __future__ import annotations

import json
import sqlite3
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
    source_before = source.read_bytes()
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
    assert source.read_bytes() == source_before

    migrated = store.get_db()
    try:
        assert migrated.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert migrated.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 3
        assert migrated.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2
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
