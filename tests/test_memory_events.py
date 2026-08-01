"""MemoryEvent, UUIDv7, tombstone, replay, privacy and local-only tests."""
from __future__ import annotations

import json
import sqlite3
import uuid

import pytest
import requests
from click.testing import CliRunner

from storybook import config, store
from storybook.cli import cli
from storybook.identifiers import new_uuid7
from ._helpers import basis


def _uuid7(value: str) -> uuid.UUID:
    parsed = uuid.UUID(value)
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
    return parsed


def test_new_entities_and_memory_events_use_monotonic_uuid7():
    session_id = store.add_session("test", "raw", "problem")
    first = store.add_story(
        "first", "content", ["one"], basis(0),
        source_session_ids=[session_id],
    )
    second = store.add_story("second", "content", ["two"], basis(1))
    store.add_or_update_edge(first, second, 0.5)

    session = store.get_session(session_id)
    stories = [store.get_story(first), store.get_story(second)]
    edges = store.get_edges(first)
    assert _uuid7(session["global_id"])
    assert all(_uuid7(story["global_id"]) for story in stories)
    assert _uuid7(edges[0]["global_id"])

    events = store.get_memory_events()
    assert len(events) == 2
    assert all(_uuid7(event["event_id"]) for event in events)
    assert all(_uuid7(event["device_id"]) for event in events)
    assert [event["event_id"] for event in events] == sorted(
        event["event_id"] for event in events
    )
    assert events[0]["base_version"] == 0
    assert events[0]["version"] == 1


def test_create_update_merge_and_split_have_contiguous_audit_events():
    story_id = store.add_story("title", "content", [], basis(0))
    assert store.update_story(
        story_id, title="merged", embedding=basis(1), event_type="merge"
    )
    assert store.update_story(
        story_id, sources=[], embedding=basis(2), event_type="split_source"
    )
    assert store.delete_story_vector(story_id)

    events = store.get_memory_events(story_id)
    assert [event["operation"] for event in events] == [
        "create", "merge", "split", "split"
    ]
    assert [event["base_version"] for event in events] == [0, 1, 2, 3]
    assert [event["version"] for event in events] == [1, 2, 3, 4]
    assert [revision["event_type"] for revision in store.get_story_revisions(story_id)] == [
        "create", "merge", "split_source", "split_parent"
    ]
    projection = store.replay_memory_events(events)
    entity_id = store.get_story(story_id)["global_id"]
    assert projection["entities"][entity_id]["version"] == 4
    assert projection["conflicts"] == []

    child_id = store.add_story(
        "child", "split detail", [], basis(3),
        parent_id=story_id, event_type="split_child",
    )
    child_event = store.get_memory_events(child_id)[0]
    assert child_event["operation"] == "split"
    assert child_event["base_version"] == 0
    assert child_event["payload"]["relationships"]["parent_entity_id"] == entity_id


def test_delete_writes_terminal_tombstone_and_replay_never_resurrects():
    story_id = store.add_story("title", "secret content", [], basis(0))
    entity_id = store.get_story(story_id)["global_id"]

    event_id = store.delete_story(story_id)
    assert _uuid7(event_id)
    assert store.delete_story(story_id) == event_id
    assert store.get_story(story_id) is None
    deleted = store.get_story(story_id, include_deleted=True)
    assert deleted["deleted_at"]
    assert deleted["tombstone_event_id"] == event_id
    assert deleted["embedding"] == []
    assert store.update_story(story_id, title="resurrected") is False
    assert store.count_stories() == 0
    assert store.search_by_vector(basis(0), top_k=5) == []
    assert store.search_by_lexical("title", top_k=5) == []

    tombstone = store.get_memory_tombstone(story_id)
    assert tombstone["deleted_event_id"] == event_id
    assert tombstone["deleted_version"] == 2

    events = store.get_memory_events(story_id)
    stale_recreate = dict(events[0])
    stale_recreate.update({
        "sequence": 999,
        "event_id": new_uuid7(),
        "base_version": 2,
        "version": 3,
        "operation": "create",
        "created_at": "2999-01-01T00:00:00Z",
    })
    projection = store.replay_memory_events(
        list(reversed(events)) + [stale_recreate]
    )
    assert entity_id not in projection["entities"]
    assert projection["tombstones"][entity_id]["event_id"] == event_id
    assert any(item["event_id"] == stale_recreate["event_id"] for item in projection["ignored"])


def test_event_envelope_excludes_story_text_paths_and_raw_external_ids():
    raw_external_id = "external-session-plain-secret"
    absolute_path = "/Users/alice/private/acme/credentials.txt"
    session_id = store.add_session(
        "adapter",
        f"opened {absolute_path}",
        "private task",
        context={
            "session": {"external_session_hash": raw_external_id},
            "workspace": {"cwd_alias": absolute_path},
        },
    )
    story_id = store.add_story(
        "plaintext-sensitive-title",
        f"read {absolute_path}",
        ["credential"],
        basis(0),
        source_session_ids=[session_id],
        sources=[{"session_id": session_id, "evidence": [absolute_path]}],
    )

    event = store.get_memory_events(story_id)[0]
    serialized = json.dumps(event, ensure_ascii=False, sort_keys=True)
    assert raw_external_id not in serialized
    assert absolute_path not in serialized
    assert "plaintext-sensitive-title" not in serialized
    assert event["payload"]["relationships"]["source_entity_ids"] == [
        store.get_session(session_id)["global_id"]
    ]
    assert len(event["payload"]["revision_sha256"]) == 64
    assert event["payload_ciphertext"] is None
    assert event["encryption_key_id"] is None


def test_event_and_tombstone_tables_are_append_only():
    story_id = store.add_story("title", "content", [], basis(0))
    event_id = store.delete_story(story_id)
    db = store.get_db(load_vector_extension=False)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="memory_events are immutable"):
            db.execute(
                "UPDATE memory_events SET operation = 'update' WHERE event_id = ?",
                (event_id,),
            )
        db.rollback()
        with pytest.raises(sqlite3.DatabaseError, match="memory_tombstones are immutable"):
            db.execute(
                "DELETE FROM memory_tombstones WHERE deleted_event_id = ?",
                (event_id,),
            )
    finally:
        db.close()


def test_story_mutation_rolls_back_if_audit_event_cannot_be_appended():
    story_id = store.add_story("before", "content", [], basis(0))
    db = store.get_db(load_vector_extension=False)
    try:
        db.execute(
            """CREATE TRIGGER reject_new_memory_events
               BEFORE INSERT ON memory_events BEGIN
                   SELECT RAISE(ABORT, 'event append rejected');
               END"""
        )
        db.commit()
    finally:
        db.close()

    with pytest.raises(sqlite3.DatabaseError, match="event append rejected"):
        store.update_story(story_id, title="after")

    story = store.get_story(story_id)
    assert story["title"] == "before"
    assert story["version"] == 1
    assert len(store.get_memory_events(story_id)) == 1


def test_sync_status_is_local_only_and_performs_no_network_request(monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("sync status attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", fail_network)
    before = store.get_memory_events()
    result = CliRunner().invoke(cli, ["sync", "status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "profile_id": config.PROFILE_ID,
        "sync_state": "local_only",
        "enabled": False,
        "message": "v0.2 stores all memory locally; cross-device sync is not enabled.",
    }
    assert store.get_memory_events() == before
