"""Privacy-safe MemoryEvent contracts and deterministic local replay.

The clear-text event envelope deliberately excludes Story text and raw source
locators.  It is sufficient for conflict detection, relationship routing and
deletion replay; the existing local revision chain remains the lossless audit
record.  A future sync transport can attach an encrypted revision payload
without coupling events to SQLite pages.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Iterable


SCHEMA_VERSION = 1
ENTITY_TYPE_STORY = "story"
OPERATIONS = frozenset({"create", "update", "merge", "split", "delete"})


def operation_for_event_type(event_type: str) -> str:
    """Map internal revision labels onto the stable wire operation enum."""

    normalized = (event_type or "update").strip().lower()
    if normalized == "create" or normalized == "migrate":
        return "create"
    if normalized == "merge":
        return "merge"
    if normalized.startswith("split"):
        return "split"
    if normalized == "delete":
        return "delete"
    return "update"


def parse_event(row: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready MemoryEvent dictionary from a database row."""

    event = dict(row)
    payload = event.pop("payload_json", event.get("payload", {}))
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    event["payload"] = payload if isinstance(payload, dict) else {}
    return event


def validate_event(event: dict[str, Any]) -> None:
    """Validate the portable fields required by replay and future transport."""

    parsed_ids = {}
    for field in ("event_id", "entity_id", "device_id"):
        try:
            parsed_ids[field] = uuid.UUID(str(event[field]))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid MemoryEvent {field}") from exc
    if parsed_ids["event_id"].version != 7:
        raise ValueError("MemoryEvent event_id must be UUIDv7")
    if event.get("entity_type") != ENTITY_TYPE_STORY:
        raise ValueError("unsupported MemoryEvent entity_type")
    if event.get("operation") not in OPERATIONS:
        raise ValueError("unsupported MemoryEvent operation")
    base_version = event.get("base_version")
    version = event.get("version")
    if not isinstance(base_version, int) or base_version < 0:
        raise ValueError("invalid MemoryEvent base_version")
    if not isinstance(version, int) or version <= base_version:
        raise ValueError("invalid MemoryEvent version")
    if not isinstance(event.get("created_at"), str) or not event["created_at"]:
        raise ValueError("invalid MemoryEvent created_at")


def replay(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Replay event envelopes into an in-memory version/tombstone projection.

    Tombstones are terminal in the v0.2 local-only model.  They are collected
    before active state projection, so shuffled input and a stale event that
    appears after a delete cannot resurrect the entity.
    """

    parsed = [parse_event(event) for event in events]
    for event in parsed:
        validate_event(event)
    ordered = sorted(
        parsed,
        key=lambda item: (
            item.get("sequence", 0), item["created_at"], item["event_id"]
        ),
    )

    tombstones: dict[str, dict[str, Any]] = {}
    for event in ordered:
        if event["operation"] != "delete":
            continue
        current = tombstones.get(event["entity_id"])
        if current is None or event["version"] > current["version"]:
            tombstones[event["entity_id"]] = {
                "event_id": event["event_id"],
                "version": event["version"],
                "device_id": event["device_id"],
                "deleted_at": event["created_at"],
            }

    entities: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for event in ordered:
        entity_id = event["entity_id"]
        if event["operation"] == "delete":
            entities.pop(entity_id, None)
            continue
        if entity_id in tombstones:
            ignored.append({
                "event_id": event["event_id"],
                "entity_id": entity_id,
                "reason": "tombstoned",
            })
            continue

        current = entities.get(entity_id)
        expected_base = current["version"] if current is not None else 0
        if event["base_version"] != expected_base:
            conflicts.append({
                "event_id": event["event_id"],
                "entity_id": entity_id,
                "expected_base_version": expected_base,
                "actual_base_version": event["base_version"],
            })
            continue
        entities[entity_id] = {
            "event_id": event["event_id"],
            "version": event["version"],
            "operation": event["operation"],
            "device_id": event["device_id"],
            "updated_at": event["created_at"],
            "payload": event["payload"],
        }

    return {
        "entities": entities,
        "tombstones": tombstones,
        "conflicts": conflicts,
        "ignored": ignored,
        "event_count": len(ordered),
    }
