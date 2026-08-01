"""Portable, time-sortable identifiers used by sync-ready entities.

Python 3.11 does not expose :func:`uuid.uuid7`, so Storybook implements the
RFC 9562 bit layout locally.  The generator is monotonic within a process when
multiple identifiers are requested in the same millisecond (or the wall clock
moves backwards), while retaining 74 random bits for collision resistance.
"""
from __future__ import annotations

import secrets
import threading
import time
import uuid


_UUID7_LOCK = threading.Lock()
_UUID7_LAST_MS = -1
_UUID7_RANDOM = 0
_UUID7_RANDOM_MASK = (1 << 74) - 1
_UUID7_TIMESTAMP_MASK = (1 << 48) - 1


def uuid7() -> uuid.UUID:
    """Return a standards-compliant, process-monotonic UUIDv7."""

    global _UUID7_LAST_MS, _UUID7_RANDOM

    timestamp_ms = time.time_ns() // 1_000_000
    with _UUID7_LOCK:
        if timestamp_ms > _UUID7_LAST_MS:
            _UUID7_LAST_MS = timestamp_ms
            _UUID7_RANDOM = secrets.randbits(74)
        else:
            timestamp_ms = _UUID7_LAST_MS
            _UUID7_RANDOM = (_UUID7_RANDOM + 1) & _UUID7_RANDOM_MASK
            if _UUID7_RANDOM == 0:
                timestamp_ms += 1
                _UUID7_LAST_MS = timestamp_ms
                _UUID7_RANDOM = secrets.randbits(74)

        rand_a = _UUID7_RANDOM >> 62
        rand_b = _UUID7_RANDOM & ((1 << 62) - 1)
        value = (
            ((timestamp_ms & _UUID7_TIMESTAMP_MASK) << 80)
            | (0x7 << 76)
            | (rand_a << 64)
            | (0b10 << 62)
            | rand_b
        )
        return uuid.UUID(int=value)


def new_uuid7() -> str:
    """Return the canonical string representation of a new UUIDv7."""

    return str(uuid7())
