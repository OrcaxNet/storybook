"""SDK history reads with bounded raw provenance checks; never start a turn."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import json
import hashlib

from claude_agent_sdk import get_session_messages, list_sessions
# Pinned SDK internals: mirror its exact/prefix directory lookup, without worktrees.
from claude_agent_sdk._internal.sessions import _find_project_dir, _canonicalize_path

from .store import HubError, redact


def discover(root: Path):
    root = root.resolve(strict=True)
    offset = 0
    while True:
        page = list_sessions(directory=str(root), limit=100, offset=offset, include_worktrees=False)
        for session in page:
            # Advisory only: the SDK may fill cwd from the requested directory.
            # collect() separately checks metadata actually present in the transcript.
            if session.cwd and Path(session.cwd).resolve() == root:
                yield session
        if len(page) < 100:
            return
        offset += len(page)
        if offset >= 10000:
            raise HubError("DISCOVERY_LIMIT_REACHED")


def verified_source(root, session_id):
    directory = _find_project_dir(_canonicalize_path(str(root)))
    if directory is None:
        raise HubError("SESSION_NOT_IN_APPROVED_ROOT", 403)
    path = directory / (session_id + ".jsonl")
    try:
        with path.open("rb") as source:
            raw = source.read(10 * 1024 * 1024 + 1)
    except OSError:
        raise HubError("SOURCE_UNAVAILABLE", 409)
    if len(raw) > 10 * 1024 * 1024:
        raise HubError("SOURCE_SIZE_UNVERIFIED_OR_TOO_LARGE", 413)
    proved = set()
    try:
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError("invalid record")
            cwd = entry.get("cwd")
            if cwd is not None and (not isinstance(cwd, str) or not Path(cwd).is_absolute()
                                    or Path(cwd).resolve() != root):
                raise HubError("SESSION_NOT_IN_APPROVED_ROOT", 403)
            if entry.get("type") in {"user", "assistant"} and "message" in entry:
                if not cwd or entry.get("sessionId") != session_id:
                    raise HubError("SOURCE_OWNERSHIP_UNVERIFIED", 403)
                proved.add(entry.get("uuid"))
    except (ValueError, TypeError, UnicodeError):
        raise HubError("SOURCE_METADATA_INVALID", 403)
    if not proved:
        raise HubError("SOURCE_OWNERSHIP_UNVERIFIED", 403)
    return hashlib.sha256(raw).hexdigest(), proved


def chunks(text):
    """60 KB chunks with 512-character overlap and normalized UTF-8 source ranges."""
    start, byte_start = 0, 0
    while start < len(text):
        end, size = start, 0
        while end < len(text) and size + len(text[end].encode()) <= 60000:
            size += len(text[end].encode())
            end += 1
        yield text[start:end], byte_start, byte_start + size
        if end == len(text):
            break
        next_start = end - 512
        byte_start += len(text[start:next_start].encode())
        start = next_start


def visible_text(message):
    content = message.get("content", [])
    if isinstance(content, str):
        return content, 0
    parts, omitted = [], 0
    for block in content:
        if not isinstance(block, dict):
            omitted += 1
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            result = block.get("content", "")
            if isinstance(result, str):
                parts.append("[tool_result] " + result)
            elif isinstance(result, list):
                parts.extend("[tool_result] " + b["text"] for b in result
                             if isinstance(b, dict) and b.get("type") == "text")
                omitted += sum(not isinstance(b, dict) or b.get("type") != "text" for b in result)
            else:
                omitted += 1
        else:
            # Includes thinking, images, attachments, tool input; never serialize wholesale.
            omitted += 1
    return "\n".join(parts), omitted


def collect(root: Path, session_id: str):
    root = root.resolve(strict=True)
    info = next((s for s in discover(root) if s.session_id == session_id), None)
    if info is None:
        raise HubError("SESSION_NOT_IN_APPROVED_ROOT", 403)
    if info.file_size is None or info.file_size > 10 * 1024 * 1024:
        raise HubError("SOURCE_SIZE_UNVERIFIED_OR_TOO_LARGE", 413)
    fingerprint, proved = verified_source(root, session_id)
    # This SDK reads the whole JSONL before pagination. Read once with an explicit size ceiling.
    native = get_session_messages(session_id, directory=str(root))
    messages, omitted = [], 0
    for position, item in enumerate(native):
        if item.uuid not in proved:
            raise HubError("SOURCE_OWNERSHIP_UNVERIFIED", 403)
        text, skipped = visible_text(item.message)
        omitted += skipped
        if not text.strip():
            continue
        text, redacted = redact(text)
        for chunk_no, (chunk, start, end) in enumerate(chunks(text)):
            messages.append({"id": item.uuid + "." + str(chunk_no), "position": position * 10000 + chunk_no,
                             "role": item.type, "text": chunk, "redacted": redacted,
                             "source_span": {"message": item.uuid, "start": start, "end": end}})
    if not messages:
        raise HubError("NO_VISIBLE_MESSAGES")
    after = next((s for s in discover(root) if s.session_id == session_id), None)
    if (after is None or (info.last_modified, info.file_size) != (after.last_modified, after.file_size)
            or verified_source(root, session_id)[0] != fingerprint):
        raise HubError("SOURCE_CHANGED_RETRY", 409)
    coverage = {"mode": "explicit_session_snapshot", "sdk": version("claude-agent-sdk"),
                "native_messages": len(native), "stored_chunks": len(messages), "omitted_blocks": omitted,
                "complete_raw_log": False, "live": "unverified", "chunk_overlap_chars": 512,
                "source_ranges": "UTF-8 bytes in normalized, redacted visible message",
                "limitation": "SDK-visible chain only; hidden reasoning, tool inputs and nontext blocks omitted"}
    return {"root": str(root), "session_id": session_id,
            "source_version": f"claude-sdk/{version('claude-agent-sdk')}:adapter2:{fingerprint}",
            "messages": messages, "coverage": coverage}


def upload(client, project, snapshot):
    """Bounded idempotent batches; rerun the snapshot after interrupted upload."""
    batch, results, size = [], [], 0
    base = {k: v for k, v in snapshot.items() if k != "messages"}
    base["project"] = project
    # Incomplete until final batch, even if a later batch or process fails.
    base["coverage"] = {**base["coverage"], "upload_complete": False}
    for message in snapshot["messages"]:
        cost = len(json.dumps(message, ensure_ascii=False).encode())
        if batch and (len(batch) == 100 or size + cost > 800000):
            results.append(client.call("import", {**base, "messages": batch}))
            batch, size = [], 0
        batch.append(message)
        size += cost
    base["coverage"]["upload_complete"] = True
    results.append(client.call("import", {**base, "messages": batch}))
    return {"batches": len(results), "added": sum(r["added"] for r in results),
            "session_ref": results[-1]["session_ref"], "revision": results[-1]["revision"],
            "coverage": results[-1]["coverage"]}
