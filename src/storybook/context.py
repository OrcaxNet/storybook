"""Environment context capture, normalization, privacy and recall scoring.

``ContextEnvelope`` is deliberately independent from a repository path.  It
contains only stable IDs, user-facing aliases and privacy-safe hashes; raw
external session IDs, absolute paths, host names, remote hosts and repository
URLs never leave this module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


PROVENANCE_VALUES = frozenset({
    "unknown", "detected", "reported", "inferred", "user_confirmed",
})
TOOL_TYPES = frozenset({"claude_code", "cursor", "codex", "other"})
INTEGRATION_MODES = frozenset({"mcp", "hook", "log_import", "manual"})
RUNTIME_KINDS = frozenset({"local", "ssh", "devcontainer", "ci", "unknown"})

_LEAF_FIELDS = (
    "tool.type", "tool.version", "tool.integration_mode", "tool.installation_id",
    "device.id", "device.os_family", "device.os_version", "device.arch",
    "device.display_name", "session.id", "session.external_session_hash",
    "session.started_at", "session.locale", "workspace.id",
    "workspace.repo_fingerprint", "workspace.project_label",
    "workspace.cwd_alias", "workspace.branch", "runtime.kind",
    "runtime.remote_host_hash", "runtime.container_id_hash", "runtime.shell",
    "runtime.versions", "captured_at",
)
_FIELD_ALIASES = {
    "tool_type": "tool.type",
    "tool_version": "tool.version",
    "agent_installation_id": "tool.installation_id",
    "device_id": "device.id",
    "os_family": "device.os_family",
    "os_version": "device.os_version",
    "arch": "device.arch",
    "session_id": "session.id",
    "external_session_hash": "session.external_session_hash",
    "workspace_id": "workspace.id",
    "repo_fingerprint": "workspace.repo_fingerprint",
    "runtime_kind": "runtime.kind",
}
_SAFE_ALIAS_RE = re.compile(r"[^\w .@+-]+", re.UNICODE)


def utc_now() -> str:
    """Return an RFC3339 UTC timestamp without a local timezone dependency."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _uuid_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def _safe_alias(value: Any) -> str | None:
    """Return a short display alias, never an absolute or multi-segment path."""

    if value in (None, ""):
        return None
    text = str(value).strip().replace("\\", "/")
    if "/" in text:
        text = text.rstrip("/").rsplit("/", 1)[-1]
    text = _SAFE_ALIAS_RE.sub("-", text).strip(" .-")[:80]
    if not text or text == os.environ.get("USER"):
        return None
    return text


def _identity_path(name: str) -> Path:
    path = config.DB_PATH.parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_or_create_bytes(name: str, size: int) -> bytes:
    """Load a private local identity file, creating it race-safely if absent."""

    path = _identity_path(name)
    try:
        data = path.read_bytes()
        if len(data) == size:
            return data
    except FileNotFoundError:
        pass

    data = os.urandom(size)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = path.read_bytes()
        return existing if len(existing) == size else data
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    if os.name != "nt":
        path.chmod(0o600)
    return data


def _local_device_id() -> str:
    path = _identity_path(".device_id")
    try:
        existing = _uuid_or_none(path.read_text(encoding="ascii").strip())
        if existing:
            return existing
    except FileNotFoundError:
        pass

    value = str(uuid.uuid4())
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _uuid_or_none(path.read_text(encoding="ascii").strip())
        return existing or value
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(value)
    if os.name != "nt":
        path.chmod(0o600)
    return value


def local_hash(value: Any, namespace: str) -> str | None:
    """HMAC a local-only sensitive identifier with the Profile-local key."""

    if value in (None, ""):
        return None
    key = _load_or_create_bytes(".context_hmac_key", 32)
    digest = hmac.new(
        key,
        f"{namespace}\0{value}".encode("utf-8", errors="replace"),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{digest}"


def repository_fingerprint(repo_url: Any) -> str | None:
    """Hash a normalized repository URL without retaining credentials or URL."""

    if repo_url in (None, ""):
        return None
    raw = str(repo_url).strip().lower().rstrip("/")
    raw = re.sub(r"^[a-z][a-z0-9+.-]*://", "", raw)
    raw = raw.split("@", 1)[-1]
    if raw.endswith(".git"):
        raw = raw[:-4]
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def external_session_hash(external_session_id: Any) -> str | None:
    return local_hash(external_session_id, "external-session")


def workspace_path_hash(path: Any) -> str | None:
    if path in (None, ""):
        return None
    try:
        normalised = str(Path(str(path)).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        normalised = str(path)
    return local_hash(normalised, "workspace-path")


def _runtime_kind() -> str:
    if any(os.environ.get(k) for k in ("CI", "GITHUB_ACTIONS", "BUILDKITE", "GITLAB_CI")):
        return "ci"
    if any(os.environ.get(k) for k in ("REMOTE_CONTAINERS", "DEVCONTAINER", "CODESPACES")):
        return "devcontainer"
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        return "ssh"
    return "local"


def _source_tool(source: str | None) -> tuple[str, str]:
    value = (source or "").lower()
    if value.startswith("claude"):
        return "claude_code", "log_import"
    if value.startswith("cursor"):
        return "cursor", "log_import"
    if value.startswith("codex"):
        return "codex", "log_import"
    return "other", "manual"


def _empty_envelope(profile_id: str, captured_at: str) -> dict:
    envelope = {
        "profile_id": profile_id,
        "tool": {
            "type": None, "version": None, "integration_mode": None,
            "installation_id": None,
        },
        "device": {
            "id": None, "os_family": None, "os_version": None, "arch": None,
            "display_name": None,
        },
        "session": {
            "id": None, "external_session_hash": None, "started_at": None,
            "locale": None,
        },
        "workspace": {
            "id": None, "repo_fingerprint": None, "project_label": None,
            "cwd_alias": None, "branch": None,
        },
        "runtime": {
            "kind": None, "remote_host_hash": None, "container_id_hash": None,
            "shell": None, "versions": {},
        },
        "captured_at": captured_at,
        "provenance": {field: "unknown" for field in _LEAF_FIELDS},
    }
    envelope["provenance"]["captured_at"] = "detected"
    return envelope


def unknown_envelope(
    *,
    profile_id: str,
    session_id: str,
    source: str | None = None,
    captured_at: str | None = None,
) -> dict:
    """Build a non-fabricated envelope for legacy evidence captured pre-E5."""

    envelope = _empty_envelope(
        _uuid_or_none(profile_id) or config.PROFILE_ID,
        _normalise_timestamp(captured_at) or utc_now(),
    )
    tool_type, integration_mode = _source_tool(source)
    if tool_type != "other":
        envelope["tool"]["type"] = tool_type
        envelope["tool"]["integration_mode"] = integration_mode
        envelope["provenance"]["tool.type"] = "reported"
        envelope["provenance"]["tool.integration_mode"] = "reported"
    envelope["session"]["id"] = _uuid_or_none(session_id)
    envelope["provenance"]["session.id"] = "detected"
    envelope["runtime"]["kind"] = "unknown"
    envelope["provenance"]["runtime.kind"] = "unknown"
    return _finalise(envelope)


def capture_context(
    *,
    tool_type: str = "other",
    tool_version: str | None = None,
    integration_mode: str = "manual",
    external_session_id: str | None = None,
    session_id: str | None = None,
    workspace_path: str | Path | None = None,
    repo_url: str | None = None,
    project_label: str | None = None,
    branch: str | None = None,
    runtime_kind: str | None = None,
    remote_host: str | None = None,
    container_id: str | None = None,
    shell: str | None = None,
    versions: dict | None = None,
    started_at: str | None = None,
    locale: str | None = None,
    captured_at: str | None = None,
    provenance: str = "reported",
) -> dict:
    """Capture a privacy-safe adapter context with field-level provenance."""

    captured = _normalise_timestamp(captured_at) or utc_now()
    envelope = _empty_envelope(config.PROFILE_ID, captured)
    prov = provenance if provenance in PROVENANCE_VALUES else "reported"

    tool_type = tool_type if tool_type in TOOL_TYPES else "other"
    integration_mode = (
        integration_mode if integration_mode in INTEGRATION_MODES else "manual"
    )
    device_id = _local_device_id()
    try:
        install_id = str(uuid.uuid5(
            uuid.UUID(device_id), f"{tool_type}:{integration_mode}",
        ))
    except ValueError:
        install_id = str(uuid.uuid4())

    os_family = platform.system().lower() or None
    os_version = platform.release() or None
    arch = platform.machine().lower() or None
    display_name = " ".join(v for v in (os_family, arch) if v) or None

    repo_hash = repository_fingerprint(repo_url)
    path_hash = workspace_path_hash(workspace_path)
    workspace_fingerprint = repo_hash or path_hash
    workspace_id = None
    if workspace_fingerprint:
        workspace_id = str(uuid.uuid5(
            uuid.UUID(config.PROFILE_ID), workspace_fingerprint,
        ))
    alias = _safe_alias(project_label) or _safe_alias(workspace_path)

    detected_runtime = runtime_kind is None
    kind = runtime_kind or _runtime_kind()
    if kind not in RUNTIME_KINDS:
        kind = "unknown"
    if remote_host is None and kind == "ssh":
        ssh = os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT") or ""
        remote_host = ssh.split()[0] if ssh else None
    if container_id is None and kind == "devcontainer":
        container_id = os.environ.get("HOSTNAME")
    if shell is None:
        shell = _safe_alias(os.environ.get("SHELL"))

    envelope.update({
        "tool": {
            "type": tool_type,
            "version": str(tool_version) if tool_version else None,
            "integration_mode": integration_mode,
            "installation_id": install_id,
        },
        "device": {
            "id": device_id,
            "os_family": os_family,
            "os_version": os_version,
            "arch": arch,
            "display_name": display_name,
        },
        "session": {
            "id": _uuid_or_none(session_id),
            "external_session_hash": external_session_hash(external_session_id),
            "started_at": _normalise_timestamp(started_at),
            "locale": _safe_alias(locale),
        },
        "workspace": {
            "id": workspace_id,
            "repo_fingerprint": workspace_fingerprint,
            "project_label": alias,
            "cwd_alias": alias,
            "branch": _safe_alias(branch),
        },
        "runtime": {
            "kind": kind,
            "remote_host_hash": local_hash(remote_host, "remote-host"),
            "container_id_hash": local_hash(container_id, "container-id"),
            "shell": _safe_alias(shell),
            "versions": {
                str(k)[:40]: str(v)[:80]
                for k, v in (versions or {}).items() if v not in (None, "")
            },
        },
    })

    reported_fields = (
        "tool.type", "tool.version", "tool.integration_mode",
        "session.external_session_hash", "session.started_at", "session.locale",
        "workspace.project_label", "workspace.branch",
    )
    detected_fields = (
        "tool.installation_id", "device.id", "device.os_family",
        "device.os_version", "device.arch", "runtime.kind",
        "runtime.remote_host_hash", "runtime.container_id_hash", "runtime.shell",
        "runtime.versions", "workspace.id", "workspace.repo_fingerprint",
    )
    for field in reported_fields:
        if _get(envelope, field) not in (None, "", {}):
            envelope["provenance"][field] = prov
    for field in detected_fields:
        if _get(envelope, field) not in (None, "", {}):
            envelope["provenance"][field] = (
                "detected" if field != "runtime.kind" or detected_runtime else prov
            )
    if alias:
        envelope["provenance"]["workspace.cwd_alias"] = (
            prov if project_label else "inferred"
        )
    envelope["provenance"]["device.display_name"] = (
        "inferred" if display_name else "unknown"
    )
    return _finalise(envelope)


def normalize_envelope(
    value: dict | str | None,
    *,
    profile_id: str | None = None,
    session_id: str | None = None,
    source: str | None = None,
    captured_at: str | None = None,
) -> dict:
    """Normalize nested or PRD-style flattened input to a canonical envelope."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    value = value if isinstance(value, dict) else {}
    source_tool, source_mode = _source_tool(source)
    base = capture_context(
        tool_type=source_tool,
        integration_mode=source_mode,
        captured_at=captured_at or value.get("captured_at"),
    )
    base["profile_id"] = _uuid_or_none(profile_id or value.get("profile_id")) or config.PROFILE_ID

    supplied_provenance = value.get("provenance")
    supplied_provenance = supplied_provenance if isinstance(supplied_provenance, dict) else {}

    for group in ("tool", "device", "session", "workspace", "runtime"):
        incoming = value.get(group)
        if not isinstance(incoming, dict):
            continue
        for key in tuple(base[group]):
            if key not in incoming:
                continue
            field = f"{group}.{key}"
            raw = incoming[key]
            if field.endswith(".id") or field == "tool.installation_id":
                cleaned = _uuid_or_none(raw)
            elif field in ("session.started_at",):
                cleaned = _normalise_timestamp(raw)
            elif field == "runtime.kind":
                cleaned = raw if raw in RUNTIME_KINDS else "unknown"
            elif field == "tool.type":
                cleaned = raw if raw in TOOL_TYPES else "other"
            elif field == "tool.integration_mode":
                cleaned = raw if raw in INTEGRATION_MODES else "manual"
            elif field in ("workspace.project_label", "workspace.cwd_alias", "workspace.branch",
                           "device.display_name", "session.locale", "runtime.shell"):
                cleaned = _safe_alias(raw)
            elif field == "runtime.versions":
                cleaned = raw if isinstance(raw, dict) else {}
            else:
                cleaned = str(raw) if raw not in (None, "") else None
            base[group][key] = cleaned
            base["provenance"][field] = _provenance_for(
                supplied_provenance, field, cleaned, default="reported",
            )

    # Flattened ContextEnvelope compatibility.
    flattened = {
        "device_id": ("device", "id"),
        "agent_installation_id": ("tool", "installation_id"),
        "workspace_id": ("workspace", "id"),
        "session_id": ("session", "id"),
        "external_session_hash": ("session", "external_session_hash"),
    }
    for key, (group, field_name) in flattened.items():
        if key not in value:
            continue
        cleaned = _uuid_or_none(value[key]) if key.endswith("_id") else str(value[key])
        base[group][field_name] = cleaned
        dotted = f"{group}.{field_name}"
        base["provenance"][dotted] = _provenance_for(
            supplied_provenance, key, cleaned, default="reported",
        )

    # Raw sensitive aliases accepted only as input and immediately transformed.
    session_in = value.get("session") if isinstance(value.get("session"), dict) else {}
    workspace_in = value.get("workspace") if isinstance(value.get("workspace"), dict) else {}
    runtime_in = value.get("runtime") if isinstance(value.get("runtime"), dict) else {}
    if session_in.get("external_session_id"):
        base["session"]["external_session_hash"] = external_session_hash(
            session_in["external_session_id"]
        )
        base["provenance"]["session.external_session_hash"] = "reported"
    if workspace_in.get("repo_url"):
        base["workspace"]["repo_fingerprint"] = repository_fingerprint(
            workspace_in["repo_url"]
        )
    if workspace_in.get("path") or workspace_in.get("cwd"):
        raw_path = workspace_in.get("path") or workspace_in.get("cwd")
        base["workspace"]["repo_fingerprint"] = (
            base["workspace"]["repo_fingerprint"] or workspace_path_hash(raw_path)
        )
        alias = _safe_alias(raw_path)
        base["workspace"]["project_label"] = base["workspace"]["project_label"] or alias
        base["workspace"]["cwd_alias"] = base["workspace"]["cwd_alias"] or alias
    if runtime_in.get("remote_host"):
        base["runtime"]["remote_host_hash"] = local_hash(
            runtime_in["remote_host"], "remote-host"
        )
    if runtime_in.get("container_id"):
        base["runtime"]["container_id_hash"] = local_hash(
            runtime_in["container_id"], "container-id"
        )

    if session_id:
        base["session"]["id"] = _uuid_or_none(session_id)
        base["provenance"]["session.id"] = "detected"
    timestamp = _normalise_timestamp(value.get("captured_at") or captured_at)
    if timestamp:
        base["captured_at"] = timestamp
        base["provenance"]["captured_at"] = _provenance_for(
            supplied_provenance, "captured_at", timestamp, default="reported",
        )
    return _finalise(base)


def _provenance_for(mapping: dict, field: str, value: Any, *, default: str) -> str:
    if value in (None, "", {}):
        return "unknown"
    candidate = mapping.get(field, mapping.get(_FIELD_ALIASES.get(field, ""), default))
    return candidate if candidate in PROVENANCE_VALUES else default


def _finalise(envelope: dict) -> dict:
    """Enforce the null+unknown invariant for every canonical leaf."""

    for field in _LEAF_FIELDS:
        value = _get(envelope, field)
        if value in (None, "", {}):
            if field == "runtime.versions":
                envelope["runtime"]["versions"] = {}
            elif field != "captured_at":
                group, key = field.split(".", 1)
                envelope[group][key] = None
            envelope["provenance"][field] = "unknown"
        elif value == "unknown":
            envelope["provenance"][field] = "unknown"
        elif envelope["provenance"].get(field) not in PROVENANCE_VALUES:
            envelope["provenance"][field] = "reported"
    if not envelope.get("captured_at"):
        envelope["captured_at"] = utc_now()
        envelope["provenance"]["captured_at"] = "detected"
    return envelope


def normalize_applicability(value: dict | str | None) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    value = value if isinstance(value, dict) else {}
    return {
        "applies_when": value.get("applies_when", []),
        "excludes_when": value.get("excludes_when", []),
        "required_versions": value.get("required_versions", {}),
    }


def merge_environments(existing: list | str | None, additions: list[dict]) -> list[dict]:
    if isinstance(existing, str):
        try:
            existing = json.loads(existing)
        except json.JSONDecodeError:
            existing = []
    merged: list[dict] = []
    seen: set[str] = set()
    for item in list(existing or []) + list(additions or []):
        if not isinstance(item, dict):
            continue
        canonical = normalize_envelope(item, profile_id=item.get("profile_id"))
        session_key = canonical["session"].get("id")
        key = session_key or json.dumps(canonical, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(canonical)
    merged.sort(key=lambda item: item.get("captured_at") or "")
    return merged


def evaluate_story_context(
    current: dict | None,
    environments: list[dict] | None,
    applicability: dict | None,
) -> dict:
    """Return a bounded soft score and explainable differences for one Story."""

    if not current:
        return {
            "environment_score": 0.0,
            "warnings": [],
            "strict_excluded": False,
            "matched_environment": (environments or [None])[-1],
        }
    current = normalize_envelope(current, profile_id=current.get("profile_id"))
    environments = [
        normalize_envelope(item, profile_id=item.get("profile_id"))
        for item in (environments or []) if isinstance(item, dict)
    ]
    applicability = normalize_applicability(applicability)

    comparisons = [_compare_environment(current, item) for item in environments]
    if comparisons:
        best = max(comparisons, key=lambda item: item["environment_score"])
        strict_env_excluded = all(item["has_conflict"] for item in comparisons)
    else:
        best = {
            "environment_score": 0.0,
            "warnings": [],
            "has_conflict": False,
            "matched_environment": None,
        }
        strict_env_excluded = False

    app_score, app_warnings, app_excluded = _evaluate_applicability(current, applicability)
    score = max(-1.0, min(1.0, best["environment_score"] + app_score))
    warnings = list(dict.fromkeys(best["warnings"] + app_warnings))
    return {
        "environment_score": round(score, 4),
        "warnings": warnings,
        "strict_excluded": bool(app_excluded or strict_env_excluded),
        "matched_environment": best.get("matched_environment"),
    }


def _compare_environment(current: dict, story: dict) -> dict:
    fields = (
        ("workspace.id", "workspace", 0.35, 0.08),
        ("runtime.kind", "runtime", 0.25, 0.25),
        ("device.os_family", "OS", 0.15, 0.18),
        ("device.arch", "architecture", 0.10, 0.15),
        ("tool.type", "tool", 0.08, 0.03),
    )
    score = 0.0
    conflicts = []
    compared = False
    for field, label, match_weight, conflict_weight in fields:
        left = _get(current, field)
        right = _get(story, field)
        if left in (None, "", "unknown") or right in (None, "", "unknown"):
            continue
        compared = True
        if left == right:
            score += match_weight
        else:
            score -= conflict_weight
            conflicts.append(f"{label} differs: current={left}, story={right}")
    return {
        "environment_score": score if compared else 0.0,
        "warnings": conflicts,
        "has_conflict": bool(conflicts),
        "matched_environment": story,
    }


def _evaluate_applicability(current: dict, applicability: dict) -> tuple[float, list[str], bool]:
    applies = applicability.get("applies_when") or []
    excludes = applicability.get("excludes_when") or []
    score = 0.0
    warnings = []
    excluded = False

    evaluable_applies = [rule for rule in _as_rules(applies) if _rule_evaluable(rule)]
    if evaluable_applies:
        if any(_rule_matches(rule, current) for rule in evaluable_applies):
            score += 0.12
        else:
            score -= 0.2
            warnings.append("current context does not satisfy applies_when")
            excluded = True

    for rule in _as_rules(excludes):
        if _rule_evaluable(rule) and _rule_matches(rule, current):
            score -= 0.35
            warnings.append("current context matches excludes_when")
            excluded = True
            break
    return score, warnings, excluded


def _as_rules(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _rule_evaluable(rule: Any) -> bool:
    return isinstance(rule, dict)


def _rule_matches(rule: dict, current: dict) -> bool:
    if "field" in rule:
        field = _FIELD_ALIASES.get(str(rule["field"]), str(rule["field"]))
        actual = _get(current, field)
        expected = rule.get("in", rule.get("equals", rule.get("value")))
        values = expected if isinstance(expected, list) else [expected]
        return actual not in (None, "", "unknown") and actual in values
    matched_any = False
    for raw_field, expected in rule.items():
        field = _FIELD_ALIASES.get(str(raw_field), str(raw_field))
        actual = _get(current, field)
        values = expected if isinstance(expected, list) else [expected]
        if actual in (None, "", "unknown") or actual not in values:
            return False
        matched_any = True
    return matched_any


def environment_label(envelope: dict | None) -> str:
    if not envelope:
        return "unknown"
    bits = [
        _get(envelope, "tool.type"),
        _get(envelope, "workspace.project_label"),
        _get(envelope, "runtime.kind"),
        _get(envelope, "device.os_family"),
        _get(envelope, "device.arch"),
    ]
    return " · ".join(str(value) for value in bits if value not in (None, "", "unknown")) or "unknown"


def applicability_labels(applicability: dict | None) -> tuple[list[str], list[str]]:
    applicability = normalize_applicability(applicability)
    applies = [
        json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
        for item in _as_rules(applicability["applies_when"])
    ]
    excludes = [
        json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
        for item in _as_rules(applicability["excludes_when"])
    ]
    return applies, excludes


def _get(value: dict, dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
