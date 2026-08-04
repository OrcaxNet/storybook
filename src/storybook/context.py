"""Environment context capture, normalization, privacy and recall scoring.

``ContextEnvelope`` is deliberately independent from a repository path.  It
contains only stable IDs, user-facing aliases and privacy-safe hashes; raw
external session IDs, absolute paths, host names, remote hosts and repository
URLs never leave this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import json
import os
import platform
import re
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from . import config
from .identifiers import new_uuid7

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl.
    fcntl = None


PROVENANCE_VALUES = frozenset({
    "unknown", "detected", "reported", "inferred", "user_confirmed",
})
TOOL_TYPES = frozenset({
    "claude_code", "cursor", "codex", "gemini_cli", "cline", "other"
})
INTEGRATION_MODES = frozenset({"mcp", "hook", "log_import", "manual"})
RUNTIME_KINDS = frozenset({"local", "ssh", "devcontainer", "ci", "unknown"})

_LEAF_FIELDS = (
    "tool.type", "tool.version", "tool.adapter", "tool.adapter_version",
    "tool.integration_mode", "tool.installation_id",
    "device.id", "device.os_family", "device.os_version", "device.arch",
    "device.display_name", "session.id", "session.external_session_hash",
    "session.started_at", "session.locale", "workspace.id",
    "workspace.repo_fingerprint", "workspace.path_fingerprint",
    "workspace.project_label",
    "workspace.cwd_alias", "workspace.branch", "runtime.kind",
    "runtime.remote_host_hash", "runtime.container_id_hash", "runtime.shell",
    "runtime.versions", "captured_at",
)
_FIELD_ALIASES = {
    "tool_type": "tool.type",
    "tool_version": "tool.version",
    "tool_adapter": "tool.adapter",
    "tool_adapter_version": "tool.adapter_version",
    "agent_installation_id": "tool.installation_id",
    "device_id": "device.id",
    "os_family": "device.os_family",
    "os_version": "device.os_version",
    "arch": "device.arch",
    "session_id": "session.id",
    "external_session_hash": "session.external_session_hash",
    "workspace_id": "workspace.id",
    "repo_fingerprint": "workspace.repo_fingerprint",
    "path_fingerprint": "workspace.path_fingerprint",
    "runtime_kind": "runtime.kind",
}
_SAFE_ALIAS_RE = re.compile(r"[^\w .@+-]+", re.UNICODE)
_CANONICAL_HASH_RE = re.compile(
    r"^(?P<algorithm>hmac-sha256|sha256):(?P<digest>[0-9a-f]{64})$",
    re.IGNORECASE,
)
_IDENTITY_THREAD_LOCK = threading.Lock()


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


@contextmanager
def _locked_identity_file(name: str):
    """Open one private identity file under an inter-thread/process lock."""

    path = _identity_path(name)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    with _IDENTITY_THREAD_LOCK:
        fd = os.open(path, flags, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            if os.name != "nt":
                os.fchmod(fd, 0o600)
            yield fd
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _read_identity(fd: int, expected_size: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    return os.read(fd, expected_size + 1)


def _write_identity(fd: int, data: bytes) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:  # pragma: no cover - defensive OS failure guard.
            raise OSError("failed to persist local identity")
        remaining = remaining[written:]
    os.fsync(fd)


def _load_or_create_bytes(name: str, size: int) -> bytes:
    """Load or durably repair a fixed-size private local identity."""

    with _locked_identity_file(name) as fd:
        data = _read_identity(fd, size)
        if len(data) == size:
            return data
        data = os.urandom(size)
        _write_identity(fd, data)
        return data


def _local_device_id() -> str:
    with _locked_identity_file(".device_id") as fd:
        raw = _read_identity(fd, 36)
        try:
            existing = _uuid_or_none(raw.decode("ascii").strip())
        except UnicodeDecodeError:
            existing = None
        if existing:
            canonical = existing.encode("ascii")
            if raw != canonical:
                _write_identity(fd, canonical)
            return existing
        value = new_uuid7()
        _write_identity(fd, value.encode("ascii"))
        return value


def local_device_id() -> str:
    """Return the stable, privacy-safe identifier for this local device."""

    return _local_device_id()


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


def _canonical_digest(value: Any, algorithms: set[str]) -> str | None:
    """Return a normalized supported digest, or ``None`` for unsafe input."""

    if not isinstance(value, str):
        return None
    match = _CANONICAL_HASH_RE.fullmatch(value.strip())
    if not match or match.group("algorithm").lower() not in algorithms:
        return None
    return f"{match.group('algorithm').lower()}:{match.group('digest').lower()}"


def _canonical_local_hash(value: Any, namespace: str) -> str | None:
    if value in (None, ""):
        return None
    existing = _canonical_digest(value, {"hmac-sha256"})
    return existing or local_hash(value, namespace)


def repository_fingerprint(repo_url: Any) -> str | None:
    """Hash a normalized repository URL without retaining credentials or URL."""

    if repo_url in (None, ""):
        return None
    existing = _canonical_digest(repo_url, {"sha256"})
    if existing:
        return existing

    raw = str(repo_url).strip()
    if "://" in raw:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        try:
            port = parsed.port
        except ValueError:
            port = None
        default_port = {
            "http": 80,
            "https": 443,
            "ssh": 22,
            "git+ssh": 22,
        }.get(parsed.scheme.lower())
        if port and port != default_port:
            host = f"{host}:{port}"
        identity = host + unquote(parsed.path)
    else:
        without_suffix = raw.split("#", 1)[0].split("?", 1)[0]
        scp_style = re.fullmatch(
            r"(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)",
            without_suffix,
        )
        if scp_style:
            identity = f"{scp_style.group('host')}/{scp_style.group('path')}"
        else:
            identity = without_suffix.split("@", 1)[-1]

    identity = unquote(identity).replace("\\", "/").strip().lower()
    identity = re.sub(r"/+", "/", identity).rstrip("/")
    if identity.endswith(".git"):
        identity = identity[:-4]
    return "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _canonical_workspace_fingerprint(value: Any) -> str | None:
    if value in (None, ""):
        return None
    existing = _canonical_digest(value, {"sha256", "hmac-sha256"})
    return existing or repository_fingerprint(value)


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


def _git_workspace(path: Any) -> tuple[str | None, str | None]:
    """Resolve a cwd to its repository root and optional remote without leaking it."""

    if path in (None, ""):
        return None, None
    try:
        candidate = Path(str(path)).expanduser().resolve(strict=False)
        root_result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if root_result.returncode != 0 or not root_result.stdout.strip():
            return None, None
        root = root_result.stdout.strip()
        remote_result = subprocess.run(
            ["git", "-C", root, "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        remote = remote_result.stdout.strip() if remote_result.returncode == 0 else None
        return root, remote or None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None


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
    if value.startswith("gemini"):
        return "gemini_cli", "log_import"
    if value.startswith("cline"):
        return "cline", "log_import"
    return "other", "manual"


def _empty_envelope(profile_id: str, captured_at: str) -> dict:
    envelope = {
        "profile_id": profile_id,
        "tool": {
            "type": None, "version": None, "adapter": None,
            "adapter_version": None, "integration_mode": None,
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
            "id": None, "repo_fingerprint": None, "path_fingerprint": None,
            "project_label": None, "cwd_alias": None, "branch": None,
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
    tool_adapter: str | None = None,
    tool_adapter_version: str | None = None,
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
    remote_host_supplied = remote_host is not None
    container_id_supplied = container_id is not None
    shell_supplied = shell is not None

    tool_type = tool_type if tool_type in TOOL_TYPES else "other"
    integration_mode = (
        integration_mode if integration_mode in INTEGRATION_MODES else "manual"
    )
    device_id = local_device_id()
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

    git_root, detected_remote = _git_workspace(workspace_path)
    canonical_workspace_path = git_root or workspace_path
    repo_hash = repository_fingerprint(repo_url or detected_remote)
    path_hash = workspace_path_hash(canonical_workspace_path)
    workspace_fingerprint = repo_hash or path_hash
    workspace_id = None
    if workspace_fingerprint:
        workspace_id = str(uuid.uuid5(
            uuid.UUID(config.PROFILE_ID), workspace_fingerprint,
        ))
    alias = _safe_alias(project_label) or _safe_alias(canonical_workspace_path)

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
            "adapter": str(tool_adapter) if tool_adapter else None,
            "adapter_version": (
                str(tool_adapter_version) if tool_adapter_version else None
            ),
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
            "path_fingerprint": path_hash,
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
        "tool.type", "tool.version", "tool.adapter", "tool.adapter_version",
        "tool.integration_mode",
        "session.external_session_hash", "session.started_at", "session.locale",
        "workspace.branch",
    )
    detected_fields = (
        "tool.installation_id", "device.id", "device.os_family",
        "device.os_version", "device.arch", "runtime.kind",
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
        alias_provenance = prov if project_label else "inferred"
        envelope["provenance"]["workspace.project_label"] = alias_provenance
        envelope["provenance"]["workspace.cwd_alias"] = alias_provenance
    if workspace_fingerprint:
        envelope["provenance"]["workspace.repo_fingerprint"] = prov
        envelope["provenance"]["workspace.id"] = "inferred"
    if path_hash:
        envelope["provenance"]["workspace.path_fingerprint"] = "detected"
    if envelope["runtime"]["remote_host_hash"]:
        envelope["provenance"]["runtime.remote_host_hash"] = (
            prov if remote_host_supplied else "detected"
        )
    if envelope["runtime"]["container_id_hash"]:
        envelope["provenance"]["runtime.container_id_hash"] = (
            prov if container_id_supplied else "detected"
        )
    if envelope["runtime"]["shell"]:
        envelope["provenance"]["runtime.shell"] = (
            prov if shell_supplied else "detected"
        )
    if envelope["runtime"]["versions"]:
        envelope["provenance"]["runtime.versions"] = prov
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

    explicit_context = value is not None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    value = value if isinstance(value, dict) else {}
    source_tool, source_mode = _source_tool(source)
    target_profile = (
        _uuid_or_none(profile_id or value.get("profile_id")) or config.PROFILE_ID
    )
    if explicit_context:
        # A supplied envelope is evidence, often imported from another machine.
        # Missing leaves must remain unknown instead of inheriting this process.
        base = _empty_envelope(
            target_profile,
            _normalise_timestamp(captured_at or value.get("captured_at")) or utc_now(),
        )
        if source_tool != "other":
            base["tool"]["type"] = source_tool
            base["tool"]["integration_mode"] = source_mode
            base["provenance"]["tool.type"] = "reported"
            base["provenance"]["tool.integration_mode"] = "reported"
    else:
        # No context means a new live Session, where local detection is desired.
        base = capture_context(
            tool_type=source_tool,
            integration_mode=source_mode,
            captured_at=captured_at,
        )
    base["profile_id"] = target_profile

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
            elif field == "session.external_session_hash":
                cleaned = _canonical_local_hash(raw, "external-session")
            elif field == "workspace.repo_fingerprint":
                cleaned = _canonical_workspace_fingerprint(raw)
            elif field == "workspace.path_fingerprint":
                cleaned = _canonical_local_hash(raw, "workspace-path")
            elif field == "runtime.remote_host_hash":
                cleaned = _canonical_local_hash(raw, "remote-host")
            elif field == "runtime.container_id_hash":
                cleaned = _canonical_local_hash(raw, "container-id")
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
        "repo_fingerprint": ("workspace", "repo_fingerprint"),
        "path_fingerprint": ("workspace", "path_fingerprint"),
        "remote_host_hash": ("runtime", "remote_host_hash"),
        "container_id_hash": ("runtime", "container_id_hash"),
    }
    for key, (group, field_name) in flattened.items():
        if key not in value:
            continue
        if key.endswith("_id"):
            cleaned = _uuid_or_none(value[key])
        elif key == "external_session_hash":
            cleaned = _canonical_local_hash(value[key], "external-session")
        elif key == "repo_fingerprint":
            cleaned = _canonical_workspace_fingerprint(value[key])
        elif key == "path_fingerprint":
            cleaned = _canonical_local_hash(value[key], "workspace-path")
        elif key == "remote_host_hash":
            cleaned = _canonical_local_hash(value[key], "remote-host")
        else:
            cleaned = _canonical_local_hash(value[key], "container-id")
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
        base["provenance"]["session.external_session_hash"] = _provenance_for(
            supplied_provenance,
            "session.external_session_id",
            base["session"]["external_session_hash"],
            default="reported",
        )
    if workspace_in.get("repo_url"):
        base["workspace"]["repo_fingerprint"] = repository_fingerprint(
            workspace_in["repo_url"]
        )
        base["provenance"]["workspace.repo_fingerprint"] = _provenance_for(
            supplied_provenance,
            "workspace.repo_url",
            base["workspace"]["repo_fingerprint"],
            default="reported",
        )
    if workspace_in.get("path") or workspace_in.get("cwd"):
        raw_path = workspace_in.get("path") or workspace_in.get("cwd")
        raw_path_field = (
            "workspace.path" if workspace_in.get("path") else "workspace.cwd"
        )
        git_root, detected_remote = _git_workspace(raw_path)
        canonical_workspace_path = git_root or raw_path
        path_fingerprint = workspace_path_hash(canonical_workspace_path)
        if path_fingerprint:
            base["workspace"]["path_fingerprint"] = path_fingerprint
            base["provenance"]["workspace.path_fingerprint"] = _provenance_for(
                supplied_provenance,
                raw_path_field,
                path_fingerprint,
                default="reported",
            )
        if detected_remote and not workspace_in.get("repo_url"):
            base["workspace"]["repo_fingerprint"] = repository_fingerprint(
                detected_remote
            )
            base["provenance"]["workspace.repo_fingerprint"] = _provenance_for(
                supplied_provenance,
                raw_path_field,
                base["workspace"]["repo_fingerprint"],
                default="reported",
            )
        elif not base["workspace"]["repo_fingerprint"]:
            base["workspace"]["repo_fingerprint"] = path_fingerprint
            base["provenance"]["workspace.repo_fingerprint"] = _provenance_for(
                supplied_provenance,
                raw_path_field,
                base["workspace"]["repo_fingerprint"],
                default="reported",
            )
        alias = _safe_alias(canonical_workspace_path)
        if not base["workspace"]["project_label"] and alias:
            base["workspace"]["project_label"] = alias
            base["provenance"]["workspace.project_label"] = "inferred"
        if not base["workspace"]["cwd_alias"] and alias:
            base["workspace"]["cwd_alias"] = alias
            base["provenance"]["workspace.cwd_alias"] = "inferred"
    if runtime_in.get("remote_host"):
        base["runtime"]["remote_host_hash"] = local_hash(
            runtime_in["remote_host"], "remote-host"
        )
        base["provenance"]["runtime.remote_host_hash"] = _provenance_for(
            supplied_provenance,
            "runtime.remote_host",
            base["runtime"]["remote_host_hash"],
            default="reported",
        )
    if runtime_in.get("container_id"):
        base["runtime"]["container_id_hash"] = local_hash(
            runtime_in["container_id"], "container-id"
        )
        base["provenance"]["runtime.container_id_hash"] = _provenance_for(
            supplied_provenance,
            "runtime.container_id",
            base["runtime"]["container_id_hash"],
            default="reported",
        )

    if base["workspace"]["repo_fingerprint"] and not base["workspace"]["id"]:
        base["workspace"]["id"] = str(uuid.uuid5(
            uuid.UUID(base["profile_id"]),
            base["workspace"]["repo_fingerprint"],
        ))
        base["provenance"]["workspace.id"] = "inferred"

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


def project_identity(envelope: dict | None) -> tuple[str, str] | None:
    """Return a privacy-safe project identity, preferring repository evidence."""

    if not isinstance(envelope, dict):
        return None
    workspace = envelope.get("workspace")
    if not isinstance(workspace, dict):
        return None
    fingerprint = workspace.get("repo_fingerprint")
    if fingerprint:
        return "repo", str(fingerprint)
    workspace_id = workspace.get("id")
    if workspace_id:
        return "workspace", str(workspace_id)
    return None


def project_identity_values(envelope: dict | None) -> frozenset[str]:
    """Return primary and compatibility-safe project identifiers.

    New contexts prefer a remote-derived repository fingerprint but retain the
    HMAC of the canonical local Git root.  The latter lets them match Stories
    created before remote detection was introduced without persisting a path.
    """

    if not isinstance(envelope, dict):
        return frozenset()
    workspace = envelope.get("workspace")
    if not isinstance(workspace, dict):
        return frozenset()
    return frozenset(
        str(value)
        for value in (
            workspace.get("repo_fingerprint"),
            workspace.get("path_fingerprint"),
            workspace.get("id"),
        )
        if value
    )


def project_selector(
    envelope: dict | None,
) -> tuple[str | None, frozenset[str]]:
    """Return strong remote identity plus all privacy-safe compatibility IDs."""

    identity = project_identity(envelope)
    strong_remote = (
        identity[1]
        if identity and identity[0] == "repo" and identity[1].startswith("sha256:")
        else None
    )
    return strong_remote, project_identity_values(envelope)


def project_selectors_match(
    current: tuple[str | None, frozenset[str]],
    candidate: tuple[str | None, frozenset[str]],
) -> bool:
    """Match projects while making explicit remote conflicts authoritative."""

    current_remote, current_values = current
    candidate_remote, candidate_values = candidate
    if current_remote and candidate_remote:
        return current_remote == candidate_remote
    return bool(current_values.intersection(candidate_values))


def story_matches_project(
    current_context: dict | None, environments: list[dict] | None
) -> bool:
    """Require at least one Story environment to match the current project."""

    current = project_selector(current_context)
    if not current[1]:
        return False
    return environment_matches_project_selector(current, environments)


def environment_matches_project_selector(
    selector: tuple[str | None, frozenset[str]] | None,
    environments: list[dict] | None,
) -> bool:
    if selector is None or not selector[1]:
        return False
    candidates = [project_selector(item) for item in (environments or [])]
    current_remote = selector[0]
    candidate_remotes = {
        remote for remote, _values in candidates if remote is not None
    }
    if current_remote and candidate_remotes:
        return current_remote in candidate_remotes
    return any(project_selectors_match(selector, item) for item in candidates)


def environment_matches_project_identities(
    identities: frozenset[str] | set[str] | tuple[str, ...] | None,
    environments: list[dict] | None,
) -> bool:
    """Match any primary or compatibility identity in a Story environment."""

    expected = frozenset(identities or ())
    if not expected:
        return False
    return any(
        bool(expected.intersection(project_identity_values(item)))
        for item in (environments or [])
    )


def environment_matches_project_identity(
    identity: tuple[str, str] | None, environments: list[dict] | None
) -> bool:
    if identity is None:
        return False
    return environment_matches_project_identities(
        frozenset((identity[1],)), environments
    )


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
        if field == "tool.type" and "other" in (left, right):
            # ``other`` means the caller could not identify its host agent; it
            # must not create a concrete cross-tool conflict.
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
