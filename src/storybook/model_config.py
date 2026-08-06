"""Versioned, profile-local model provider configuration.

The file deliberately stores only the *name* of a credential environment
variable.  Secret values are resolved at request time and never serialized.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = 1
PROVIDERS = frozenset({"ollama", "api", "anthropic"})
PROTOCOLS = frozenset({"ollama", "openai", "anthropic"})
_PROTOCOL_BY_PROVIDER = {
    "ollama": "ollama",
    "api": "openai",
    "anthropic": "anthropic",
}
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ModelConfigError(RuntimeError):
    """Invalid or unsafe model configuration."""


@dataclass(frozen=True)
class ModelEndpoint:
    provider: str
    base_url: str
    model: str
    credential_env: str | None = None
    protocol: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    generation: ModelEndpoint
    embedding: ModelEndpoint
    source: str = "profile"

    def persisted_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("source", None)
        return payload

    def public_dict(self, environ: Mapping[str, str] | None = None) -> dict:
        env = os.environ if environ is None else environ
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "generation": _public_endpoint(self.generation, env),
            "embedding": _public_endpoint(self.embedding, env),
        }


def _public_endpoint(endpoint: ModelEndpoint, environ: Mapping[str, str]) -> dict:
    return {
        "provider": endpoint.provider,
        "protocol": (
            endpoint.protocol
            if endpoint.protocol is not None
            else _PROTOCOL_BY_PROVIDER.get(endpoint.provider)
        ),
        "base_url": safe_url(endpoint.base_url),
        "model": endpoint.model,
        "credential_env": endpoint.credential_env,
        "credential_status": (
            "not_required"
            if endpoint.provider == "ollama"
            else "configured"
            if endpoint.credential_env and bool(environ.get(endpoint.credential_env))
            else "missing"
        ),
    }


def validate_url(value: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelConfigError("base URL 必须是有效的 http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ModelConfigError("base URL 不得包含凭据、query 或 fragment")
    return value


def safe_url(value: str) -> str:
    """Return a URL safe for diagnostics, even for a legacy unsafe value."""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<redacted-url>"


def endpoint(
    provider: str,
    base_url: str,
    model: str,
    credential_env: str | None,
    protocol: str | None = None,
    *,
    label: str = "",
) -> ModelEndpoint:
    """Build a validated endpoint with an explicit wire ``protocol``.

    ``protocol`` is one of ``ollama | openai | anthropic`` and must agree with
    ``provider`` (``ollama | api | anthropic``).  When omitted it is inferred
    from ``provider``, keeping v1 files (which carry no ``protocol``) readable
    without migration.  ``label`` prefixes error messages so field-level
    diagnostics can point at LLM vs Embedding.
    """

    prefix = f"{label} " if label else ""
    provider = (provider or "").strip().lower()
    if provider not in PROVIDERS:
        raise ModelConfigError(
            f"{prefix}provider 不支持: {provider!r}（可用 ollama / api / anthropic）"
        )
    if protocol is not None:
        protocol = str(protocol).strip().lower()
        if protocol not in PROTOCOLS:
            raise ModelConfigError(
                f"{prefix}protocol 不支持: {protocol!r}（可用 ollama / openai / anthropic）"
            )
        if protocol != _PROTOCOL_BY_PROVIDER[provider]:
            raise ModelConfigError(
                f"{prefix}protocol {protocol!r} 与 provider {provider!r} 不一致"
            )
    resolved_protocol = _PROTOCOL_BY_PROVIDER[provider]
    model = (model or "").strip()
    if not model:
        raise ModelConfigError(f"{prefix}model 不能为空")
    env_name = (credential_env or "").strip() or None
    if provider == "ollama":
        # 本地端点不持久化凭据：留空或任意输入都视为无凭据。
        env_name = None
    elif env_name is not None and not ENV_NAME_RE.fullmatch(env_name):
        raise ModelConfigError(
            f"{prefix}credential env-name {env_name!r} 不是合法环境变量名"
            f"（需匹配 {ENV_NAME_RE.pattern}）"
        )
    return ModelEndpoint(provider, validate_url(base_url), model, env_name, resolved_protocol)


def _provider_from_protocol(protocol: str | None) -> str | None:
    """Map a CLI wire protocol (openai/anthropic/ollama) to a config provider."""

    if protocol is None:
        return None
    protocol = str(protocol).strip().lower()
    for provider, mapped in _PROTOCOL_BY_PROVIDER.items():
        if mapped == protocol:
            return provider
    return None


def build(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    api_key_env: str | None = None,
    llm_protocol: str | None = None,
    llm_base_url: str | None = None,
    llm_credential_env: str | None = None,
    embedding_protocol: str | None = None,
    embedding_base_url: str | None = None,
    embedding_credential_env: str | None = None,
) -> ModelConfig:
    """Build a dual-endpoint config from independent generation/embedding args.

    ``provider`` / ``base_url`` / ``api_key_env`` remain the legacy shorthand
    applied to both endpoints; any ``llm_*`` / ``embedding_*`` override takes
    precedence.  Embedding base URL and credential default to the LLM values
    (inheritance), matching the interactive onboarding contract.
    """

    gen_provider = _provider_from_protocol(llm_protocol) or provider or "ollama"
    gen_url = llm_base_url or base_url or DEFAULT_OLLAMA_URL
    gen_env = (
        llm_credential_env
        if llm_credential_env is not None
        else api_key_env
    )
    emb_provider = (
        _provider_from_protocol(embedding_protocol)
        or provider
        or gen_provider
    )
    emb_url = embedding_base_url or llm_base_url or base_url or DEFAULT_OLLAMA_URL
    emb_env = (
        embedding_credential_env
        if embedding_credential_env is not None
        else (llm_credential_env or api_key_env)
    )
    return ModelConfig(
        SCHEMA_VERSION,
        endpoint(
            gen_provider, gen_url, llm_model or DEFAULT_LLM_MODEL, gen_env,
            protocol=llm_protocol, label="LLM",
        ),
        endpoint(
            emb_provider, emb_url, embedding_model or DEFAULT_EMBED_MODEL, emb_env,
            protocol=embedding_protocol, label="Embedding",
        ),
    )


def _decode_endpoint(raw: object, field: str) -> ModelEndpoint:
    if not isinstance(raw, dict):
        raise ModelConfigError(f"{field} 必须是 object")
    allowed = {"provider", "base_url", "model", "credential_env", "protocol"}
    if set(raw) - allowed:
        raise ModelConfigError(f"{field} 含未知字段")
    return endpoint(
        str(raw.get("provider", "")), str(raw.get("base_url", "")),
        str(raw.get("model", "")), raw.get("credential_env"),
        protocol=raw.get("protocol"), label=field,
    )


def load(path: Path) -> ModelConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelConfigError(f"无法读取 model config: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ModelConfigError("不支持的 model config schema_version")
    if set(raw) - {"schema_version", "generation", "embedding"}:
        raise ModelConfigError("model config 含未知字段")
    return ModelConfig(
        SCHEMA_VERSION,
        _decode_endpoint(raw.get("generation"), "generation"),
        _decode_endpoint(raw.get("embedding"), "embedding"),
    )


def legacy(environ: Mapping[str, str]) -> ModelConfig:
    """Read-only compatibility: old env vars win only without profile config."""
    ollama_url = environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
    embed_model = environ.get("STORYBOOK_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    api_url = environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    llm_model = environ.get(
        "STORYBOOK_LLM_MODEL",
        environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "deepseek-v4-flash"),
    )
    credential = (
        "ANTHROPIC_AUTH_TOKEN"
        if environ.get("ANTHROPIC_AUTH_TOKEN")
        else "DEEPSEEK_KEY"
    )
    # deepseek_anthropic remains runtime-only and is never accepted in new files.
    return ModelConfig(
        SCHEMA_VERSION,
        ModelEndpoint(
            "deepseek_anthropic", safe_url(api_url).rstrip("/"), llm_model,
            credential, protocol="anthropic",
        ),
        endpoint("ollama", ollama_url, embed_model, None),
        source="legacy_env",
    )


def resolve(path: Path, environ: Mapping[str, str] | None = None) -> ModelConfig:
    env = os.environ if environ is None else environ
    return load(path) if path.is_file() else legacy(env)


def save(path: Path, value: ModelConfig) -> None:
    """Atomically persist a secret-free config with private permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value.persisted_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            path.chmod(0o600)
    finally:
        tmp.unlink(missing_ok=True)
