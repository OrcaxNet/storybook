"""Profile-local model configuration: one protocol/baseUrl/secret/model tuple per role."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SCHEMA_VERSION = 2
PROTOCOLS = ("openai", "anthropic", "ollama")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"


class ModelConfigError(ValueError):
    """Invalid model configuration; messages must never contain secrets."""


@dataclass(frozen=True)
class ModelEndpoint:
    protocol: str
    base_url: str
    secret: str = field(repr=False)
    model: str

    @property
    def provider(self) -> str:
        """Internal storage/adapter identifier, derived solely from the protocol."""
        return "api" if self.protocol == "openai" else self.protocol

    @property
    def credential_ref(self) -> str:
        """Opaque identity for a serving index's credential snapshot, never an env var."""
        return "file_" + hashlib.sha256(self.secret.encode()).hexdigest() if self.secret else ""

    def public_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "base_url": safe_url(self.base_url),
            "secret": "********" if self.secret else "",
            "model": self.model,
            "credential_status": "configured" if self.secret else "not_required",
        }


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    generation: ModelEndpoint
    embedding: ModelEndpoint
    embedding_dimension: int | None = None
    source: str = "profile"

    def persisted_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("source")
        if self.embedding_dimension is None:
            payload.pop("embedding_dimension")
        return payload

    def public_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "generation": self.generation.public_dict(),
            "embedding": self.embedding.public_dict(),
            "embedding_dimension": self.embedding_dimension,
        }


def validate_url(value: str) -> str:
    value = value.strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        parsed.port  # Validate malformed ports before issuing any request.
    except ValueError:
        raise ModelConfigError("baseUrl 必须是有效的 http(s) URL") from None
    if not valid:
        raise ModelConfigError("baseUrl 必须是有效的 http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ModelConfigError("baseUrl 不得包含凭据、query 或 fragment")
    return value


def safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<redacted-url>"


def request_url(base_url: str, protocol: str, resource: str) -> str:
    root = base_url.rstrip("/")
    suffix = "api" if protocol == "ollama" else "v1"
    if not root.endswith(f"/{suffix}"):
        root = f"{root}/{suffix}"
    return f"{root}/{resource}"


def request_headers(protocol: str, secret: str) -> dict[str, str]:
    if protocol == "anthropic":
        headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
        if secret:
            headers["x-api-key"] = secret
        return headers
    return {"Authorization": f"Bearer {secret}"} if secret else {}


def endpoint(protocol: str, base_url: str, secret: str, model: str, *, label="") -> ModelEndpoint:
    for name, value in (("protocol", protocol), ("base_url", base_url), ("secret", secret), ("model", model)):
        if not isinstance(value, str):
            raise ModelConfigError(f"{label}.{name} 必须是字符串")
    protocol = protocol.strip().lower()
    if protocol not in PROTOCOLS:
        raise ModelConfigError(f"{label}.protocol 必须是 openai / anthropic / ollama")
    if not model.strip():
        raise ModelConfigError(f"{label}.model 不能为空")
    return ModelEndpoint(protocol, validate_url(base_url), "" if secret == "-" else secret.strip(), model.strip())


def build(
    *, llm_protocol=None, llm_base_url=None, llm_secret=None, llm_model=None,
    embedding_protocol=None, embedding_base_url=None, embedding_secret=None, embedding_model=None,
) -> ModelConfig:
    generation = endpoint(
        llm_protocol if llm_protocol is not None else "ollama",
        llm_base_url if llm_base_url is not None else DEFAULT_OLLAMA_URL,
        llm_secret if llm_secret is not None else "",
        llm_model if llm_model is not None else DEFAULT_LLM_MODEL, label="LLM",
    )
    embedding = endpoint(
        embedding_protocol if embedding_protocol is not None else generation.protocol,
        embedding_base_url if embedding_base_url is not None else generation.base_url,
        embedding_secret if embedding_secret is not None else generation.secret,
        embedding_model if embedding_model is not None else generation.model, label="Embedding",
    )
    if embedding.protocol == "anthropic":
        raise ModelConfigError("Embedding 不支持 anthropic 协议，请选择 openai 或 ollama")
    return ModelConfig(SCHEMA_VERSION, generation, embedding)


def defaults() -> ModelConfig:
    value = build(embedding_model=DEFAULT_EMBED_MODEL)
    return ModelConfig(SCHEMA_VERSION, value.generation, value.embedding, source="defaults")


def load(path: Path) -> ModelConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ModelConfigError("无法读取模型配置文件，请检查路径与 JSON 格式") from None
    if not isinstance(raw, dict) or set(raw) - {"schema_version", "generation", "embedding", "embedding_dimension"}:
        raise ModelConfigError("模型配置必须包含 generation 和 embedding 两组四元组")
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ModelConfigError(f"schema_version 必须是 {SCHEMA_VERSION}")
    values = {}
    for kind in ("generation", "embedding"):
        item = raw.get(kind)
        if not isinstance(item, dict) or set(item) - {"protocol", "base_url", "secret", "model"}:
            raise ModelConfigError(f"{kind} 仅接受 protocol、base_url、secret、model")
        inherited = values.get("generation", {})
        values[kind] = {**inherited, **item}
        if set(values[kind]) != {"protocol", "base_url", "secret", "model"}:
            raise ModelConfigError(f"{kind} 需完整填写 protocol、base_url、secret、model")
    generation = endpoint(**values["generation"], label="generation")
    embedding = endpoint(**values["embedding"], label="embedding")
    if embedding.protocol == "anthropic":
        raise ModelConfigError("Embedding 不支持 anthropic 协议，请选择 openai 或 ollama")
    dimension = raw.get("embedding_dimension")
    if dimension is not None and (type(dimension) is not int or dimension < 1):
        raise ModelConfigError("embedding_dimension 必须是正整数，或省略以自动检测")
    return ModelConfig(SCHEMA_VERSION, generation, embedding, dimension)


def resolve(path: Path) -> ModelConfig:
    return load(path) if path.is_file() else defaults()


def save(path: Path, value: ModelConfig) -> None:
    # Serving indexes retain their own credential snapshot when a target tuple
    # is edited for a rebuild. Users only edit model-config.json.
    secrets = {item.credential_ref: item.secret for item in (value.generation, value.embedding) if item.secret}
    if secrets:
        _save_json(path.with_name("model-secrets.json"), {**_read_secrets(path), **secrets})
    _save_json(path, value.persisted_dict())


def _read_secrets(config_path: Path) -> dict[str, str]:
    path = config_path.with_name("model-secrets.json")
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw.items()):
            raise ValueError
        return raw
    except (OSError, ValueError):
        raise ModelConfigError("无法读取索引凭据快照") from None


def credential_value(reference: str, *, path: Path) -> str | None:
    if not reference:
        return None
    if path.is_file():
        value = load(path)
        for item in (value.generation, value.embedding):
            if item.credential_ref == reference:
                return item.secret
    return _read_secrets(path).get(reference)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
