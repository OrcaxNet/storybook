from __future__ import annotations

import json
import os
import pty
import select
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from storybook import (
    config,
    embeddings,
    inference_cache,
    model_config,
    query_cache,
    store,
)
from storybook.profiles import PlatformRoots, ProfileRegistry
from storybook.setup_manager import SetupError, SetupManager


SENTINEL = "sb-secret-MUST-NOT-LEAK"


@contextmanager
def _provider_server(*, models=(), api_mode="ready"):
    state = {"models": set(models), "calls": [], "api_mode": api_mode}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        def _send(self, payload, *, lines=False):
            body = json.dumps(payload).encode() + (b"\n" if lines else b"")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            state["calls"].append(("GET", self.path, None))
            if self.path == "/api/tags":
                self._send({"models": [{"name": name} for name in state["models"]]})
                return
            self.send_error(404)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            state["calls"].append(("POST", self.path, self.headers.get("Authorization")))
            if self.path == "/api/pull":
                state["models"].add(payload["name"])
                self._send({"status": "success"}, lines=True)
            elif self.path == "/api/chat":
                self._send({"message": {"role": "assistant", "content": "OK"}})
            elif self.path == "/api/embeddings":
                self._send({"embedding": [0.1] * config.EMBED_DIM})
            elif self.path == "/v1/chat/completions":
                response = (
                    {"choices": [None]}
                    if state["api_mode"] == "generation_null"
                    else {"choices": [{"message": {"content": "OK"}}]}
                )
                self._send(response)
            elif self.path == "/v1/embeddings":
                response = (
                    {"data": [None]}
                    if state["api_mode"] == "embedding_null"
                    else {"data": [{"embedding": [0.1] * config.EMBED_DIM}]}
                )
                self._send(response)
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _subprocess_env(tmp_path: Path, storybook_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "STORYBOOK_HOME": str(storybook_home),
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "home" / ".codex"),
        "PATH": "",
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
    })
    return env


def _roots(tmp_path: Path) -> PlatformRoots:
    return PlatformRoots(
        config=tmp_path / "config", data=tmp_path / "data",
        cache=tmp_path / "cache", state=tmp_path / "state", logs=tmp_path / "logs",
    )


def _api_config() -> model_config.ModelConfig:
    return model_config.build(
        provider="api", base_url="https://models.example.test",
        llm_model="chat-v1", embedding_model="embed-1024",
        api_key_env="TEST_STORYBOOK_KEY",
    )


class Response:
    def __init__(self, payload=None, status=200):
        self.payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError("provider rejected request")
            error.response = self
            raise error

    def json(self):
        return self.payload


def test_profile_config_precedes_legacy_env_and_never_persists_secret(tmp_path):
    path = tmp_path / "profile" / "model-config.json"
    value = _api_config()
    model_config.save(path, value)
    resolved = model_config.resolve(path, {
        "TEST_STORYBOOK_KEY": SENTINEL,
        "OLLAMA_HOST": "http://legacy.invalid:11434",
    })

    raw = path.read_text(encoding="utf-8")
    public = json.dumps(resolved.public_dict({"TEST_STORYBOOK_KEY": SENTINEL}))
    assert resolved.source == "profile"
    assert resolved.generation.provider == "api"
    assert "TEST_STORYBOOK_KEY" in raw
    assert SENTINEL not in raw
    assert SENTINEL not in public


def test_legacy_environment_is_read_only_fallback(tmp_path):
    value = model_config.resolve(tmp_path / "missing.json", {
        "OLLAMA_HOST": "http://127.0.0.1:9999",
        "STORYBOOK_EMBED_MODEL": "legacy-embed",
        "DEEPSEEK_KEY": SENTINEL,
    })
    assert value.source == "legacy_env"
    assert value.embedding.base_url == "http://127.0.0.1:9999"
    assert value.embedding.model == "legacy-embed"
    assert not (tmp_path / "missing.json").exists()


def test_model_config_path_tracks_active_registry_without_writing(tmp_path):
    roots = _roots(tmp_path)
    registry = ProfileRegistry(roots.config / "profiles.json", roots=roots, environ={})
    old_registry = config.PROFILE_REGISTRY
    try:
        config.PROFILE_REGISTRY = registry
        config.refresh_profile(create=False)
        assert config.MODEL_CONFIG_PATH == config.PROFILE_PATHS.root / "model-config.json"
        assert not config.MODEL_CONFIG_PATH.exists()
    finally:
        config.PROFILE_REGISTRY = old_registry
        config.refresh_profile(create=False)


@pytest.mark.parametrize("url", [
    "https://user:pass@example.test", "https://example.test?token=secret",
    "https://example.test/#secret", "file:///tmp/model",
])
def test_unsafe_base_urls_are_rejected(url):
    with pytest.raises(model_config.ModelConfigError):
        model_config.build(
            provider="api", base_url=url, llm_model="chat",
            embedding_model="embed", api_key_env="API_KEY",
        )


def test_build_mixed_providers_persists_independent_endpoints(tmp_path):
    value = model_config.build(
        llm_protocol="openai", llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-v4-flash", llm_credential_env="STORYBOOK_API_KEY",
        embedding_protocol="ollama", embedding_base_url="http://localhost:11434",
        embedding_model="bge-m3",
    )
    assert value.generation.provider == "api"
    assert value.generation.protocol == "openai"
    assert value.embedding.provider == "ollama"
    assert value.embedding.protocol == "ollama"
    assert value.embedding.credential_env is None
    assert value.generation.base_url == "https://api.deepseek.com"
    assert value.embedding.base_url == "http://localhost:11434"

    path = tmp_path / "model-config.json"
    model_config.save(path, value)
    raw = path.read_text(encoding="utf-8")
    assert "STORYBOOK_API_KEY" in raw
    assert '"protocol": "openai"' in raw
    assert '"protocol": "ollama"' in raw
    loaded = model_config.load(path)
    assert loaded.generation.provider == "api"
    assert loaded.generation.protocol == "openai"
    assert loaded.embedding.provider == "ollama"
    assert loaded.embedding.protocol == "ollama"
    assert loaded.generation.credential_env == "STORYBOOK_API_KEY"


def test_empty_or_dash_secret_on_ollama_endpoints_never_invalid():
    value = model_config.build(
        llm_protocol="ollama", llm_base_url="http://localhost:11434",
        llm_model="chat", llm_credential_env="-",
        embedding_protocol="ollama", embedding_base_url="http://localhost:11434",
        embedding_model="embed", embedding_credential_env="",
    )
    assert value.generation.credential_env is None
    assert value.embedding.credential_env is None


def test_anthropic_generation_protocol_is_persisted():
    value = model_config.build(
        llm_protocol="anthropic", llm_base_url="https://api.deepseek.com/anthropic",
        llm_model="deepseek-v4-flash", llm_credential_env="ANTHROPIC_AUTH_TOKEN",
        embedding_protocol="ollama", embedding_base_url="http://localhost:11434",
        embedding_model="bge-m3",
    )
    assert value.generation.provider == "anthropic"
    assert value.generation.protocol == "anthropic"
    assert value.generation.credential_env == "ANTHROPIC_AUTH_TOKEN"


def test_v1_schema_without_protocol_loads_with_inferred_protocol(tmp_path):
    path = tmp_path / "model-config.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "generation": {
            "provider": "api", "base_url": "https://g.test",
            "model": "chat", "credential_env": "K",
        },
        "embedding": {
            "provider": "ollama", "base_url": "http://127.0.0.1:11434",
            "model": "embed",
        },
    }), encoding="utf-8")
    value = model_config.load(path)
    assert value.generation.provider == "api"
    assert value.generation.protocol == "openai"
    assert value.embedding.provider == "ollama"
    assert value.embedding.protocol == "ollama"


def test_invalid_secret_env_name_error_names_the_field():
    with pytest.raises(model_config.ModelConfigError) as caught:
        model_config.build(
            llm_protocol="openai", llm_base_url="https://g.test", llm_model="m",
            llm_credential_env="-",
        )
    assert "LLM" in str(caught.value)
    assert "credential env-name" in str(caught.value)

    with pytest.raises(model_config.ModelConfigError) as caught:
        model_config.build(
            llm_protocol="openai", llm_base_url="https://g.test", llm_model="m",
            llm_credential_env="K",
            embedding_protocol="openai", embedding_base_url="https://e.test",
            embedding_model="em", embedding_credential_env="-",
        )
    assert "Embedding" in str(caught.value)


def test_protocol_must_agree_with_provider():
    with pytest.raises(model_config.ModelConfigError):
        model_config.endpoint(
            "api", "https://g.test", "m", "K", protocol="anthropic"
        )


def test_api_provider_happy_path_checks_both_contracts(tmp_path, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/chat/completions"):
            return Response({"choices": [{"message": {"content": "OK"}}]})
        return Response({"data": [{"embedding": [0.1] * config.EMBED_DIM}]})

    monkeypatch.setattr("storybook.setup_manager.requests.post", post)
    manager = SetupManager(environ={"TEST_STORYBOOK_KEY": SENTINEL}, adapters=(), roots=_roots(tmp_path))
    result = manager._probe_provider(_api_config())
    serialized = json.dumps(result)
    assert [item[0].rsplit("/", 1)[-1] for item in calls] == ["completions", "embeddings"]
    assert len(result) == 2
    assert SENTINEL not in serialized
    assert all(call[1]["headers"]["Authorization"] == f"Bearer {SENTINEL}" for call in calls)


def test_api_provider_uses_independent_endpoint_credentials(tmp_path, monkeypatch):
    value = model_config.ModelConfig(
        model_config.SCHEMA_VERSION,
        model_config.endpoint("api", "https://generation.example.test", "chat", "GEN_KEY"),
        model_config.endpoint("api", "https://embedding.example.test", "embed", "EMBED_KEY"),
    )
    authorizations = []

    def post(url, **kwargs):
        authorizations.append(kwargs["headers"]["Authorization"])
        if url.endswith("/chat/completions"):
            return Response({"choices": [{"message": {"content": "OK"}}]})
        return Response({"data": [{"embedding": [0.1] * config.EMBED_DIM}]})

    monkeypatch.setattr("storybook.setup_manager.requests.post", post)
    manager = SetupManager(
        environ={"GEN_KEY": "generation-secret", "EMBED_KEY": "embedding-secret"},
        adapters=(), roots=_roots(tmp_path),
    )
    manager._probe_provider(value)
    assert authorizations == ["Bearer generation-secret", "Bearer embedding-secret"]


def test_api_missing_embedding_credential_fails_before_network(tmp_path, monkeypatch):
    value = model_config.ModelConfig(
        model_config.SCHEMA_VERSION,
        model_config.endpoint("api", "https://generation.example.test", "chat", "GEN_KEY"),
        model_config.endpoint("api", "https://embedding.example.test", "embed", "EMBED_KEY"),
    )
    monkeypatch.setattr(
        "storybook.setup_manager.requests.post",
        lambda *a, **k: pytest.fail("network must not be called"),
    )
    manager = SetupManager(environ={"GEN_KEY": SENTINEL}, adapters=(), roots=_roots(tmp_path))
    with pytest.raises(SetupError) as caught:
        manager._probe_provider(value)
    assert caught.value.code == "SB_MODEL_CREDENTIALS_MISSING"
    assert "embedding" in str(caught.value)


def test_existing_index_provider_switch_fails_before_any_config_write(
    tmp_path, monkeypatch
):
    roots = _roots(tmp_path)
    registry = ProfileRegistry(roots.config / "profiles.json", roots=roots, environ={})
    old_registry = config.PROFILE_REGISTRY
    try:
        config.PROFILE_REGISTRY = registry
        config.refresh_profile(create=True)
        original_config = model_config.build(
            provider="ollama", base_url="http://127.0.0.1:11434",
            llm_model="local-chat", embedding_model="shared-model",
            api_key_env=None,
        )
        model_config.save(config.MODEL_CONFIG_PATH, original_config)
        original_bytes = config.MODEL_CONFIG_PATH.read_bytes()
        config.refresh_model_config(environ={})
        store.init_db()
        store.add_story("existing", "memory", [], [0.1] * config.EMBED_DIM)
        before = store.get_embedding_index_state()
        candidate = model_config.ModelConfig(
            model_config.SCHEMA_VERSION,
            model_config.endpoint(
                "api", "https://new.example.test", "chat", "GEN_KEY"
            ),
            model_config.endpoint(
                "api", "https://new.example.test", before["active_model"], "EMBED_KEY"
            ),
        )
        manager = SetupManager(
            environ={"GEN_KEY": SENTINEL, "EMBED_KEY": SENTINEL},
            adapters=(), roots=roots,
        )
        monkeypatch.setattr(
            manager,
            "_probe_provider",
            lambda value: pytest.fail("provider network probe must not run"),
        )

        with pytest.raises(SetupError) as dry_run_error:
            manager.plan(requested_agents=(), provider_config=candidate)
        assert dry_run_error.value.code == "SB_MODEL_INDEX_INCOMPATIBLE"

        with pytest.raises(SetupError) as caught:
            manager.execute(requested_agents=(), provider_config=candidate)

        assert caught.value.code == "SB_MODEL_INDEX_INCOMPATIBLE"
        assert "profile create provider-migration --switch" in caught.value.hint
        assert config.MODEL_CONFIG_PATH.read_bytes() == original_bytes
        assert not manager.state_path.exists()
        assert store.get_embedding_index_state() == before
        assert before["active_provider"] == "ollama"
        assert before["active_base_url"] == "http://127.0.0.1:11434"
    finally:
        config.PROFILE_REGISTRY = old_registry
        config.refresh_profile(create=False)
        config.refresh_model_config()


def test_embedding_inference_cache_isolated_by_provider_and_base_url(
    monkeypatch
):
    values = {}
    calls = []

    def cache_get(namespace, payload):
        return values.get((namespace, inference_cache.input_hash(payload)))

    def cache_set(namespace, payload, value):
        values[(namespace, inference_cache.input_hash(payload))] = value

    class ProviderResponse(Response):
        def json(self):
            if config.EMBED_PROVIDER == "api":
                return {"data": [{"embedding": [0.1] * config.EMBED_DIM}]}
            return {"embedding": [0.1] * config.EMBED_DIM}

    monkeypatch.setattr(inference_cache, "get", cache_get)
    monkeypatch.setattr(inference_cache, "set", cache_set)
    monkeypatch.setattr(
        embeddings.requests,
        "post",
        lambda url, **kwargs: calls.append(url) or ProviderResponse(),
    )
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(config, "EMBED_BASE_URL", "http://provider-a.test")
    assert embeddings.embed("same", model="shared-model") is not None

    monkeypatch.setattr(config, "EMBED_PROVIDER", "api")
    monkeypatch.setattr(config, "EMBED_BASE_URL", "https://provider-b.test")
    monkeypatch.setattr(config, "EMBED_API_KEY", SENTINEL)
    assert embeddings.embed("same", model="shared-model") is not None

    assert calls == [
        "http://provider-a.test/api/embeddings",
        "https://provider-b.test/v1/embeddings",
    ]


def test_query_cache_identity_isolated_by_active_provider_spec():
    first = query_cache.index_identity(7, embedding_spec={
        "active_provider": "ollama",
        "active_base_url": "http://provider-a.test",
        "active_model": "shared-model",
        "active_version": "v1",
    })
    second = query_cache.index_identity(7, embedding_spec={
        "active_provider": "api",
        "active_base_url": "https://provider-b.test",
        "active_model": "shared-model",
        "active_version": "v1",
    })
    assert first != second


@pytest.mark.parametrize("status", [401, 403])
def test_api_auth_failures_have_stable_code_and_no_secret(tmp_path, monkeypatch, status):
    monkeypatch.setattr(
        "storybook.setup_manager.requests.post", lambda *a, **k: Response(status=status)
    )
    manager = SetupManager(environ={"TEST_STORYBOOK_KEY": SENTINEL}, adapters=(), roots=_roots(tmp_path))
    with pytest.raises(SetupError) as caught:
        manager._probe_provider(_api_config())
    assert caught.value.code == "SB_MODEL_AUTH_FAILED"
    assert SENTINEL not in str(caught.value)


def test_api_timeout_has_stable_code(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("contains no useful safe detail")

    monkeypatch.setattr("storybook.setup_manager.requests.post", timeout)
    manager = SetupManager(environ={"TEST_STORYBOOK_KEY": SENTINEL}, adapters=(), roots=_roots(tmp_path))
    with pytest.raises(SetupError) as caught:
        manager._probe_provider(_api_config())
    assert caught.value.code == "SB_MODEL_TIMEOUT"


def test_ollama_missing_service_reports_both_models(tmp_path, monkeypatch):
    value = model_config.build(
        provider="ollama", base_url="http://127.0.0.1:11434",
        llm_model="local-chat", embedding_model="local-embed", api_key_env=None,
    )
    monkeypatch.setattr(
        "storybook.setup_manager._ollama_tags",
        lambda: (_ for _ in ()).throw(requests.exceptions.ConnectionError("offline")),
    )
    manager = SetupManager(environ={}, adapters=(), roots=_roots(tmp_path))
    models, degraded = manager._ensure_models(
        download=True, progress=None, provider_config=value,
    )
    assert [item["name"] for item in models] == ["local-chat", "local-embed"]
    assert all(item["status"] == "unavailable" for item in models)
    assert len(degraded) == 1


def test_ollama_skip_download_reports_each_missing_model(tmp_path, monkeypatch):
    value = model_config.build(
        provider="ollama", base_url="http://127.0.0.1:11434",
        llm_model="local-chat", embedding_model="local-embed", api_key_env=None,
    )
    monkeypatch.setattr("storybook.setup_manager._ollama_tags", lambda: {})
    manager = SetupManager(environ={}, adapters=(), roots=_roots(tmp_path))
    models, degraded = manager._ensure_models(
        download=False, progress=None, provider_config=value,
    )
    assert [item["status"] for item in models] == ["skipped", "skipped"]
    assert degraded == ["model missing: local-chat", "model missing: local-embed"]


def test_ollama_downloads_missing_generation_and_embedding(tmp_path, monkeypatch):
    value = model_config.build(
        provider="ollama", base_url="http://127.0.0.1:11434",
        llm_model="local-chat", embedding_model="local-embed", api_key_env=None,
    )
    pulled = []
    monkeypatch.setattr("storybook.setup_manager._ollama_tags", lambda: {})
    monkeypatch.setattr("storybook.setup_manager._pull_model", lambda name, progress=None: pulled.append(name))
    manager = SetupManager(environ={}, adapters=(), roots=_roots(tmp_path))
    models, degraded = manager._ensure_models(
        download=True, progress=None, provider_config=value,
    )
    assert pulled == ["local-chat", "local-embed"]
    assert [item["status"] for item in models] == ["downloaded", "downloaded"]
    assert degraded == []


def test_ensure_models_mixed_remote_generation_and_ollama_embedding(tmp_path, monkeypatch):
    value = model_config.build(
        llm_protocol="openai", llm_base_url="https://models.example.test",
        llm_model="remote-chat", llm_credential_env="GEN_KEY",
        embedding_protocol="ollama", embedding_base_url="http://127.0.0.1:11434",
        embedding_model="local-embed",
    )
    pulled = []
    monkeypatch.setattr("storybook.setup_manager._ollama_tags", lambda: {})
    monkeypatch.setattr(
        "storybook.setup_manager._pull_model",
        lambda name, progress=None: pulled.append(name),
    )
    manager = SetupManager(environ={"GEN_KEY": SENTINEL}, adapters=(), roots=_roots(tmp_path))
    models, degraded = manager._ensure_models(
        download=True, progress=None, provider_config=value
    )
    # 只有 embedding 走本地 Ollama 管理；remote generation 只标记 remote，不拉取。
    assert pulled == ["local-embed"]
    statuses = {item["name"]: item["status"] for item in models}
    assert statuses["remote-chat"] == "remote"
    assert statuses["local-embed"] == "downloaded"
    assert degraded == []


def test_probe_provider_mixed_openai_generation_ollama_embedding(tmp_path, monkeypatch):
    value = model_config.build(
        llm_protocol="openai", llm_base_url="https://models.example.test",
        llm_model="chat-v1", llm_credential_env="GEN_KEY",
        embedding_protocol="ollama", embedding_base_url="http://127.0.0.1:11434",
        embedding_model="embed-1024",
    )
    urls = []

    def post(url, **kwargs):
        urls.append(url)
        if url.endswith("/chat/completions"):
            return Response({"choices": [{"message": {"content": "OK"}}]})
        return Response({"embedding": [0.1] * config.EMBED_DIM})

    monkeypatch.setattr("storybook.setup_manager.requests.post", post)
    manager = SetupManager(environ={"GEN_KEY": SENTINEL}, adapters=(), roots=_roots(tmp_path))
    result = manager._probe_provider(value)
    assert len(result) == 2
    # generation 走 OpenAI /v1/chat/completions；embedding 走 Ollama /api/embeddings。
    assert urls[0].endswith("/v1/chat/completions")
    assert urls[1].endswith("/api/embeddings")


def test_probe_provider_anthropic_generation_uses_messages_api(tmp_path, monkeypatch):
    value = model_config.build(
        llm_protocol="anthropic", llm_base_url="https://api.deepseek.com/anthropic",
        llm_model="deepseek-v4-flash", llm_credential_env="ANTHROPIC_AUTH_TOKEN",
        embedding_protocol="ollama", embedding_base_url="http://127.0.0.1:11434",
        embedding_model="embed-1024",
    )
    urls = []

    def post(url, **kwargs):
        urls.append(url)
        if url.endswith("/messages"):
            return Response({"content": [{"type": "text", "text": "OK"}]})
        return Response({"embedding": [0.1] * config.EMBED_DIM})

    monkeypatch.setattr("storybook.setup_manager.requests.post", post)
    manager = SetupManager(
        environ={"ANTHROPIC_AUTH_TOKEN": SENTINEL}, adapters=(), roots=_roots(tmp_path)
    )
    result = manager._probe_provider(value)
    assert len(result) == 2
    assert urls[0].endswith("/v1/messages")
    assert urls[1].endswith("/api/embeddings")


@pytest.mark.parametrize(
    ("responses", "code"),
    [
        ([Response({})], "SB_MODEL_GENERATION_FAILED"),
        ([Response({"choices": [None]})], "SB_MODEL_GENERATION_FAILED"),
        ([Response({"choices": [{"message": {"content": "OK"}}]}), Response({})], "SB_MODEL_EMBEDDING_FAILED"),
        ([Response({"choices": [{"message": {"content": "OK"}}]}), Response({"data": [None]})],
         "SB_MODEL_EMBEDDING_FAILED"),
        ([Response({"choices": [{"message": {"content": "OK"}}]}), Response({"data": [{"embedding": [0.1] * 8}]})],
         "SB_MODEL_EMBED_DIM_MISMATCH"),
    ],
)
def test_api_capability_failures_have_stable_codes(tmp_path, monkeypatch, responses, code):
    remaining = list(responses)
    monkeypatch.setattr("storybook.setup_manager.requests.post", lambda *a, **k: remaining.pop(0))
    manager = SetupManager(environ={"TEST_STORYBOOK_KEY": SENTINEL}, adapters=(), roots=_roots(tmp_path))
    with pytest.raises(SetupError) as caught:
        manager._probe_provider(_api_config())
    assert caught.value.code == code


def test_json_dry_run_is_single_json_zero_write_and_secret_free(tmp_path):
    storybook_home = tmp_path / "storybook"
    env = os.environ.copy()
    env.update({
        "STORYBOOK_HOME": str(storybook_home), "HOME": str(tmp_path / "home"),
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        "TEST_STORYBOOK_KEY": SENTINEL,
    })
    completed = subprocess.run(
        [sys.executable, "-m", "storybook.cli", "setup", "--provider", "api",
         "--base-url", "https://models.example.test", "--llm-model", "chat-v1",
         "--embedding-model", "embed-1024", "--api-key-env", "TEST_STORYBOOK_KEY",
         "--dry-run", "--json"],
        cwd=Path(__file__).parents[1], env=env, text=True, capture_output=True, check=False,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0, completed.stderr
    assert payload["writes_performed"] == 0
    assert payload["plan"]["model_config"]["generation"]["credential_status"] == "configured"
    assert SENTINEL not in completed.stdout + completed.stderr
    assert not storybook_home.exists()


@pytest.mark.skipif(os.name == "nt", reason="PTY contract is POSIX-only")
def test_tty_setup_prompts_dual_endpoints_with_protocol_and_inheritance(tmp_path):
    storybook_home = tmp_path / "storybook"
    env = _subprocess_env(tmp_path, storybook_home)
    env["TTY_API_KEY"] = SENTINEL
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-m", "storybook.cli", "setup", "--dry-run"],
        cwd=Path(__file__).parents[1], env=env,
        stdin=slave, stdout=slave, stderr=slave, close_fds=True,
    )
    os.close(slave)
    chunks = []

    def expect(text: str) -> None:
        deadline = time.monotonic() + 10
        while text not in b"".join(chunks).decode(errors="replace"):
            if time.monotonic() > deadline or process.poll() is not None:
                process.kill()
                pytest.fail(
                    f"interactive setup did not reach {text!r}: "
                    + b"".join(chunks).decode(errors="replace")
                )
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                chunks.append(os.read(master, 8192))

    try:
        for prompt, answer in (
            ("LLM protocol (openai, anthropic, ollama)", "openai"),
            ("LLM base URL（openai", "https://models.example.test"),
            ("LLM model", "chat-v1"),
            ("LLM secret env-name", "TTY_API_KEY"),
            ("Embedding protocol (openai, anthropic, ollama)", "ollama"),
            ("Embedding base URL（ollama", "http://127.0.0.1:11434"),
            ("Embedding model", "embed-1024"),
            # Embedding secret 直接回车留空：Ollama 本地端点视为无凭据（AC2）
            ("Embedding secret env-name", ""),
        ):
            expect(prompt)
            os.write(master, f"{answer}\n".encode())
        expect("Dry-run complete: no writes performed.")
        process.wait(timeout=2)
    finally:
        os.close(master)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
    output = b"".join(chunks).decode(errors="replace")
    assert process.returncode == 0, output
    assert "LLM protocol (openai, anthropic, ollama)" in output
    assert "LLM base URL（openai" in output
    assert "Embedding base URL（ollama" in output
    assert "LLM model" in output
    assert "Embedding model" in output
    assert "LLM secret env-name" in output
    assert "Embedding secret env-name" in output
    assert "Dry-run complete: no writes performed." in output
    assert "SB_MODEL_BASE_URL_REQUIRED" not in output
    assert SENTINEL not in output
    assert not storybook_home.exists()


@pytest.mark.parametrize(
    ("mode", "initial_models", "extra_args", "expected_status", "expected_pulls"),
    [
        ("happy", ("local-chat", "local-embed"), (), "ready", 0),
        ("missing_models", (), (), "ready", 2),
        ("skip_download", (), ("--skip-download",), "degraded", 0),
    ],
)
def test_fresh_home_ollama_setup_contract(
    tmp_path, mode, initial_models, extra_args, expected_status, expected_pulls
):
    storybook_home = tmp_path / mode
    with _provider_server(models=initial_models) as (base_url, state):
        completed = subprocess.run(
            [
                sys.executable, "-m", "storybook.cli", "setup",
                "--provider", "ollama", "--base-url", base_url,
                "--llm-model", "local-chat", "--embedding-model", "local-embed",
                "--yes", "--json", *extra_args,
            ],
            cwd=Path(__file__).parents[1],
            env=_subprocess_env(tmp_path, storybook_home),
            text=True, capture_output=True, check=False,
        )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == expected_status
    assert sum(path == "/api/pull" for _, path, _ in state["calls"]) == expected_pulls
    assert payload["model_config"]["generation"]["provider"] == "ollama"
    assert (storybook_home / "config" / "profiles.json").is_file()
    assert SENTINEL not in completed.stdout + completed.stderr


def test_fresh_home_ollama_missing_service_is_degraded(tmp_path):
    storybook_home = tmp_path / "missing-service"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        unused_port = probe.getsockname()[1]
    completed = subprocess.run(
        [
            sys.executable, "-m", "storybook.cli", "setup",
            "--provider", "ollama", "--base-url", f"http://127.0.0.1:{unused_port}",
            "--llm-model", "local-chat", "--embedding-model", "local-embed",
            "--yes", "--json",
        ],
        cwd=Path(__file__).parents[1],
        env=_subprocess_env(tmp_path, storybook_home),
        text=True, capture_output=True, check=False,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0, completed.stderr
    assert payload["status"] == "degraded"
    assert all(item["status"] == "unavailable" for item in payload["models"])
    assert (storybook_home / "config" / "profiles.json").is_file()


@pytest.mark.parametrize(
    ("api_mode", "code"),
    [
        ("generation_null", "SB_MODEL_GENERATION_FAILED"),
        ("embedding_null", "SB_MODEL_EMBEDDING_FAILED"),
    ],
)
def test_api_malformed_response_cli_json_is_stable_and_zero_write(
    tmp_path, api_mode, code
):
    storybook_home = tmp_path / api_mode
    with _provider_server(api_mode=api_mode) as (base_url, _):
        env = _subprocess_env(tmp_path, storybook_home)
        env["TEST_STORYBOOK_KEY"] = SENTINEL
        completed = subprocess.run(
            [
                sys.executable, "-m", "storybook.cli", "setup",
                "--provider", "api", "--base-url", base_url,
                "--llm-model", "chat-v1", "--embedding-model", "embed-1024",
                "--api-key-env", "TEST_STORYBOOK_KEY", "--yes", "--json",
            ],
            cwd=Path(__file__).parents[1], env=env,
            text=True, capture_output=True, check=False,
        )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["error"]["code"] == code
    assert SENTINEL not in completed.stdout + completed.stderr
    assert not storybook_home.exists()
