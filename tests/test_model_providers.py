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
def _provider_server(*, models=(), api_mode="ready", dimension=1024):
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
                if payload["model"] not in state["models"]:
                    self.send_error(404)
                    return
                self._send({"embedding": [0.1] * dimension})
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
                    else {"data": [{"embedding": [0.1] * dimension}]}
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
    return model_config.build(llm_model="chat-v1", embedding_model="embed-1024", llm_protocol='openai', llm_base_url="https://models.example.test", llm_secret=SENTINEL)


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
        model_config.build(llm_model="chat", embedding_model="embed", llm_protocol='openai', llm_base_url=url, llm_secret=SENTINEL)


def test_build_mixed_providers_persists_independent_endpoints(tmp_path):
    value = model_config.build(llm_protocol="openai", llm_base_url="https://api.deepseek.com", llm_model="deepseek-v4-flash", embedding_protocol="ollama", embedding_base_url="http://localhost:11434", embedding_model="bge-m3", embedding_secret="", llm_secret=SENTINEL)
    assert value.generation.provider == "api"
    assert value.generation.protocol == "openai"
    assert value.embedding.provider == "ollama"
    assert value.embedding.protocol == "ollama"
    assert value.embedding.credential_ref == ""
    assert value.generation.base_url == "https://api.deepseek.com"
    assert value.embedding.base_url == "http://localhost:11434"

    path = tmp_path / "model-config.json"
    model_config.save(path, value)
    raw = path.read_text(encoding="utf-8")
    assert SENTINEL in raw
    assert '"protocol": "openai"' in raw
    assert '"protocol": "ollama"' in raw
    loaded = model_config.load(path)
    assert loaded.generation.provider == "api"
    assert loaded.generation.protocol == "openai"
    assert loaded.embedding.provider == "ollama"
    assert loaded.embedding.protocol == "ollama"
    assert loaded.generation.secret == SENTINEL


def test_empty_or_dash_secret_on_ollama_endpoints_never_invalid():
    value = model_config.build(llm_protocol="ollama", llm_base_url="http://localhost:11434", llm_model="chat", embedding_protocol="ollama", embedding_base_url="http://localhost:11434", embedding_model="embed", llm_secret='', embedding_secret='')
    assert value.generation.credential_ref == ""
    assert value.embedding.credential_ref == ""


def test_anthropic_generation_protocol_is_persisted():
    value = model_config.build(llm_protocol="anthropic", llm_base_url="https://api.deepseek.com/anthropic", llm_model="deepseek-v4-flash", embedding_protocol="ollama", embedding_base_url="http://localhost:11434", embedding_model="bge-m3", embedding_secret="", llm_secret=SENTINEL)
    assert value.generation.provider == "anthropic"
    assert value.generation.protocol == "anthropic"
    assert value.generation.secret == SENTINEL


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
        model_config.endpoint('openai', "https://generation.example.test", 'generation-secret', "chat"),
        model_config.endpoint('openai', "https://embedding.example.test", 'embedding-secret', "embed"),
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


def test_existing_index_provider_switch_fails_before_any_config_write(
    tmp_path, monkeypatch
):
    roots = _roots(tmp_path)
    registry = ProfileRegistry(roots.config / "profiles.json", roots=roots, environ={})
    old_registry = config.PROFILE_REGISTRY
    try:
        config.PROFILE_REGISTRY = registry
        config.refresh_profile(create=True)
        original_config = model_config.build(llm_model="local-chat", embedding_model="shared-model", llm_protocol='ollama', llm_base_url="http://127.0.0.1:11434", llm_secret='')
        model_config.save(config.MODEL_CONFIG_PATH, original_config)
        original_bytes = config.MODEL_CONFIG_PATH.read_bytes()
        config.refresh_model_config()
        store.init_db()
        store.add_story("existing", "memory", [], [0.1] * config.EMBED_DIM)
        before = store.get_embedding_index_state()
        candidate = model_config.ModelConfig(
            model_config.SCHEMA_VERSION,
            model_config.endpoint('openai', "https://new.example.test", 'generation-secret', "chat"),
            model_config.endpoint('openai', "https://new.example.test", 'embedding-secret', before["active_model"]),
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
    value = model_config.build(llm_model="local-chat", embedding_model="local-embed", llm_protocol='ollama', llm_base_url="http://127.0.0.1:11434", llm_secret='')
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
    value = model_config.build(llm_model="local-chat", embedding_model="local-embed", llm_protocol='ollama', llm_base_url="http://127.0.0.1:11434", llm_secret='')
    monkeypatch.setattr("storybook.setup_manager._ollama_tags", lambda **kwargs: {})
    manager = SetupManager(environ={}, adapters=(), roots=_roots(tmp_path))
    models, degraded = manager._ensure_models(
        download=False, progress=None, provider_config=value,
    )
    assert [item["status"] for item in models] == ["skipped", "skipped"]
    assert degraded == ["model missing: local-chat", "model missing: local-embed"]


def test_ollama_downloads_missing_generation_and_embedding(tmp_path, monkeypatch):
    value = model_config.build(llm_model="local-chat", embedding_model="local-embed", llm_protocol='ollama', llm_base_url="http://127.0.0.1:11434", llm_secret='')
    pulled = []
    monkeypatch.setattr("storybook.setup_manager._ollama_tags", lambda **kwargs: {})
    monkeypatch.setattr("storybook.setup_manager._pull_model", lambda name, progress=None, **kwargs: pulled.append(name))
    manager = SetupManager(environ={}, adapters=(), roots=_roots(tmp_path))
    models, degraded = manager._ensure_models(
        download=True, progress=None, provider_config=value,
    )
    assert pulled == ["local-chat", "local-embed"]
    assert [item["status"] for item in models] == ["downloaded", "downloaded"]
    assert degraded == []


def test_ensure_models_mixed_remote_generation_and_ollama_embedding(tmp_path, monkeypatch):
    value = model_config.build(llm_protocol="openai", llm_base_url="https://models.example.test", llm_model="remote-chat", embedding_protocol="ollama", embedding_base_url="http://127.0.0.1:11434", embedding_model="local-embed", llm_secret='generation-secret')
    pulled = []
    monkeypatch.setattr("storybook.setup_manager._ollama_tags", lambda **kwargs: {})
    monkeypatch.setattr(
        "storybook.setup_manager._pull_model",
        lambda name, progress=None, **kwargs: pulled.append(name),
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
    value = model_config.build(llm_protocol="openai", llm_base_url="https://models.example.test", llm_model="chat-v1", embedding_protocol="ollama", embedding_base_url="http://127.0.0.1:11434", embedding_model="embed-1024", llm_secret='generation-secret')
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
    value = model_config.build(llm_protocol="anthropic", llm_base_url="https://api.deepseek.com/anthropic", llm_model="deepseek-v4-flash", embedding_protocol="ollama", embedding_base_url="http://127.0.0.1:11434", embedding_model="embed-1024", llm_secret=SENTINEL)
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
        [sys.executable, "-m", "storybook.cli", "setup", "--llm-protocol", "openai",
         "--llm-base-url", "https://models.example.test", "--llm-model", "chat-v1",
         "--embedding-model", "embed-1024", "--llm-secret", SENTINEL,
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
            ("LLM protocol", "openai"),
            ("LLM baseUrl", "https://models.example.test/v1"),
            ("LLM secret（", SENTINEL),
            ("LLM model-id", "chat-v1"),
            ("Embedding protocol", "ollama"),
            ("Embedding baseUrl", "http://127.0.0.1:11434"),
            ("Embedding secret（", "-"),
            ("Embedding model-id", "embed-1024"),
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
    assert "LLM protocol" in output
    assert "LLM baseUrl" in output
    assert "Embedding baseUrl" in output
    assert "LLM model" in output
    assert "Embedding model" in output
    assert "LLM secret（" in output
    assert "Embedding secret（" in output
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
                "--llm-protocol", "ollama", "--llm-base-url", base_url,
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
    assert payload["model_config"]["generation"]["protocol"] == "ollama"
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
            "--llm-protocol", "ollama", "--llm-base-url", f"http://127.0.0.1:{unused_port}",
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
                "--llm-protocol", "openai", "--llm-base-url", base_url,
                "--llm-model", "chat-v1", "--embedding-model", "embed-1024",
                "--llm-secret", SENTINEL, "--yes", "--json",
            ],
            cwd=Path(__file__).parents[1], env=env,
            text=True, capture_output=True, check=False,
        )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["error"]["code"] == code
    assert SENTINEL not in completed.stdout + completed.stderr
    assert not storybook_home.exists()


@pytest.mark.parametrize("clear", ["", "-"])
def test_tuple_defaults_and_explicit_secret_clear(clear):
    first = model_config.build(llm_protocol="openai", llm_base_url="https://models.test/v1", llm_secret=SENTINEL, llm_model="shared-model")
    assert first.embedding == first.generation
    cleared = model_config.build(llm_protocol="openai", llm_base_url="https://models.test/v1", llm_secret=SENTINEL, llm_model="chat", embedding_secret=clear)
    assert cleared.embedding.credential_ref == ""
    assert cleared.embedding.secret == ""
    assert cleared.embedding.model == "chat"
    assert SENTINEL not in repr(first)
    assert SENTINEL in json.dumps(first.persisted_dict())
    assert SENTINEL not in json.dumps(first.public_dict())


def test_pasted_secret_survives_reload_and_preserves_old_index_reference(tmp_path):
    path = tmp_path / "profile" / "model-config.json"
    first = model_config.build(llm_secret=SENTINEL, llm_model="shared")
    model_config.save(path, first)
    loaded = model_config.load(path)
    assert loaded == first
    assert model_config.credential_value(
        loaded.embedding.credential_ref, path=path
    ) == SENTINEL
    second = model_config.build(llm_secret="second-secret", llm_model="shared")
    model_config.save(path, second)
    assert model_config.credential_value(
        first.embedding.credential_ref, path=path
    ) == SENTINEL
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.with_name("model-secrets.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("protocol,resource", [
    ("openai", "chat/completions"), ("openai", "embeddings"),
    ("anthropic", "messages"), ("ollama", "embeddings"),
])
def test_versioned_base_url_is_appended_once(protocol, resource):
    suffix = "api" if protocol == "ollama" else "v1"
    for base in ("https://gateway.test/prefix", f"https://gateway.test/prefix/{suffix}/"):
        assert model_config.request_url(base, protocol, resource) == f"https://gateway.test/prefix/{suffix}/{resource}"


def test_anthropic_probe_uses_same_auth_as_runtime(tmp_path, monkeypatch):
    value = model_config.build(llm_protocol="anthropic", llm_base_url="https://gateway.test/v1", llm_secret=SENTINEL, llm_model="chat", embedding_protocol="ollama", embedding_base_url="http://localhost:11434", embedding_secret="", embedding_model="embed")
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs["headers"]))
        return Response({"content": [{"type": "text", "text": "OK"}]})
    monkeypatch.setattr("storybook.setup_manager.requests.post", post)
    SetupManager(environ={}, adapters=(), roots=_roots(tmp_path))._probe_provider(value, kinds=("generation",))
    assert calls == [("https://gateway.test/v1/messages", {
        "x-api-key": SENTINEL, "anthropic-version": "2023-06-01", "content-type": "application/json",
    })]


@pytest.mark.parametrize("embedding_protocol,dimension", [("ollama", 1024), ("ollama", 768), ("openai", 768)])
def test_fresh_mixed_tuple_setup_and_runtime_in_a_new_process(tmp_path, embedding_protocol, dimension):
    storybook_home = tmp_path / "storybook"
    env = _subprocess_env(tmp_path, storybook_home)
    with _provider_server() as (llm_url, llm_state), _provider_server(dimension=dimension) as (embed_url, embed_state):
        completed = subprocess.run(
            [sys.executable, "-m", "storybook.cli", "setup", "--yes", "--json",
             "--llm-protocol", "openai", "--llm-base-url", llm_url + "/v1",
             "--llm-secret", SENTINEL, "--llm-model", "chat-v1",
             "--embedding-protocol", embedding_protocol,
             "--embedding-base-url", embed_url + ("/v1" if embedding_protocol == "openai" else ""),
             "--embedding-secret", "", "--embedding-model", "local-embed"],
            env=env, text=True, capture_output=True, timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["status"] == "ready"
        assert SENTINEL not in completed.stdout + completed.stderr
        runtime = subprocess.run(
            [sys.executable, "-c", f"from storybook import llm, embeddings; assert llm._chat('hello') == 'OK'; assert len(embeddings.embed('hello')) == {dimension}"],
            env=env, text=True, capture_output=True, timeout=30,
        )
        assert runtime.returncode == 0, runtime.stderr
    assert all(auth == f"Bearer {SENTINEL}" for _, _, auth in llm_state["calls"])
    assert all(auth is None for _, _, auth in embed_state["calls"])
    embed_paths = [path for _, path, _ in embed_state["calls"]]
    if embedding_protocol == "ollama":
        assert embed_paths.index("/api/pull") < embed_paths.index("/api/embeddings")
    else:
        assert all(path == "/v1/embeddings" for path in embed_paths)
    model_path, = storybook_home.rglob("model-config.json")
    assert json.loads(model_path.read_text())["embedding_dimension"] == dimension
    assert SENTINEL in model_path.read_text()
    assert SENTINEL in model_path.with_name("model-secrets.json").read_text()


def test_ollama_model_management_uses_each_tuple_url(tmp_path):
    with _provider_server() as (first_url, first_state), _provider_server() as (second_url, second_state):
        value = model_config.build(llm_protocol="ollama", llm_base_url=first_url, llm_model="same-model", embedding_protocol="ollama", embedding_base_url=second_url, embedding_model="same-model")
        results, degraded = SetupManager(environ={}, adapters=(), roots=_roots(tmp_path))._ensure_models(
            download=True, progress=None, provider_config=value,
        )
    assert not degraded
    assert len(results) == 2
    assert "/api/pull" in [path for _, path, _ in first_state["calls"]]
    assert "/api/pull" in [path for _, path, _ in second_state["calls"]]


def test_file_inheritance_uses_empty_secret_as_explicit_override(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "generation": {"protocol": "openai", "base_url": "https://models.test/v1", "secret": SENTINEL, "model": "chat"},
        "embedding": {"model": "embed", "secret": ""},
    }))
    value = model_config.load(path)
    assert value.embedding.protocol == value.generation.protocol
    assert value.embedding.base_url == value.generation.base_url
    assert value.embedding.secret == ""
    assert value.embedding.model == "embed"
    assert value.generation.secret == SENTINEL
    assert set(value.persisted_dict()["generation"]) == {"protocol", "base_url", "secret", "model"}


def test_model_file_is_the_only_source_and_import_is_zero_write(tmp_path):
    storybook_home = tmp_path / "storybook"
    env = _subprocess_env(tmp_path, storybook_home)
    env.update({
        "STORYBOOK_LLM_MODEL": "wrong-model", "ANTHROPIC_AUTH_TOKEN": "wrong-secret",
        "STORYBOOK_EMBED_MODEL": "wrong-embed", "STORYBOOK_EMBED_BASE_URL": "https://wrong.test",
    })
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "generation": {"protocol": "openai", "base_url": "https://models.test", "secret": SENTINEL, "model": "chat"},
        "embedding": {"protocol": "ollama", "base_url": "http://localhost:11434", "secret": "", "model": "embed"},
    }))
    book = str(Path(sys.executable).with_name("book"))
    result = subprocess.run([book, "init", "--config", str(path), "--dry-run", "--json"], env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["plan"]["model_config"]["generation"]["model"] == "chat"
    assert payload["plan"]["model_config"]["embedding"]["model"] == "embed"
    assert SENTINEL not in result.stdout + result.stderr
    assert "wrong-" not in result.stdout
    assert not storybook_home.exists()


@pytest.mark.parametrize("invalid", [
    {"provider": "api"}, {"credential_env": "API_KEY"}, {"secret": None},
    {"model": ""}, {"base_url": "http://host:bad"}, {"protocol": "deepseek"},
])
def test_invalid_file_fields_fail_without_exposing_secret(tmp_path, invalid):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({
        "generation": {"protocol": "openai", "base_url": "https://models.test", "secret": SENTINEL, "model": "chat", **invalid},
        "embedding": {},
    }))
    with pytest.raises(model_config.ModelConfigError) as caught:
        model_config.load(path)
    assert SENTINEL not in str(caught.value)


def test_file_import_persists_and_config_command_redacts_secrets(tmp_path):
    storybook_home = tmp_path / "storybook"
    env = _subprocess_env(tmp_path, storybook_home)
    book = str(Path(sys.executable).with_name("book"))
    source = tmp_path / "models.json"
    with _provider_server(dimension=768) as (url, _):
        source.write_text(json.dumps({
            "generation": {"protocol": "openai", "base_url": url + "/v1", "secret": SENTINEL, "model": "chat"},
            "embedding": {"model": "embed"},
        }))
        result = subprocess.run([book, "init", "--config", str(source), "--yes", "--json"], env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == "ready"
    source.unlink()
    # The imported Profile file, including its secret and detected dimension,
    # remains sufficient after the original source disappears.
    shown = subprocess.run([book, "config"], env=env, text=True, capture_output=True, timeout=15)
    path_result = subprocess.run([book, "config", "--path"], env=env, text=True, capture_output=True, timeout=15)
    assert shown.returncode == path_result.returncode == 0
    config_path = Path(path_result.stdout.strip())
    assert json.loads(config_path.read_text())["generation"]["secret"] == SENTINEL
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(shown.stdout)["embedding_dimension"] == 768
    assert SENTINEL not in shown.stdout + shown.stderr + result.stdout + result.stderr
