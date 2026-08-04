from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from storybook import config, model_config
from storybook.profiles import PlatformRoots, ProfileRegistry
from storybook.setup_manager import SetupError, SetupManager


SENTINEL = "sb-secret-MUST-NOT-LEAK"


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


@pytest.mark.parametrize(
    ("responses", "code"),
    [
        ([Response({})], "SB_MODEL_GENERATION_FAILED"),
        ([Response({"choices": [{"message": {"content": "OK"}}]}), Response({})], "SB_MODEL_EMBEDDING_FAILED"),
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
