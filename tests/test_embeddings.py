"""Embedding keep-alive、请求预算与运行态 warm/cold 状态。"""
from __future__ import annotations

import requests

from storybook import config, embeddings

from ._helpers import basis


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"embedding": basis(0)}


def test_embed_sends_keep_alive_and_bounded_http_timeout(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    result = embeddings.embed(
        "query", timeout_seconds=2.0, keep_alive="7m"
    )

    assert result == basis(0)
    assert captured["url"] == f"{config.OLLAMA_HOST}/api/embeddings"
    assert captured["json"] == {
        "model": config.EMBED_MODEL,
        "prompt": "query",
        "keep_alive": "7m",
    }
    assert captured["timeout"] == (1.0, 2.0)
    assert embeddings.model_state() == "warm"


def test_prewarm_uses_cold_budget_and_configured_keep_alive(monkeypatch):
    captured = {}

    def fake_embed(text, **kwargs):
        captured.update({"text": text, **kwargs})
        return basis(0)

    monkeypatch.setattr(embeddings, "embed", fake_embed)

    assert embeddings.prewarm() is True
    assert captured["text"] == "storybook embedding warmup"
    assert captured["timeout_seconds"] == config.QUERY_COLD_TIMEOUT_SECONDS
    assert captured["keep_alive"] == config.EMBED_KEEP_ALIVE


def test_default_embed_uses_active_serving_model_during_config_switch(monkeypatch):
    """Changing target config must not mix query vectors into the old index."""

    from storybook import store

    active_model = store.get_embedding_index_state()["active_model"]
    captured = {}

    def fake_post(url, *, json, timeout):
        captured.update(json)
        captured["url"] = url
        return _Response()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    monkeypatch.setattr(config, "EMBED_MODEL", "future-target-model")
    monkeypatch.setattr(config, "EMBED_BASE_URL", "https://endpoint-b.example/v1")
    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")

    assert embeddings.embed("query") == basis(0)
    assert captured["model"] == active_model
    assert captured["url"] == f"{config.OLLAMA_HOST}/api/embeddings"


def test_endpoint_switch_uses_active_credentials_until_atomic_activation(monkeypatch):
    from storybook import store

    story_id = store.add_story("credential routing", "content", [], basis(0))
    db = store.get_db(load_vector_extension=False)
    try:
        db.execute(
            """UPDATE embedding_index_state
               SET active_endpoint = 'https://endpoint-a.example/v1',
                   active_adapter = 'openai_compatible',
                   active_api_key_env = 'TOKEN_A'
               WHERE id = 1"""
        )
        db.commit()
    finally:
        db.close()

    monkeypatch.setattr(config, "EMBED_BASE_URL", "https://endpoint-b.example/v1")
    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")
    monkeypatch.setattr(config, "EMBED_API_KEY_ENV", "TOKEN_B")
    monkeypatch.setenv("TOKEN_A", "alpha-token")
    monkeypatch.setenv("TOKEN_B", "beta-token")
    requests_seen = []

    class Response(_Response):
        def json(self):
            return {"data": [{"embedding": basis(1)}]}

    def fake_post(url, **kwargs):
        requests_seen.append((url, kwargs.get("headers")))
        return Response()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)

    assert embeddings.embed("before activation") == basis(1)
    assert requests_seen[-1] == (
        "https://endpoint-a.example/v1/embeddings",
        {"Authorization": "Bearer alpha-token"},
    )

    result = embeddings.backfill(
        model=config.EMBED_MODEL,
        version=config.EMBED_VERSION,
        batch_size=10,
    )
    assert result["activation"]["activated"] == 1
    assert requests_seen[-1] == (
        "https://endpoint-b.example/v1/embeddings",
        {"Authorization": "Bearer beta-token"},
    )
    active_state = store.get_embedding_index_state()
    assert active_state["active_api_key_env"] == "TOKEN_B"
    assert "alpha-token" not in str(active_state)
    assert "beta-token" not in str(active_state)

    assert embeddings.embed("after activation") == basis(1)
    assert requests_seen[-1] == (
        "https://endpoint-b.example/v1/embeddings",
        {"Authorization": "Bearer beta-token"},
    )
    assert store.get_story(story_id) is not None


def test_default_embed_rejects_target_dimension_before_serving_index(monkeypatch):
    """A target API cannot inject its new dimension into the old active index."""

    class Response(_Response):
        def json(self):
            return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")
    monkeypatch.setattr(config, "EMBED_DIM", 2)
    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: Response())

    assert embeddings.embed("query") is None


def test_default_embed_keeps_old_serving_dimension_during_target_backfill(monkeypatch):
    monkeypatch.setattr(config, "EMBED_DIM", 2)
    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: _Response())

    assert embeddings.embed("old serving query") == basis(0)


def test_openai_compatible_api_uses_no_ollama_endpoint_or_parameters(monkeypatch):
    captured = {}

    class Response(_Response):
        def json(self):
            return {"data": [{"embedding": basis(0)}]}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")
    monkeypatch.setattr(config, "EMBED_BASE_URL", "https://embed.example/v1")
    monkeypatch.setattr(config, "EMBED_API_KEY_ENV", "")
    monkeypatch.setattr(embeddings.requests, "post", fake_post)

    assert embeddings.embed(
        "query", model=config.EMBED_MODEL, keep_alive="7m"
    ) == basis(0)
    assert captured["url"] == "https://embed.example/v1/embeddings"
    assert captured["json"] == {"model": config.EMBED_MODEL, "input": "query"}
    assert "headers" not in captured


def test_api_credential_is_read_from_named_environment_variable(monkeypatch):
    captured = {}

    class Response(_Response):
        def json(self):
            return {"data": [{"embedding": basis(0)}]}

    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")
    monkeypatch.setattr(config, "EMBED_API_KEY_ENV", "PRIVATE_EMBED_TOKEN")
    monkeypatch.setenv("PRIVATE_EMBED_TOKEN", "never-log-this")
    monkeypatch.setattr(
        embeddings.requests,
        "post",
        lambda url, **kwargs: captured.update(kwargs) or Response(),
    )

    assert embeddings.embed("query", model=config.EMBED_MODEL) == basis(0)
    assert captured["headers"] == {"Authorization": "Bearer never-log-this"}
    assert "never-log-this" not in str(embeddings.api_identity())


def test_probe_classifies_credentials_protocol_and_dimension(monkeypatch):
    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")
    monkeypatch.setattr(config, "EMBED_API_KEY_ENV", "MISSING_EMBED_TOKEN")
    monkeypatch.delenv("MISSING_EMBED_TOKEN", raising=False)
    assert embeddings.probe()["reason"] == "credentials_missing"

    monkeypatch.setattr(config, "EMBED_API_KEY_ENV", "")

    class BadProtocol(_Response):
        def json(self):
            return {"embedding": basis(0)}

    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: BadProtocol())
    assert embeddings.probe()["reason"] == "response_protocol_incompatible"

    class WrongDimension(_Response):
        def json(self):
            return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: WrongDimension())
    result = embeddings.probe()
    assert result == {"ok": False, "reason": "dimension_mismatch", "dimension": 2}


def test_probe_classifies_endpoint_and_authentication_failures(monkeypatch):
    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")
    monkeypatch.setattr(config, "EMBED_API_KEY_ENV", "")
    monkeypatch.setattr(
        embeddings.requests,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")),
    )
    assert embeddings.probe()["reason"] == "endpoint_unreachable"

    class Unauthorized(_Response):
        status_code = 401

    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: Unauthorized())
    assert embeddings.probe()["reason"] == "authentication_failed"


def test_successful_ollama_probe_marks_model_warm(monkeypatch):
    embeddings.mark_model_cold()
    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: _Response())

    assert embeddings.probe()["ok"] is True
    assert embeddings.model_state() == "warm"


def test_embedding_cache_identity_isolated_by_endpoint_adapter_model_and_version(
    monkeypatch,
):
    first = embeddings.api_identity()
    monkeypatch.setattr(config, "EMBED_BASE_URL", "https://other.example/v1")
    monkeypatch.setattr(config, "EMBED_ADAPTER", "openai_compatible")
    monkeypatch.setattr(config, "EMBED_MODEL", "other-model")
    monkeypatch.setattr(config, "EMBED_VERSION", "other-version")
    second = embeddings.api_identity()

    assert first != second
    assert second == {
        "type": "api",
        "base_url": "https://other.example/v1",
        "adapter": "openai_compatible",
        "credential_env": config.EMBED_API_KEY_ENV,
        "model": "other-model",
        "version": "other-version",
        "dimension": config.EMBED_DIM,
    }
