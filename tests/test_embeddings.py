"""Embedding keep-alive、请求预算与运行态 warm/cold 状态。"""
from __future__ import annotations

from storybook import config, embeddings

from ._helpers import basis


class _Response:
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


def test_default_embed_rejects_configured_model_drift_before_cache_or_network(
    monkeypatch,
):
    """A configured model change must fail closed until its index is active."""

    def unexpected_call(*args, **kwargs):
        raise AssertionError("model drift must fail before cache or network access")

    monkeypatch.setattr(embeddings.inference_cache, "get", unexpected_call)
    monkeypatch.setattr(embeddings.requests, "post", unexpected_call)
    monkeypatch.setattr(config, "EMBED_MODEL", "future-target-model")

    assert embeddings.embed("query") is None


def test_explicit_shadow_model_remains_available_during_config_switch(monkeypatch):
    """Backfill may explicitly embed with the target model before activation."""

    captured = {}

    def fake_post(url, *, json, timeout):
        captured.update(json)
        return _Response()

    monkeypatch.setattr(embeddings.inference_cache, "get", lambda *args: None)
    monkeypatch.setattr(embeddings.inference_cache, "set", lambda *args: None)
    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    monkeypatch.setattr(config, "EMBED_MODEL", "future-target-model")

    assert embeddings.embed(
        "query", model="future-target-model", cache_version="future-v"
    ) == basis(0)
    assert captured["model"] == "future-target-model"
