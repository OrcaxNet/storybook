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
