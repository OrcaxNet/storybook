"""DeepSeek Anthropic-compatible LLM transport and configuration tests."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import requests

from storybook import config, llm


class _Response:
    def __init__(self, payload=None, *, status=200, json_error=False):
        self._payload = payload
        self.status_code = status
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError("provider response")
            error.response = self
            raise error

    def json(self):
        if self._json_error:
            raise requests.exceptions.JSONDecodeError("bad", "x", 0)
        return self._payload


def test_messages_request_and_text_block_parsing(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response({"content": [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": " hello "},
            {"type": "text", "text": "world"},
        ]})

    monkeypatch.setattr(config, "LLM_BASE_URL", "https://api.deepseek.com/anthropic/")
    monkeypatch.setattr(config, "LLM_API_KEY", "unit-test-secret")
    monkeypatch.setattr(config, "LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(config, "LLM_THINK", False)
    monkeypatch.setattr(llm.requests, "post", fake_post)

    result = llm._chat("question", system="rules", timeout_seconds=1.25, num_predict=384)

    assert result == "hello world"
    assert captured["url"] == "https://api.deepseek.com/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "unit-test-secret"
    assert captured["timeout"] == 1.25
    assert captured["json"] == {
        "model": "deepseek-v4-flash",
        "system": "rules",
        "messages": [{"role": "user", "content": "question"}],
        "max_tokens": 384,
        "temperature": 0.3,
        "thinking": {"type": "disabled"},
    }


def test_thinking_enabled_is_explicit(monkeypatch):
    captured = {}
    monkeypatch.setattr(config, "LLM_API_KEY", "secret")
    monkeypatch.setattr(config, "LLM_THINK", True)
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda url, **kwargs: captured.update(kwargs) or _Response(
            {"content": [{"type": "text", "text": "ok"}]}
        ),
    )

    assert llm._chat("q") == "ok"
    assert captured["json"]["thinking"] == {"type": "enabled"}


@pytest.mark.parametrize("status", [401, 402, 429, 500, 503])
def test_http_failures_return_none_without_leaking_secret(
    monkeypatch, caplog, status
):
    secret = "never-log-this-key"
    monkeypatch.setattr(config, "LLM_API_KEY", secret)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Response(status=status))

    with caplog.at_level(logging.ERROR):
        assert llm._chat("private conversation") is None

    assert f"status={status}" in caplog.text
    assert secret not in caplog.text
    assert "private conversation" not in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        _Response(json_error=True),
        _Response({"content": []}),
        _Response({"content": [{"type": "thinking", "thinking": "x"}]}),
        _Response({"not_content": []}),
    ],
)
def test_invalid_or_empty_response_returns_none(monkeypatch, response):
    monkeypatch.setattr(config, "LLM_API_KEY", "secret")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: response)
    assert llm._chat("q") is None


def test_timeout_returns_none(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "secret")

    def timeout(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(llm.requests, "post", timeout)
    assert llm._chat("q", timeout_seconds=0.2) is None


def test_missing_credentials_skips_request(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *a, **k: pytest.fail("request must not be sent"),
    )
    assert llm._chat("q") is None


def test_config_precedence_aliases_and_tilde_expansion(tmp_path: Path):
    home = tmp_path / "home"
    llm_file = home / ".chrc" / "dpsk.sh"
    llm_file.parent.mkdir(parents=True)
    llm_file.write_text(
        "export ANTHROPIC_AUTH_TOKEN=file-auth\n"
        "ANTHROPIC_BASE_URL=https://file.example/anthropic\n"
        "ANTHROPIC_DEFAULT_HAIKU_MODEL=file-model\n",
        encoding="utf-8",
    )
    project = tmp_path / ".env"
    project.write_text(
        "ANTHROPIC_AUTH_TOKEN=project-auth\n"
        "STORYBOOK_LLM_MODEL=project-model\n",
        encoding="utf-8",
    )

    resolved = config.resolve_llm_config(
        process_env={
            "STORYBOOK_LLM_ENV_FILE": "~/.chrc/dpsk.sh",
            "DEEPSEEK_KEY": "process-fallback",
            "STORYBOOK_LLM_MODEL": "process-model",
        },
        project_env_path=project,
        home=home,
    )

    assert resolved["env_file"] == str(llm_file)
    assert resolved["api_key"] == "process-fallback"
    assert resolved["model"] == "process-model"
    assert resolved["base_url"] == "https://file.example/anthropic"


def test_config_missing_file_and_project_fallback(tmp_path: Path):
    project = tmp_path / ".env"
    project.write_text(
        "DEEPSEEK_KEY=project-key\nANTHROPIC_DEFAULT_HAIKU_MODEL=project-model\n",
        encoding="utf-8",
    )
    resolved = config.resolve_llm_config(
        process_env={"STORYBOOK_LLM_ENV_FILE": "~/missing.sh"},
        project_env_path=project,
        home=tmp_path,
    )
    assert resolved["api_key"] == "project-key"
    assert resolved["model"] == "project-model"
    assert resolved["think"] is False

