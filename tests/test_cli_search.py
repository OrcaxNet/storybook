import json

import click
import pytest
from click.testing import CliRunner

from storybook.cli import cli


def _result(*, retrieval_mode="fast", top_matches=None, degraded=False):
    return {
        "query": "开发一个语音机器人",
        "retrieval_mode": retrieval_mode,
        "mode": "lexical_fallback" if degraded else "vector",
        "degraded": degraded,
        "degraded_reasons": ["embedding_timeout"] if degraded else [],
        "latency_ms": {"total": 12.5},
        "keywords": ["语音机器人"],
        "top_matches": top_matches if top_matches is not None else [],
    }


def _match(*, warnings=None):
    return {
        "story_id": 42,
        "title": "未命名记忆",
        "content": "实现语音机器人",
        "keywords": ["voice"],
        "similarity": 0.91,
        "warnings": warnings or [],
        "related": [
            {
                "story_id": 77,
                "title": "关联记忆",
                "content": "语音识别经验",
                "weight": 0.8,
                "edge_type": "semantic",
            }
        ],
    }


@pytest.mark.parametrize("retrieval_mode", ["fast", "auto", "deep"])
def test_search_json_is_clean_and_preserves_story_ids(
    monkeypatch, retrieval_mode
):
    expected = _result(
        retrieval_mode=retrieval_mode,
        top_matches=[_match(warnings=["runtime differs"])],
    )
    monkeypatch.setattr("storybook.cli.store.init_db", lambda: None)
    monkeypatch.setattr(
        "storybook.cli.context_module.capture_context", lambda **kwargs: {}
    )

    def fake_search(query, **kwargs):
        click.echo("search diagnostic", err=True)
        assert kwargs["retrieval_mode"] == retrieval_mode
        return expected

    monkeypatch.setattr("storybook.cli.search_module.search", fake_search)

    result = CliRunner().invoke(
        cli,
        ["search", expected["query"], "--mode", retrieval_mode, "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == expected
    assert payload["top_matches"][0]["story_id"] == 42
    assert payload["top_matches"][0]["related"][0]["story_id"] == 77
    assert "search diagnostic" not in result.stdout
    assert "search diagnostic" in result.stderr


@pytest.mark.parametrize(
    ("top_matches", "degraded"),
    [([], False), ([], True), ([_match(warnings=["os differs"])], True)],
)
def test_search_json_covers_empty_degraded_and_warning_paths(
    monkeypatch, top_matches, degraded
):
    expected = _result(top_matches=top_matches, degraded=degraded)
    monkeypatch.setattr("storybook.cli.store.init_db", lambda: None)
    monkeypatch.setattr(
        "storybook.cli.context_module.capture_context", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        "storybook.cli.search_module.search", lambda query, **kwargs: expected
    )

    result = CliRunner().invoke(
        cli, ["search", expected["query"], "--json"]
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected


def test_search_text_shows_ids_and_keeps_environment_warning(monkeypatch):
    expected = _result(top_matches=[_match(warnings=["os differs"])])
    monkeypatch.setattr("storybook.cli.store.init_db", lambda: None)
    monkeypatch.setattr(
        "storybook.cli.context_module.capture_context", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        "storybook.cli.search_module.search", lambda query, **kwargs: expected
    )

    result = CliRunner().invoke(cli, ["search", expected["query"]])

    assert result.exit_code == 0
    assert "📌 #42 未命名记忆" in result.stdout
    assert "💭 #77" in result.stdout
    assert "当前环境差异: os differs" in result.stdout
