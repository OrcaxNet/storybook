"""Mixed-provider doctor checks."""
from __future__ import annotations

from storybook import config, health


def _ready_local_dependencies(monkeypatch, tmp_path) -> None:
    database = tmp_path / "memory.db"
    database.touch()
    monkeypatch.setattr(config, "DB_PATH", database)
    monkeypatch.setattr(
        health,
        "_check_ollama_reachable",
        lambda: (
            True,
            {"models": [{"name": config.EMBED_MODEL}]},
            "",
        ),
    )
    monkeypatch.setattr(
        health,
        "_probe_embed_dim",
        lambda: (True, config.EMBED_DIM, ""),
    )
    monkeypatch.setattr(health.store, "check_vec_extension", lambda: True)
    monkeypatch.setattr(health.store, "stories_table_exists", lambda: True)
    monkeypatch.setattr(health.store, "story_vectors_table_exists", lambda: True)
    monkeypatch.setattr(
        health.store,
        "vector_consistency",
        lambda: {
            "missing_vec": [],
            "orphan_vec": [],
            "blob_count": 1,
            "vec_count": 1,
        },
    )


def test_doctor_accepts_deepseek_config_and_only_checks_ollama_embedding(
    monkeypatch, tmp_path, capsys
):
    _ready_local_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "LLM_API_KEY", "never-print-this-secret")
    monkeypatch.setattr(
        health.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("doctor must not send a generation request")
        ),
    )

    assert health.run_doctor() is True

    output = capsys.readouterr().out
    assert "LLM 配置" in output
    assert "provider=deepseek_anthropic" in output
    assert "ollama pull deepseek-v4-flash" not in output
    assert "never-print-this-secret" not in output


def test_doctor_reports_missing_llm_credentials_without_network_or_secret(
    monkeypatch, tmp_path, capsys
):
    _ready_local_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    monkeypatch.setattr(
        health.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("doctor must not send a generation request")
        ),
    )

    assert health.run_doctor() is False

    output = capsys.readouterr().out
    assert "reason=llm_credentials_missing" in output
    assert "ANTHROPIC_AUTH_TOKEN" in output
    assert "ollama pull deepseek-v4-flash" not in output
