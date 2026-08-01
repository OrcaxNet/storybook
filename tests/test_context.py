"""ContextEnvelope persistence, privacy and environment-aware recall tests."""
from __future__ import annotations

import json
import uuid

import pytest

from storybook import collector, context as context_module, search, store
from ._helpers import basis, with_cos


def _context(
    *,
    tool: str,
    runtime: str,
    workspace: str | None = None,
) -> dict:
    return context_module.capture_context(
        tool_type=tool,
        integration_mode="log_import",
        runtime_kind=runtime,
        workspace_path=workspace,
        project_label=workspace.rsplit("/", 1)[-1] if workspace else None,
    )


class TestContextEnvelopePersistence:
    def test_new_session_has_complete_explicit_envelope(self):
        session_id = store.add_session("manual", "raw", "problem")
        row = store.get_session(session_id)
        envelope = store.get_session_context(session_id)

        assert set(envelope) == {
            "profile_id", "tool", "device", "session", "workspace", "runtime",
            "captured_at", "provenance",
        }
        uuid.UUID(envelope["profile_id"])
        uuid.UUID(envelope["device"]["id"])
        uuid.UUID(envelope["session"]["id"])
        assert envelope["session"]["id"] == row["global_id"]
        assert envelope["runtime"]["kind"] in context_module.RUNTIME_KINDS
        assert envelope["captured_at"].endswith("Z")

        # Every unknown leaf is explicit null/empty-map plus provenance=unknown.
        for group in ("tool", "device", "session", "workspace", "runtime"):
            for key, value in envelope[group].items():
                assert value != ""
                if value is None or value == {}:
                    assert envelope["provenance"][f"{group}.{key}"] == "unknown"

        db = store.get_db()
        try:
            assert db.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM agent_installations").fetchone()[0] == 1
        finally:
            db.close()

    def test_unknown_runtime_is_not_presented_as_detected_fact(self):
        envelope = context_module.normalize_envelope({
            "runtime": {"kind": "not-a-runtime"},
            "workspace": {"id": None},
        })
        assert envelope["runtime"]["kind"] == "unknown"
        assert envelope["provenance"]["runtime.kind"] == "unknown"
        assert envelope["workspace"]["id"] is None
        assert envelope["provenance"]["workspace.id"] == "unknown"

    def test_sensitive_adapter_values_are_hashed_or_aliased(self):
        envelope = context_module.capture_context(
            tool_type="codex",
            integration_mode="mcp",
            external_session_id="raw-session-123",
            workspace_path="/Users/alice/private/payments-api",
            repo_url="https://token@example.com/acme/payments-api.git",
            project_label="payments-api",
            runtime_kind="ssh",
            remote_host="prod-secret.internal",
            container_id="container-secret",
        )
        encoded = json.dumps(envelope, ensure_ascii=False)

        for secret in (
            "raw-session-123",
            "/Users/alice/private/payments-api",
            "https://token@example.com/acme/payments-api.git",
            "prod-secret.internal",
            "container-secret",
        ):
            assert secret not in encoded
        assert envelope["session"]["external_session_hash"].startswith("hmac-sha256:")
        assert envelope["workspace"]["repo_fingerprint"].startswith("sha256:")
        assert envelope["runtime"]["remote_host_hash"].startswith("hmac-sha256:")
        assert envelope["workspace"]["project_label"] == "payments-api"

    def test_claude_adapter_reports_context_without_raw_external_id(self, tmp_path):
        log = tmp_path / "session-raw-id.jsonl"
        log.write_text(
            "\n".join([
                json.dumps({
                    "type": "user",
                    "cwd": "/Users/alice/private/repo-a",
                    "version": "1.2.3",
                    "gitBranch": "feature/context",
                    "timestamp": "2026-08-01T01:02:03Z",
                    "message": {"role": "user", "content": "How do I fix this failure?"},
                }),
                json.dumps({
                    "type": "assistant",
                    "message": {"role": "assistant", "content": "Use the safe fix."},
                }),
            ]),
            encoding="utf-8",
        )

        parsed = collector._parse_claude_jsonl(log, "session-raw-id")
        assert parsed["source"] == "claude_code"
        assert "session-raw-id" not in json.dumps(parsed["context"])
        assert parsed["context"]["tool"]["type"] == "claude_code"
        assert parsed["context"]["tool"]["version"] == "1.2.3"
        assert parsed["context"]["workspace"]["project_label"] == "repo-a"


class TestStoryEnvironmentHistory:
    def test_story_keeps_all_source_environments(self):
        claude = store.add_session(
            "claude_code", "raw-a", context=_context(
                tool="claude_code", runtime="local", workspace="/work/repo-a"
            )
        )
        cursor = store.add_session(
            "cursor", "raw-b", context=_context(
                tool="cursor", runtime="devcontainer", workspace="/work/repo-b"
            )
        )
        story_id = store.add_story(
            "memory", "content", [], basis(0), source_session_ids=[claude]
        )
        store.update_story_raw_sessions(story_id, [claude, cursor])

        story = store.get_story(story_id)
        assert story["source_session_ids"] == [claude, cursor]
        assert len(story["environments"]) == 2
        assert {item["tool"]["type"] for item in story["environments"]} == {
            "claude_code", "cursor",
        }
        assert {item["runtime"]["kind"] for item in story["environments"]} == {
            "local", "devcontainer",
        }


class TestEnvironmentAwareRecall:
    def _seed_two_runtimes(self):
        dev_session = store.add_session(
            "cursor", "dev", context=_context(tool="cursor", runtime="devcontainer")
        )
        local_session = store.add_session(
            "codex", "local", context=_context(tool="codex", runtime="local")
        )
        dev_story = store.add_story(
            "dev experience", "content", [], with_cos(0, 0.80),
            source_session_ids=[dev_session],
        )
        local_story = store.add_story(
            "local experience", "content", [], with_cos(0, 0.80),
            source_session_ids=[local_session],
        )
        return dev_story, local_story

    def test_default_scope_soft_reranks_but_keeps_cross_environment(self, fake_embedder):
        dev_story, local_story = self._seed_two_runtimes()
        fake_embedder.register("q", basis(0))
        current = _context(tool="codex", runtime="local")

        result = search.search("q", top_k=2, context=current)

        assert [item["story_id"] for item in result["top_matches"]] == [
            local_story, dev_story,
        ]
        assert result["top_matches"][0]["score"] > result["top_matches"][1]["score"]
        assert result["top_matches"][1]["warnings"]
        assert result["strict_filtered"] == 0

    def test_strict_scope_filters_environment_conflict_only_when_requested(self, fake_embedder):
        dev_story, local_story = self._seed_two_runtimes()
        fake_embedder.register("q", basis(0))
        current = _context(tool="codex", runtime="local")

        result = search.search("q", top_k=2, context=current, scope="strict")

        assert [item["story_id"] for item in result["top_matches"]] == [local_story]
        assert dev_story not in [item["story_id"] for item in result["top_matches"]]
        assert result["strict_filtered"] == 1

    def test_explicit_exclusion_warns_soft_and_filters_strict(self, fake_embedder):
        session_id = store.add_session(
            "codex", "raw", context=_context(tool="codex", runtime="local")
        )
        story_id = store.add_story(
            "excluded local", "content", [], basis(0),
            source_session_ids=[session_id],
            applicability={
                "applies_when": [],
                "excludes_when": [{"field": "runtime.kind", "in": ["local"]}],
            },
        )
        fake_embedder.register("q", basis(0))
        current = _context(tool="codex", runtime="local")

        soft = search.search("q", context=current)
        assert soft["top_matches"][0]["story_id"] == story_id
        assert "current context matches excludes_when" in soft["top_matches"][0]["warnings"]

        strict = search.search("q", context=current, scope="strict")
        assert strict["top_matches"] == []
        assert strict["strict_filtered"] == 1

    def test_invalid_scope_is_rejected(self, fake_embedder):
        fake_embedder.register("q", basis(0))
        with pytest.raises(ValueError, match="scope"):
            search.search("q", scope="project-only")
