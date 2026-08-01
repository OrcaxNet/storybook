"""ContextEnvelope persistence, privacy and environment-aware recall tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import sqlite3
import stat
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

    def test_partial_historical_envelope_does_not_detect_current_machine(
        self, monkeypatch
    ):
        def fail_local_detection(*args, **kwargs):
            raise AssertionError("historical normalization read local environment")

        monkeypatch.setattr(context_module, "_local_device_id", fail_local_detection)
        monkeypatch.setattr(context_module, "_runtime_kind", fail_local_detection)

        envelope = context_module.normalize_envelope({
            "tool": {"type": "cursor", "integration_mode": "log_import"},
        })

        assert envelope["tool"]["type"] == "cursor"
        assert envelope["tool"]["installation_id"] is None
        assert all(value is None for value in envelope["device"].values())
        assert envelope["runtime"] == {
            "kind": None,
            "remote_host_hash": None,
            "container_id_hash": None,
            "shell": None,
            "versions": {},
        }
        for field in (*envelope["device"], *envelope["runtime"]):
            group = "device" if field in envelope["device"] else "runtime"
            assert envelope["provenance"][f"{group}.{field}"] == "unknown"

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

    def test_canonical_sensitive_slots_are_sanitized_before_persistence(self):
        session_id = store.add_session(
            "json",
            "raw",
            context={
                "session": {"external_session_hash": "raw-session-123"},
                "workspace": {
                    "repo_fingerprint": (
                        "https://token@example.com/acme/payments-api.git"
                    ),
                },
                "runtime": {"remote_host_hash": "prod-secret.internal"},
            },
        )

        envelope = store.get_session_context(session_id)
        encoded = json.dumps(envelope, ensure_ascii=False)
        for secret in (
            "raw-session-123",
            "https://token@example.com/acme/payments-api.git",
            "prod-secret.internal",
        ):
            assert secret not in encoded
        assert envelope["session"]["external_session_hash"].startswith(
            "hmac-sha256:"
        )
        assert envelope["workspace"]["repo_fingerprint"].startswith("sha256:")
        assert envelope["runtime"]["remote_host_hash"].startswith("hmac-sha256:")

    def test_safe_canonical_hashes_are_not_hashed_again(self):
        original = context_module.capture_context(
            external_session_id="session-id",
            repo_url="https://github.com/acme/payments.git",
            remote_host="remote.internal",
        )

        normalized = context_module.normalize_envelope({
            "session": {
                "external_session_hash": original["session"]["external_session_hash"],
            },
            "workspace": {
                "repo_fingerprint": original["workspace"]["repo_fingerprint"],
            },
            "runtime": {
                "remote_host_hash": original["runtime"]["remote_host_hash"],
            },
        })

        assert (
            normalized["session"]["external_session_hash"]
            == original["session"]["external_session_hash"]
        )
        assert (
            normalized["workspace"]["repo_fingerprint"]
            == original["workspace"]["repo_fingerprint"]
        )
        assert (
            normalized["runtime"]["remote_host_hash"]
            == original["runtime"]["remote_host_hash"]
        )

    def test_corrupt_hmac_key_is_repaired_once_and_remains_stable(self, tmp_db):
        key_path = tmp_db.parent / ".context_hmac_key"
        key_path.write_bytes(b"short")
        key_path.chmod(0o644)

        with ThreadPoolExecutor(max_workers=8) as executor:
            hashes = list(executor.map(
                lambda _: context_module.external_session_hash("stable-session"),
                range(16),
            ))

        assert len(set(hashes)) == 1
        assert len(key_path.read_bytes()) == 32
        if os.name != "nt":
            assert stat.S_IMODE(key_path.stat().st_mode) == 0o600

    def test_invalid_device_id_is_repaired_and_remains_stable(self, tmp_db):
        device_path = tmp_db.parent / ".device_id"
        device_path.write_text("not-a-uuid", encoding="ascii")
        device_path.chmod(0o644)

        first = context_module.capture_context()["device"]["id"]
        second = context_module.capture_context()["device"]["id"]

        assert first == second == device_path.read_text(encoding="ascii")
        uuid.UUID(first)
        if os.name != "nt":
            assert stat.S_IMODE(device_path.stat().st_mode) == 0o600

    def test_workspace_and_sensitive_alias_provenance_tracks_real_source(self):
        captured = context_module.capture_context(
            workspace_path="/Users/alice/private/payments-api",
            project_label=None,
        )
        assert captured["workspace"]["project_label"] == "payments-api"
        assert captured["provenance"]["workspace.project_label"] == "inferred"

        normalized = context_module.normalize_envelope({
            "workspace": {
                "repo_url": "git@github.com:acme/payments.git",
                "path": "/Users/alice/private/payments-api",
            },
            "runtime": {"remote_host": "remote.internal"},
            "provenance": {
                "workspace.repo_url": "reported",
                "workspace.path": "reported",
                "runtime.remote_host": "detected",
            },
        })

        assert normalized["provenance"]["workspace.repo_fingerprint"] == "reported"
        assert normalized["provenance"]["workspace.project_label"] == "inferred"
        assert normalized["provenance"]["workspace.cwd_alias"] == "inferred"
        assert normalized["provenance"]["workspace.id"] == "inferred"
        assert normalized["provenance"]["runtime.remote_host_hash"] == "detected"

    def test_equivalent_https_and_ssh_repo_urls_share_fingerprint(self):
        https = context_module.repository_fingerprint(
            "https://github.com/Acme/Payments.git"
        )
        scp_ssh = context_module.repository_fingerprint(
            "git@github.com:Acme/Payments.git"
        )
        url_ssh = context_module.repository_fingerprint(
            "ssh://git@github.com/Acme/Payments.git"
        )

        assert https == scp_ssh == url_ssh

    def test_default_repo_ports_do_not_split_the_same_repository(self):
        expected = context_module.repository_fingerprint(
            "git@github.com:acme/payments.git"
        )

        assert context_module.repository_fingerprint(
            "https://github.com:443/acme/payments.git"
        ) == expected
        assert context_module.repository_fingerprint(
            "ssh://git@github.com:22/acme/payments.git"
        ) == expected
        assert context_module.repository_fingerprint(
            "https://github.com:8443/acme/payments.git"
        ) != expected

    def test_cursor_cache_directory_is_not_fabricated_as_workspace(self, tmp_path):
        cache_dir = tmp_path / "workspaceStorage" / "opaque-cache-id"
        cache_dir.mkdir(parents=True)
        vscdb = cache_dir / "state.vscdb"
        db = sqlite3.connect(vscdb)
        try:
            db.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")
            db.execute(
                "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
                (
                    "workbench.panel.aichat.view",
                    json.dumps({"messages": [{"text": "hello"}]}),
                ),
            )
            db.commit()
        finally:
            db.close()

        sessions = collector._extract_from_vscdb(vscdb)

        assert len(sessions) == 1
        workspace = sessions[0]["context"]["workspace"]
        assert workspace == {
            "id": None,
            "repo_fingerprint": None,
            "project_label": None,
            "cwd_alias": None,
            "branch": None,
        }
        for field in workspace:
            assert (
                sessions[0]["context"]["provenance"][f"workspace.{field}"]
                == "unknown"
            )

    def test_cursor_uses_workspace_metadata_when_real_evidence_exists(self, tmp_path):
        cache_dir = tmp_path / "workspaceStorage" / "opaque-cache-id"
        cache_dir.mkdir(parents=True)
        (cache_dir / "workspace.json").write_text(
            json.dumps({"folder": "file:///Users/alice/private/payments-api"}),
            encoding="utf-8",
        )
        vscdb = cache_dir / "state.vscdb"
        db = sqlite3.connect(vscdb)
        try:
            db.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")
            db.execute(
                "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
                (
                    "workbench.panel.aichat.view",
                    json.dumps({"messages": [{"text": "hello"}]}),
                ),
            )
            db.commit()
        finally:
            db.close()

        context = collector._extract_from_vscdb(vscdb)[0]["context"]

        assert context["workspace"]["id"] is not None
        assert context["workspace"]["repo_fingerprint"].startswith("hmac-sha256:")
        assert context["workspace"]["project_label"] == "payments-api"
        assert context["provenance"]["workspace.repo_fingerprint"] == "detected"
        assert context["provenance"]["workspace.project_label"] == "inferred"
        assert "/Users/alice/private/payments-api" not in json.dumps(context)

    @pytest.mark.parametrize(
        "uri",
        ["file:relative/payments-api", "file://remote.internal/payments-api"],
    )
    def test_cursor_rejects_non_absolute_or_non_local_file_uri(self, tmp_path, uri):
        cache_dir = tmp_path / "workspaceStorage" / "opaque-cache-id"
        cache_dir.mkdir(parents=True)
        (cache_dir / "workspace.json").write_text(
            json.dumps({"folder": uri}),
            encoding="utf-8",
        )

        assert collector._cursor_workspace_path(cache_dir / "state.vscdb") is None

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

    def test_claude_log_context_is_independent_of_import_runtime(
        self, tmp_path, monkeypatch
    ):
        log = tmp_path / "stable-session.jsonl"
        log.write_text(
            json.dumps({
                "type": "user",
                "cwd": "/work/repo-a",
                "timestamp": "2026-08-01T01:02:03Z",
                "message": {"role": "user", "content": "diagnose stable failure"},
            }),
            encoding="utf-8",
        )
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        local_context = collector._parse_claude_jsonl(
            log, "stable-session"
        )["context"]
        monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 1 10.0.0.2 22")
        ssh_context = collector._parse_claude_jsonl(
            log, "stable-session"
        )["context"]

        local_context.pop("captured_at")
        ssh_context.pop("captured_at")
        assert local_context == ssh_context
        assert all(value is None for value in local_context["device"].values())
        assert local_context["runtime"]["kind"] is None
        assert local_context["runtime"]["remote_host_hash"] is None


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

    def test_unknown_mcp_tool_does_not_create_false_tool_conflict(self):
        current = context_module.capture_context(
            tool_type="other", integration_mode="mcp", runtime_kind="local"
        )
        story = context_module.capture_context(
            tool_type="claude_code", integration_mode="hook", runtime_kind="local"
        )

        fit = context_module.evaluate_story_context(current, [story], {})

        assert not any("tool differs" in warning for warning in fit["warnings"])
        assert fit["strict_excluded"] is False
