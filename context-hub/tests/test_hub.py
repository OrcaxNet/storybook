import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import asyncio
import os
import subprocess
import sys
import uuid
from unittest.mock import patch

from context_hub.claude import visible_text, collect, upload
from claude_agent_sdk._internal.sessions import project_key_for_directory
from context_hub.store import Store, HubError, write_private
from context_hub.transport import Server, Client


class HubTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        Store.initialize(self.data)
        self.store = Store(self.data)
        self.token = json.loads((self.data / "admin.json").read_text())["token"]
        self.call("project", root=str(self.root))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def call(self, op, **args):
        return self.store.dispatch(self.token, op, {"project": "p", **args})

    def error(self, code, fn, *args, **kwargs):
        with self.assertRaises(HubError) as raised:
            fn(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)

    def revision(self):
        return self.call("status")["revision"]

    def ingest(self, text="网关超时排查：先查询请求日志，不关闭校验。", session="session-a", mid="message-a", **overrides):
        return self.call("import", session_id=session, root=str(self.root), source_version="test-v1",
                         messages=[{"id": mid, "position": 0, "role": "user", "text": text, **overrides}])

    def memory(self, ref, text="先查询请求日志，不关闭校验。"):
        return self.call("propose", text=text, scope="p / test only", evidence_ids=[ref],
                         expected_revision=self.revision())["memory_id"]

    def transcript(self, root, text, cwd=True):
        sid = str(uuid.uuid4())
        directory = self.root / "synthetic-claude" / "projects" / project_key_for_directory(root)
        directory.mkdir(parents=True, exist_ok=True)
        entry = {"type": "user", "uuid": str(uuid.uuid4()), "parentUuid": None,
                 "sessionId": sid, "timestamp": "2026-11-09T00:00:00.000Z",
                 "message": {"role": "user", "content": text}}
        if cwd:
            entry["cwd"] = str(root)
        path = directory / (sid + ".jsonl")
        path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
        return sid, path

    def snapshot(self, text):
        sid, _ = self.transcript(self.root, text)
        with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.root / "synthetic-claude")}):
            return collect(self.root, sid)

    def upload(self, snapshot):
        test = self
        class LocalClient:
            def call(self, op, args):
                return test.call(op, **args)
        return upload(LocalClient(), "p", snapshot)

    def test_real_sdk_rejects_unproven_and_colliding_project_ownership(self):
        approved, other = self.root / "a-b", self.root / "a" / "b"
        approved.mkdir()
        other.mkdir(parents=True)
        self.assertEqual(project_key_for_directory(approved), project_key_for_directory(other))
        with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.root / "synthetic-claude")}):
            missing, _ = self.transcript(other, "OTHER_PROJECT_MARKER", cwd=False)
            self.error("SOURCE_OWNERSHIP_UNVERIFIED", collect, approved, missing)
            mismatch, _ = self.transcript(other, "OTHER_PROJECT_MARKER")
            self.error("SESSION_NOT_IN_APPROVED_ROOT", collect, approved, mismatch)
            normal, _ = self.transcript(approved, "approved text")
            self.assertEqual(collect(approved, normal)["messages"][0]["text"], "approved text")
            self.assertEqual(self.call("sessions")["items"], [])

    def test_real_sdk_checks_all_recorded_cwds_not_just_first(self):
        sid, path = self.transcript(self.root, "approved text")
        other = self.root / "other"
        other.mkdir()
        with path.open("a") as output:
            output.write(json.dumps({"type": "system", "cwd": str(other)}) + "\n")
        with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.root / "synthetic-claude")}):
            self.error("SESSION_NOT_IN_APPROVED_ROOT", collect, self.root, sid)

    def test_real_sdk_source_change_rejected_even_with_same_size_and_mtime(self):
        from claude_agent_sdk import get_session_messages
        sid, path = self.transcript(self.root, "original text")
        stat = path.stat()
        def changed(*args, **kwargs):
            result = get_session_messages(*args, **kwargs)
            path.write_text(path.read_text().replace("original text", "modified text"))
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            return result
        with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.root / "synthetic-claude")}), \
                patch("context_hub.claude.get_session_messages", side_effect=changed):
            self.error("SOURCE_CHANGED_RETRY", collect, self.root, sid)

    def test_real_sdk_boundary_queries_have_readable_traceable_evidence(self):
        for original, query in [("甲" * 19999 + "超时" + "乙" * 20, "超时"),
                                ("x" * 59995 + " ERR_TIMEOUT tail", "ERR_TIMEOUT")]:
            with self.subTest(query=query):
                snapshot = self.snapshot(original)
                self.assertGreater(len(snapshot["messages"]), 1)
                reconstructed = bytearray(len(original.encode()))
                for message in snapshot["messages"]:
                    span = message["source_span"]
                    self.assertLessEqual(len(message["text"].encode()), 60000)
                    self.assertEqual(original.encode()[span["start"]:span["end"]], message["text"].encode())
                    reconstructed[span["start"]:span["end"]] = message["text"].encode()
                self.assertEqual(reconstructed.decode(), original)
                self.upload(snapshot)
                hits = self.call("context", query=query, budget_tokens=6000)["items"]
                self.assertTrue(hits)
                for hit in hits:
                    source = self.call("evidence", id=hit["id"], limit_bytes=65536)
                    self.assertIn(query, source["text"])
                    span = source["source_span"]
                    self.assertEqual(original.encode()[span["start"]:span["end"]].decode(), source["text"])
                session = self.upload(snapshot)["session_ref"]
                refs = [r[0] for r in self.store.db.execute("SELECT id FROM evidence WHERE session=? ORDER BY position", (session,))]
                memory = self.memory(refs[0], "withdraw this source")
                derivative = self.memory(refs[1], "overlap derivative")
                self.call("confirm", id=derivative, reason="test", expected_revision=self.revision())
                self.call("retract", id=memory, reason="wrong scope", expected_revision=self.revision())
                self.upload(snapshot)  # replay must not revive any overlapping chunk
                self.assertEqual(self.call("context", query=query, budget_tokens=6000)["items"], [])
                self.assertEqual(self.call("context", query="derivative", budget_tokens=6000)["items"], [])
                self.assertTrue(all(not self.call("evidence", id=ref)["current"] for ref in refs))

    def test_late_chunk_upload_inherits_source_message_withdrawal(self):
        snapshot = self.snapshot("甲" * 19999 + "超时" + "乙" * 20)
        self.upload({**snapshot, "messages": snapshot["messages"][:1]})
        ref = self.store.db.execute("SELECT id FROM evidence").fetchone()[0]
        memory = self.memory(ref)
        self.call("retract", id=memory, reason="withdraw before final upload", expected_revision=self.revision())
        self.upload(snapshot)
        self.assertEqual(self.call("context", query="超时", budget_tokens=6000)["items"], [])

    def test_memory_wording_is_searchable_only_after_confirmation(self):
        ref = self.ingest("Enable keepalive.")["accepted_ids"][0]
        mid = self.memory(ref, "网关超时处理：启用连接保活。")
        for op in ["context", "recall"]:
            self.assertEqual(self.call(op, query="网关超时", budget_tokens=6000)["items"], [])
        self.call("confirm", id=mid, reason="owner checked", expected_revision=self.revision())
        for op in ["context", "recall"]:
            hit = self.call(op, query="网关超时", budget_tokens=6000)["items"]
            self.assertEqual([item["id"] for item in hit], [mid])
            self.assertEqual(hit[0]["evidence_ids"], [ref])
        both = self.call("context", query="网关超时 keepalive", budget_tokens=6000)
        self.assertEqual(sum(item["id"] == mid for item in both["items"]), 1)
        self.call("retract", id=mid, reason="incorrect", expected_revision=self.revision())
        self.assertEqual(self.call("context", query="网关超时", budget_tokens=6000)["items"], [])

    def test_memory_wording_respects_project_and_all_source_permissions(self):
        first, second = self.ingest("keepalive"), self.ingest("backoff", session="s2")
        mid = self.call("propose", text="独有结论", scope="p", evidence_ids=first["accepted_ids"] + second["accepted_ids"],
                        expected_revision=self.revision())["memory_id"]
        self.call("confirm", id=mid, reason="owner checked", expected_revision=self.revision())
        self.call("project", project="other", root=str(self.root))
        grant = self.call("grant", project="other", consumer="other-reader")
        self.assertEqual(self.store.dispatch(grant["token"], "context", {"project": "other", "query": "独有结论"})["items"], [])
        self.error("FORBIDDEN", self.store.dispatch, grant["token"], "context", {"project": "p", "query": "独有结论"})
        self.call("revoke_source", session_ref=second["session_ref"], reason="access removed", expected_revision=self.revision())
        self.assertEqual(self.call("context", query="独有结论", budget_tokens=6000)["items"], [])
        # Defense in depth: even a stale active row cannot bypass dependency gates.
        with self.store.db:
            self.store.db.execute("UPDATE memories SET status='active' WHERE id=?", (mid,))
        self.assertEqual(self.call("context", query="独有结论", budget_tokens=6000)["items"], [])

    def test_schema_one_upgrade_indexes_active_memory_and_preserves_tombstones(self):
        ref = self.ingest("keepalive")["accepted_ids"][0]
        active = self.memory(ref, "迁移可搜索")
        self.call("confirm", id=active, reason="owner checked", expected_revision=self.revision())
        old_ref = self.ingest("withdrawn", session="old")["accepted_ids"][0]
        old = self.memory(old_ref, "撤回不可搜索")
        self.call("retract", id=old, reason="withdrawn", expected_revision=self.revision())
        # Recreate the previous schema using existing real records and credentials.
        with self.store.db:
            self.store.db.execute("DROP TABLE memory_fts")
            self.store.db.execute("DROP TABLE evidence_spans")
            self.store.db.execute("PRAGMA user_version=1")
        self.store.close()
        with patch("context_hub.store.terms", side_effect=RuntimeError("interrupted index backfill")):
            with self.assertRaisesRegex(RuntimeError, "interrupted index backfill"):
                Store(self.data)
        import sqlite3
        check = sqlite3.connect(self.data / "hub.sqlite")
        try:
            self.assertEqual(check.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertFalse(check.execute("SELECT name FROM sqlite_master WHERE name='memory_fts'").fetchall())
        finally:
            check.close()
        self.store = Store(self.data)
        self.assertEqual(self.call("status")["schema"], 2)
        self.assertEqual(self.call("context", query="迁移", budget_tokens=6000)["items"][0]["id"], active)
        self.assertEqual(self.call("context", query="撤回", budget_tokens=6000)["items"], [])
        self.assertFalse(self.call("evidence", id=old_ref)["current"])
        self.assertIsNone(self.call("evidence", id=ref)["source_span"])
        self.store.close()
        self.store = Store(self.data)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM memory_fts WHERE id=?", (active,)).fetchone()[0], 1)

    def test_evidence_utf8_pagination_hash_and_midpoint(self):
        text = "头" * 4000 + "关键中段：ECONNRESET 用连接池退避" + "尾" * 4000
        ref = self.ingest(text)["accepted_ids"][0]
        offset, pieces = 0, []
        while True:
            r = self.call("evidence", id=ref, cursor=offset, limit_bytes=127)
            pieces.append(r["text"])
            self.assertEqual(r["range"][0], offset)
            if r["next_cursor"] is None:
                break
            offset = r["next_cursor"]
        self.assertEqual("".join(pieces), text)
        self.assertIn("ECONNRESET", self.call("context", query="ECONNRESET", budget_tokens=6000)["items"][0]["text"])
        self.error("INVALID_UTF8_BOUNDARY", self.call, "evidence", id=ref, cursor=1)
        self.error("SOURCE_VERSION_MISMATCH", self.call, "evidence", id=ref, content_hash="wrong")

    def test_idempotence_conflict_and_atomic_batch_rollback(self):
        self.ingest()
        revision = self.revision()
        self.assertEqual(self.ingest()["added"], 0)
        self.assertEqual(self.revision(), revision)
        self.error("EVENT_ID_CONFLICT", self.ingest, text="changed")
        self.assertEqual(self.revision(), revision)
        self.error("UNSUPPORTED_MESSAGE_KIND", self.call, "import", session_id="rollback", root=str(self.root), source_version="v1",
                   messages=[{"id": "valid", "position": 0, "role": "user", "text": "valid"},
                             {"id": "hidden", "position": 1, "role": "thinking", "text": "private"}])
        self.assertEqual(len(self.call("sessions")["items"]), 1)

    def test_changed_snapshot_is_not_silently_relabelled_and_neighbors_are_readable(self):
        result = self.call("import", session_id="neighbors", root=str(self.root), source_version="v1",
                          messages=[{"id": str(i), "role": "user", "position": i, "text": t} for i, t in enumerate(
                              ["旧建议：关闭校验", "这个建议被否决", "只保留测试环境讨论"])])
        middle = self.call("evidence", id=result["accepted_ids"][1], around=1)
        self.assertEqual(middle["before"][0]["id"], result["accepted_ids"][0])
        self.assertEqual(middle["after"][0]["id"], result["accepted_ids"][2])
        self.error("SOURCE_VERSION_CHANGED", self.call, "import", session_id="neighbors", root=str(self.root),
                   source_version="v2", messages=[{"id": "0", "position": 0, "role": "user", "text": "旧建议：关闭校验"}])
        self.assertEqual(self.call("evidence", id=result["accepted_ids"][1])["source_version"], "v1")

    def test_candidates_never_become_requirements_automatically(self):
        ref = self.ingest("此前提议关闭校验，但方案不能用。尝试失败，尚未采纳。")["accepted_ids"][0]
        self.memory(ref, "本项目可以关闭校验")
        r = self.call("context", query="校验", budget_tokens=6000)
        self.assertTrue(r["items"])
        self.assertTrue(all(i["authority"] == "uninterpreted_source" for i in r["items"]))
        self.assertNotIn("本项目可以关闭校验", json.dumps(r, ensure_ascii=False))

    def test_retract_blocks_original_snippet_other_derivatives_and_late_proposal(self):
        ref = self.ingest()["accepted_ids"][0]
        mid = self.memory(ref)
        self.call("confirm", id=mid, reason="owner checked support", expected_revision=self.revision())
        self.memory(ref, "another dependent candidate")
        stale = self.revision()
        self.assertEqual(self.call("context", query="超时", budget_tokens=6000)["items"][0]["kind"], "memory")
        self.call("retract", id=mid, reason="scope was wrong", expected_revision=stale)
        self.assertEqual(self.call("context", query="超时", budget_tokens=6000)["items"], [])
        self.assertFalse(self.call("evidence", id=ref)["current"])
        self.error("STALE_REVISION", self.call, "propose", text="late job", scope="p", evidence_ids=[ref], expected_revision=stale)
        self.error("EVIDENCE_UNAVAILABLE", self.memory, ref)
        self.error("INVALID_TRANSITION", self.call, "confirm", id=mid, reason="retry", expected_revision=self.revision())
        self.ingest()  # replay cannot reset current=0
        self.assertEqual(self.call("context", query="超时", budget_tokens=6000)["items"], [])

    def test_project_scope_reader_revoke_and_no_write_authority(self):
        a = self.ingest()["accepted_ids"][0]
        self.call("project", project="other", root=str(self.root))
        b = self.call("import", project="other", session_id="s", root=str(self.root), source_version="v1",
                      messages=[{"id": "m", "role": "user", "position": 0, "text": "受限项目超时秘诀"}])["accepted_ids"][0]
        grant = self.call("grant", consumer="local-pilot")
        token = grant["token"]
        self.error("FORBIDDEN", self.store.dispatch, token, "context", {"project": "other", "query": "超时"})
        self.error("EVIDENCE_UNAVAILABLE", self.store.dispatch, token, "evidence", {"project": "p", "id": b})
        self.error("FORBIDDEN", self.store.dispatch, token, "retract", {"project": "p"})
        self.error("EVIDENCE_UNAVAILABLE", self.call, "propose", text="mixed", scope="p", evidence_ids=[a, b], expected_revision=self.revision())
        self.call("revoke", token_hash=grant["token_hash"])
        self.error("UNAUTHORIZED", self.store.dispatch, token, "status", {"project": "p"})

    def test_source_revocation_blocks_all_contributing_dependencies(self):
        a = self.ingest()
        b = self.ingest("部署步骤：使用回滚入口", session="s2")
        mid = self.call("propose", text="两个来源共同结论", scope="p", evidence_ids=a["accepted_ids"] + b["accepted_ids"], expected_revision=self.revision())["memory_id"]
        self.call("confirm", id=mid, reason="reviewed", expected_revision=self.revision())
        self.call("revoke_source", session_ref=b["session_ref"], reason="no access", expected_revision=self.revision())
        r = self.call("context", query="超时", budget_tokens=6000)
        self.assertNotIn("两个来源共同结论", json.dumps(r, ensure_ascii=False))
        self.error("EVIDENCE_UNAVAILABLE", self.call, "evidence", id=b["accepted_ids"][0])
        self.error("SOURCE_REVOKED", self.ingest, "部署步骤：使用回滚入口", session="s2")

    def test_secrets_redacted_before_indexing(self):
        ref = self.ingest("API_KEY=sk-abcdefghijklmnop password=hunter123 正常超时日志")["accepted_ids"][0]
        r = self.call("evidence", id=ref)
        self.assertTrue(r["redacted"])
        self.assertNotIn("hunter123", r["text"])
        self.assertEqual(self.call("context", query="hunter123", budget_tokens=6000)["items"], [])

    def test_short_chinese_match_is_literal_not_query_code(self):
        self.ingest("超时之后回滚，错误码 E_TIMEOUT 路径 /var/log/gateway.log")
        for q in ["超时", "回滚", "E_TIMEOUT", "/var/log/gateway.log", '超时\" OR \"回滚']:
            self.assertTrue(self.call("recall", query=q, budget_tokens=6000)["items"])

    def test_whole_context_budget_and_historical_fail_closed(self):
        self.ingest()
        r = self.call("context", query="超时", budget_tokens=2000)
        self.assertLessEqual(len(json.dumps(r, ensure_ascii=False).encode()), 2000)
        self.error("BUDGET_TOO_SMALL", self.call, "context", query="超时", budget_tokens=64)
        self.error("AS_OF_UNAVAILABLE", self.call, "context", query="超时", known_at="yesterday")

    def test_backup_is_consistent_and_retraction_survives_reopen(self):
        ref = self.ingest()["accepted_ids"][0]
        mid = self.memory(ref)
        self.call("retract", id=mid, reason="withdrawn", expected_revision=self.revision())
        target = self.root / "snapshot.sqlite"
        self.store.backup(target)
        import sqlite3
        with sqlite3.connect(target) as backup:
            self.assertEqual(backup.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(backup.execute("SELECT current FROM evidence WHERE id=?", (ref,)).fetchone()[0], 0)
        backup.close()
        self.store.close()
        self.store = Store(self.data)
        self.assertEqual(self.call("context", query="超时", budget_tokens=6000)["items"], [])

    def test_transport_auth_size_and_redirect_boundaries(self):
        with Server(self.store) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                endpoint = f"http://127.0.0.1:{server.server_port}"
                client = Client(endpoint, self.data / "admin.json")
                self.assertEqual(client.call("status", {"project": "p"})["model_egress"], "disabled")
                req = urllib.request.Request(endpoint + "/v1/status", data=b'{}', headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(req)
                self.assertEqual(e.exception.code, 401)
                e.exception.close()
                self.error("LOOPBACK_ENDPOINT_REQUIRED", Client, "http://example.com:80", self.data / "admin.json")
                self.error("LOOPBACK_ENDPOINT_REQUIRED", Client, "http://127.0.0.1:80@evil.test:80", self.data / "admin.json")
            finally:
                server.shutdown()
                thread.join()

    def test_adapter_only_visible_text_and_approved_root(self):
        text, skipped = visible_text({"content": [{"type": "thinking", "thinking": "NEVER STORE"},
                                                  {"type": "text", "text": "visible"},
                                                  {"type": "tool_result", "content": [{"type": "text", "text": "result"}]}]})
        self.assertNotIn("NEVER STORE", text)
        self.assertIn("result", text)
        self.assertEqual(skipped, 1)
        with patch("context_hub.claude.discover", return_value=iter([])), patch("context_hub.claude.get_session_messages") as read:
            self.error("SESSION_NOT_IN_APPROVED_ROOT", collect, self.root, "missing")
            read.assert_not_called()

    def test_process_crash_before_and_after_commit(self):
        # Fault injection exits its OWN child process, never another task/daemon.
        script = '''
import json,os,sys
from pathlib import Path
from context_hub.store import Store
d=Path(sys.argv[1]); s=Store(d); t=json.loads((d/'admin.json').read_text())['token']
if sys.argv[2]=='before':
 s.bump=lambda *a: os._exit(7)
r=s.dispatch(t,'import',{'project':'p','root':sys.argv[3],'session_id':'crash-'+sys.argv[2],
 'source_version':'v1','messages':[{'id':'m','role':'user','position':0,'text':'durable evidence'}]})
print(json.dumps(r),flush=True)
os._exit(7)
'''
        for point in ["before", "after"]:
            r = subprocess.run([sys.executable, "-c", script, str(self.data), point, str(self.root)], capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 7)
            found = [s for s in self.call("sessions")["items"] if s["native_id"] == "crash-" + point]
            self.assertEqual(len(found), int(point == "after"))
            if point == "after":
                ref = json.loads(r.stdout)["accepted_ids"][0]
                self.assertEqual(self.call("evidence", id=ref)["text"], "durable evidence")

    def test_real_stdio_mcp_roundtrip_and_revoked_reader(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        self.ingest()
        grant = self.call("grant", consumer="synthetic-test")
        credentials = self.root / "reader.json"
        write_private(credentials, grant)
        with Server(self.store) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            async def exercise():
                params = StdioServerParameters(command=sys.executable, args=["-m", "context_hub.cli", "mcp",
                    "--endpoint", f"http://127.0.0.1:{server.server_port}", "--credentials", str(credentials)])
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        self.assertEqual({t.name for t in tools.tools}, {"context_get", "recall", "evidence_read", "sessions_list"})
                        r = await session.call_tool("context_get", {"query": "超时", "budget_tokens": 6000})
                        self.assertFalse(r.isError)
                        self.assertIn("校验", r.content[0].text)
                        self.call("revoke", token_hash=grant["token_hash"])
                        r = await session.call_tool("recall", {"query": "超时"})
                        self.assertTrue(r.isError)
            try:
                asyncio.run(exercise())
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
