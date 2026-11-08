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
from unittest.mock import patch

from context_hub.claude import visible_text, collect
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
