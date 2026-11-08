"""Explicitly opt-in to a newly-created synthetic Claude session; no model invocation."""
import argparse
import json
from pathlib import Path
import tempfile
import threading

from context_hub.claude import collect, upload
from context_hub.store import Store
from context_hub.transport import Server, Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    snapshot = collect(args.root, args.session)
    if not any("FF-122 的公开合成验收语料" in m["text"] for m in snapshot["messages"]):
        raise SystemExit("Refusing non-pilot data: create a dedicated synthetic session first")
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / "hub"
        Store.initialize(data)
        store = Store(data)
        with Server(store) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            client = Client(f"http://127.0.0.1:{server.server_port}", data / "admin.json")
            try:
                client.call("project", {"project": "pilot", "root": str(args.root.resolve())})
                receipt = upload(client, "pilot", snapshot)
                replay = upload(client, "pilot", snapshot)
                assert replay["added"] == 0
                questions = [("日志位置", "/var/log/pilot-gateway.log"), ("超时阈值", "30 秒"),
                             ("HTTP 429", "1、2、4"), ("回滚目标", "pilot-v1"), ("证书校验", "被否决")]
                answers = []
                for query, expected in questions:
                    result = client.call("context", {"project": "pilot", "query": query, "budget_tokens": 6000})
                    assert expected in json.dumps(result, ensure_ascii=False), query
                    ref = result["items"][0]["evidence_ids"][0]
                    chunks, cursor = [], 0
                    while True:
                        evidence = client.call("evidence", {"project": "pilot", "id": ref, "cursor": cursor, "limit_bytes": 1024})
                        chunks.append(evidence["text"])
                        if evidence["next_cursor"] is None:
                            break
                        cursor = evidence["next_cursor"]
                    assert expected in "".join(chunks)
                    answers.append({"query": query, "supported_answer": expected, "source_bytes": len("".join(chunks).encode()),
                                    "evidence_resolved": True})
                revision = client.call("status", {"project": "pilot"})["revision"]
                proposal = client.call("propose", {"project": "pilot", "expected_revision": revision,
                    "text": "证书校验必须开启，仅适用于 staging 验收。", "scope": "pilot/staging", "evidence_ids": [ref]})
                confirmation = client.call("confirm", {"project": "pilot", "id": proposal["memory_id"],
                    "expected_revision": proposal["revision"], "reason": "pilot owner checks explicit rejection in source"})
                client.call("retract", {"project": "pilot", "id": proposal["memory_id"],
                    "expected_revision": confirmation["revision"], "reason": "exercise withdrawal, not a change to real requirements"})
                assert not client.call("context", {"project": "pilot", "query": "证书校验", "budget_tokens": 6000})["items"]
                after = collect(args.root, args.session)
                assert after == snapshot, "source history changed during read-only pilot"
                print(json.dumps({"source": "real Claude CLI persisted synthetic user message",
                    "source_version": snapshot["source_version"], "coverage": receipt["coverage"],
                    "questions": answers, "source_unchanged": True, "replay_added": replay["added"],
                    "retraction_effective_next_read": True,
                    "limitation": "No successful assistant response or live observation is implied"}, ensure_ascii=False, indent=2))
            finally:
                server.shutdown()
                thread.join()
                store.close()


if __name__ == "__main__":
    main()
