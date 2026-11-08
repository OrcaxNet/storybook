"""Synthetic local HTTP benchmark, not a retrieval quality evaluation."""
import concurrent.futures
import json
from pathlib import Path
import resource
import statistics
import tempfile
import threading
import time

from context_hub.store import Store
from context_hub.transport import Server, Client


def percentile(samples, p):
    return sorted(samples)[int((len(samples) - 1) * p)]


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "data"
        Store.initialize(data)
        store = Store(data)
        token = json.loads((data / "admin.json").read_text())["token"]
        store.dispatch(token, "project", {"project": "bench", "root": str(root)})
        refs = []
        for start in range(0, 10000, 100):
            r = store.dispatch(token, "import", {"project": "bench", "root": str(root), "session_id": "synthetic",
                "source_version": "benchmark-v1", "messages": [{"id": f"m{i}", "position": i, "role": "user",
                "text": f"E{i:05d} 网关超时问题的记录，环境 staging，日志 /var/log/gateway-{i}.log。"}
                for i in range(start, start + 100)]})
            refs.extend(r["accepted_ids"])
        with Server(store) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}"
            def query(i):
                client = Client(endpoint, data / "admin.json")
                start = time.perf_counter()
                r = client.call("context", {"project": "bench", "query": f"E{i * 97:05d}", "budget_tokens": 6000})
                assert r["items"] and f"E{i * 97:05d}" in r["items"][0]["text"]
                return (time.perf_counter() - start) * 1000
            def read(i):
                start = time.perf_counter()
                Client(endpoint, data / "admin.json").call("evidence", {"project": "bench", "id": refs[i * 97]})
                return (time.perf_counter() - start) * 1000
            try:
                first = query(0)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    context = list(pool.map(query, range(100)))
                    evidence = list(pool.map(read, range(100)))
                print(json.dumps({"corpus": "synthetic, 10000 visible messages, one session", "concurrency": 2,
                    "queries": len(context), "errors": 0, "first_query_ms_not_cold_disk": first,
                    "context_p50_ms": statistics.median(context), "context_p95_ms": percentile(context, .95),
                    "evidence_p95_ms": percentile(evidence, .95),
                    "process_maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "limitation": "Exact identifier retrieval only; no human-labelled quality or cold-start claim"}, indent=2))
            finally:
                server.shutdown()
                thread.join()
                store.close()


if __name__ == "__main__":
    main()
