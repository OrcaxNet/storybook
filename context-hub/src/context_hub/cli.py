from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

from .store import HubError, Store, write_private
from .transport import Client, Server


def main():
    parser = argparse.ArgumentParser(description="Local-only Context Hub P0 pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["init", "serve", "backup"]:
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        if name == "serve":
            p.add_argument("--port", type=int, default=7341)
        if name == "backup":
            p.add_argument("--output", type=Path, required=True)
    for name in ["call", "grant", "import-claude", "mcp"]:
        p = sub.add_parser(name)
        p.add_argument("--endpoint", default="http://127.0.0.1:7341")
        p.add_argument("--credentials", type=Path, required=True)
        if name == "call":
            p.add_argument("operation")
            p.add_argument("--input", type=Path, help="JSON file; omitted reads stdin")
        if name == "grant":
            p.add_argument("--project", required=True)
            p.add_argument("--consumer", required=True)
            p.add_argument("--output", type=Path, required=True)
        if name == "import-claude":
            p.add_argument("--root", type=Path, required=True)
            p.add_argument("--project", required=True)
            p.add_argument("--session", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if args.command == "init":
            Store.initialize(args.data)
            result = {"initialized": True, "credentials": "admin.json (keep private)"}
        elif args.command in {"serve", "backup"}:
            # OS lock survives neither a process exit nor a crash; prevents split local writers.
            with (args.data / "instance.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise HubError("HUB_ALREADY_RUNNING_STOP_BEFORE_BACKUP", 409)
                store = Store(args.data)
                try:
                    if args.command == "backup":
                        store.backup(args.output)
                        result = {"backup": str(args.output), "restore_requires_latest_revocation_state": True}
                    else:
                        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
                        with Server(store, args.port) as server:
                            print(json.dumps({"ready": True, "endpoint": f"http://127.0.0.1:{server.server_port}"}), flush=True)
                            server.serve_forever(poll_interval=0.2)
                        return
                finally:
                    store.close()
        elif args.command == "mcp":
            from .mcp import run
            run(args.endpoint, args.credentials)
            return
        else:
            client = Client(args.endpoint, args.credentials)
            if args.command == "call":
                if args.operation == "grant":
                    raise HubError("USE_GRANT_COMMAND_TO_WRITE_PRIVATE_CREDENTIALS")
                text = args.input.read_text() if args.input else sys.stdin.read(1024 * 1024 + 1)
                result = client.call(args.operation, json.loads(text))
            elif args.command == "grant":
                if args.output.exists():
                    raise HubError("OUTPUT_EXISTS")
                result = client.call("grant", {"project": args.project, "consumer": args.consumer})
                write_private(args.output, result)
                result = {k: v for k, v in result.items() if k != "token"}
            else:
                from .claude import collect, upload
                # Verify project/token before reading even the explicitly selected source.
                status = client.call("status", {"project": args.project})
                if str(args.root.resolve()) != status.get("capture_root"):
                    raise HubError("SOURCE_SCOPE_MISMATCH", 403)
                result = upload(client, args.project, collect(args.root, args.session))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        return
    except HubError as error:
        print(json.dumps({"error": error.code}), file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, KeyError):
        print(json.dumps({"error": "INVALID_LOCAL_CONFIGURATION_OR_INPUT"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
