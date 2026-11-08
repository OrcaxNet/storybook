"""Loopback-only JSON API. Credentials are never accepted from request bodies."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .store import HubError, Store

MAX_BODY = 1024 * 1024


class Server(HTTPServer):
    def __init__(self, store: Store, port: int = 0):
        self.store = store
        super().__init__(("127.0.0.1", port), Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, *_):
        pass  # No body, URL, token or private data in access logs.

    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health/live":
            return self.reply(200, {"live": True})
        if self.path == "/health/ready":
            try:
                self.server.store.db.execute("SELECT 1").fetchone()
                return self.reply(200, {"ready": True, "models": "disabled"})
            except Exception:
                return self.reply(503, {"ready": False})
        self.reply(404, {"error": "NOT_FOUND"})

    def do_POST(self):
        try:
            if self.headers.get("Origin") or self.headers.get("Transfer-Encoding"):
                raise HubError("UNSUPPORTED_REQUEST", 400)
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                raise HubError("INVALID_HOST", 400)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise HubError("JSON_REQUIRED", 415)
            sizes = self.headers.get_all("Content-Length", [])
            if len(sizes) != 1 or not sizes[0].isdigit() or not 0 < int(sizes[0]) <= MAX_BODY:
                raise HubError("INVALID_CONTENT_LENGTH", 413)
            auth = self.headers.get_all("Authorization", [])
            if len(auth) != 1 or not auth[0].startswith("Bearer "):
                raise HubError("UNAUTHORIZED", 401)
            if not self.path.startswith("/v1/") or "?" in self.path:
                raise HubError("NOT_FOUND", 404)
            payload = json.loads(self.rfile.read(int(sizes[0])))
            if not isinstance(payload, dict):
                raise HubError("JSON_OBJECT_REQUIRED")
            result = self.server.store.dispatch(auth[0][7:], self.path[4:], payload)
            self.reply(200, result)
        except HubError as error:
            self.reply(error.status, {"error": error.code})
        except (ValueError, TypeError, KeyError, AttributeError):
            self.reply(400, {"error": "INVALID_REQUEST"})
        except (socket.timeout, BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self.reply(500, {"error": "INTERNAL_ERROR"})


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise HubError("REDIRECT_REFUSED")


class Client:
    def __init__(self, endpoint: str, credentials: Path):
        url = urlsplit(endpoint)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
                or url.path not in {"", "/"} or url.query or url.fragment or url.username or url.password):
            raise HubError("LOOPBACK_ENDPOINT_REQUIRED")
        self.endpoint = endpoint.rstrip("/")
        self.credentials = json.loads(credentials.read_text())
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def call(self, operation: str, args: dict):
        body = json.dumps(args, ensure_ascii=False).encode()
        if len(body) > MAX_BODY:
            raise HubError("BATCH_TOO_LARGE", 413)
        request = urllib.request.Request(self.endpoint + "/v1/" + operation, data=body,
                                        headers={"Content-Type": "application/json",
                                                 "Authorization": "Bearer " + self.credentials["token"]})
        try:
            with self.opener.open(request, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                code = json.load(error).get("error", "HTTP_ERROR")
            raise HubError(code, error.code) from None
        except urllib.error.URLError:
            raise HubError("HUB_UNAVAILABLE", 503) from None
