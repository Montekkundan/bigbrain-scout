"""Loopback-only Scout workspace. This is a local demo server, not deployment software."""

from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .core import ScoutError
from .workflow import Workspace, WorkflowError

STATIC = Path(__file__).parent / "web"
BODY_LIMIT = 16 * 1024
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; "
       "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
       "base-uri 'none'; frame-ancestors 'none'; form-action 'self'")


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, workspace):
        if address[0] != "127.0.0.1":
            raise ValueError("Scout serves only on 127.0.0.1")
        self.workspace = workspace
        self.csrf_token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "BigBrainScout/0.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        # Log the route/status, not request bodies or data provenance.
        super().log_message(format, *args)

    def respond(self, status, content, content_type="application/json", filename=None):
        if not isinstance(content, bytes):
            content = (json.dumps(content, allow_nan=False) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(content)

    def guarded(self, mutation=False):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.respond(403, {"error": "Use this workspace's loopback address", "code": "invalid_host"})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.respond(403, {"error": "Cross-site requests are refused", "code": "invalid_origin"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            self.respond(403, {"error": "Cross-origin requests are refused", "code": "invalid_origin"})
            return False
        if mutation and not secrets.compare_digest(self.headers.get("X-Scout-Token", ""), self.server.csrf_token):
            self.respond(403, {"error": "Missing workspace token; reload the page", "code": "invalid_token"})
            return False
        return True

    def error(self, exc):
        if isinstance(exc, WorkflowError):
            status = 409 if exc.code in {"busy", "expired_plan", "unknown_plan", "capsule_limit", "plan_limit"} else 400
            if exc.code == "unknown_capsule":
                status = 404
            self.respond(status, {"error": str(exc), "code": exc.code})
        elif isinstance(exc, ScoutError):
            self.respond(400, {"error": str(exc), "code": "refused"})
        elif isinstance(exc, (ImportError, OSError, RuntimeError, ValueError)):
            self.respond(400, {"error": str(exc), "code": "operation_failed"})
        else:
            self.respond(500, {"error": "Workspace operation failed. Check the local terminal.", "code": "internal_error"})
            print(f"Scout operation failed: {type(exc).__name__}: {exc}")

    def do_GET(self):
        if not self.guarded():
            return
        try:
            path = urlsplit(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in assets:
                name, mime = assets[path]
                self.respond(200, (STATIC / name).read_bytes(), mime)
            elif path == "/favicon.ico":
                self.respond(204, b"", "image/x-icon")
            elif path == "/api/config":
                self.respond(200, {**self.server.workspace.config(), "csrf_token": self.server.csrf_token})
            else:
                parts = path.strip("/").split("/")
                if len(parts) != 4 or parts[:2] != ["api", "capsules"]:
                    self.respond(404, {"error": "Route not found", "code": "not_found"})
                elif parts[3] == "download":
                    archive = self.server.workspace.archive(parts[2])
                    self.respond(200, archive, "application/zip", f"scout-{parts[2]}.zip")
                elif parts[3] == "manifest":
                    self.respond(200, self.server.workspace.capsule(parts[2]))
                else:
                    self.respond(404, {"error": "Route not found", "code": "not_found"})
        except Exception as exc:
            self.error(exc)

    def read_body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ScoutError("Transfer encoding is not supported")
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ScoutError("Send an application/json request")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ScoutError("Invalid request length") from exc
        if not 0 < length <= BODY_LIMIT:
            raise ScoutError("Request body must be between 1 and 16384 bytes")
        content = self.rfile.read(length)
        if len(content) != length:
            raise ScoutError("Incomplete request body")
        try:
            body = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError) as exc:
            raise ScoutError("Invalid JSON request") from exc
        if not isinstance(body, dict):
            raise ScoutError("Request must be a JSON object")
        return body

    def do_POST(self):
        if not self.guarded(mutation=True):
            return
        try:
            path = urlsplit(self.path).path
            body = self.read_body()
            if path == "/api/plan":
                result = self.server.workspace.plan(body)
            elif path == "/api/build" and set(body) == {"token"}:
                result = self.server.workspace.build(body["token"])
            elif path == "/api/verify" and set(body) == {"id"}:
                result = self.server.workspace.verify(body["id"])
            else:
                self.respond(400, {"error": "Unknown route or request fields", "code": "invalid_request"})
                return
            self.respond(200, result)
        except Exception as exc:
            self.error(exc)


def serve(output, port=8765, allow_live=False):
    workspace = Workspace(Path(output), allow_live=allow_live)
    with LocalServer(("127.0.0.1", port), workspace) as server:
        print(f"BigBrain Scout: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Capsules stay local: {Path(output).resolve()}", flush=True)
        print(f"Live fetch {'enabled for fixed section presets' if allow_live else 'disabled (offline demo)'}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Scout workspace stopped")
