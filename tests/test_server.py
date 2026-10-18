"""Real loopback HTTP checks, with offline data only."""

import http.client
import json
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from bigbrain_scout.core import ScoutError
from bigbrain_scout.server import BODY_LIMIT, LocalServer
from bigbrain_scout.workflow import Workspace


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Workspace(Path(self.temporary.name).resolve() / "workspace")
        self.server = LocalServer(("127.0.0.1", 0), self.workspace)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def call(self, method, route, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        self.addCleanup(connection.close)
        default = {"X-Scout-Token": self.server.csrf_token, "Content-Type": "application/json"}
        default.update(headers or {})
        content = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        connection.request(method, route, content, default)
        response = connection.getresponse()
        payload = response.read()
        return response.status, dict(response.getheaders()), payload

    def test_static_assets_and_security_headers(self):
        for route, expected in (("/", b"Small regions."), ("/app.js", b"csrf_token"), ("/style.css", b"--teal")):
            status, headers, payload = self.call("GET", route)
            self.assertEqual(status, 200)
            self.assertIn(expected, payload)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(headers["Cache-Control"], "no-store")

    def test_full_offline_http_flow(self):
        status, _, body = self.call("GET", "/api/config")
        config = json.loads(body)
        self.assertEqual(status, 200)
        self.assertFalse(config["allow_live"])
        request = next(item["defaults"] for item in config["sources"] if item["id"] == "synthetic")
        status, _, body = self.call("POST", "/api/plan", request)
        self.assertEqual(status, 200, body)
        planned = json.loads(body)
        self.assertEqual(planned["plan"]["shape"], [2, 3, 1])
        status, _, body = self.call("POST", "/api/build", {"token": planned["token"]})
        self.assertEqual(status, 200, body)
        built = json.loads(body)
        self.assertEqual(built["manifest"]["fixture_type"], "synthetic")
        status, _, body = self.call("POST", "/api/verify", {"id": built["id"]})
        self.assertEqual(status, 200, body)
        self.assertTrue(json.loads(body)["verified"])
        status, headers, body = self.call("GET", f'/api/capsules/{built["id"]}/download')
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertTrue(body.startswith(b"PK"))
        status, _, body = self.call("GET", f'/api/capsules/{built["id"]}/manifest')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), built["manifest"])

    def test_wrong_host_and_cross_site_reads_refused(self):
        for headers in ({"Host": "evil.example"}, {"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
            self.assertEqual(self.call("GET", "/api/config", headers=headers)[0], 403)

    def test_cross_site_write_and_wrong_token_refused(self):
        for headers in ({"Origin": "http://evil.example"}, {"X-Scout-Token": "wrong"}, {"X-Scout-Token": ""}):
            self.assertEqual(self.call("POST", "/api/plan", {}, headers)[0], 403)

    def test_own_origin_accepted(self):
        headers = {"Origin": f"http://127.0.0.1:{self.server.server_port}"}
        self.assertEqual(self.call("GET", "/api/config", headers=headers)[0], 200)

    def test_body_and_field_validation(self):
        for body in (b"null", b"[]", b"not-json", b'{"x":NaN}', b"x" * (BODY_LIMIT + 1)):
            self.assertEqual(self.call("POST", "/api/plan", body)[0], 400)
        self.assertEqual(self.call("POST", "/api/plan", {}, {"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.call("POST", "/api/build", {"token": "x", "output": "/tmp/arbitrary"})[0], 400)
        self.assertIn(self.call("POST", "/api/build", {"token": []})[0], (400, 409))

    def test_unknown_routes_and_paths_refused(self):
        for route in ("/../README.md", "/.git/config", "/api/capsules/../../download", "/api/capsules/a/file"):
            self.assertEqual(self.call("GET", route)[0], 404)
        self.assertIn(self.call("GET", "/api/capsules/not-an-id/download")[0], (400, 404))

    def test_unsafe_request_is_refused_without_capsule(self):
        config = self.workspace.config()
        request = next(item["defaults"] for item in config["sources"] if item["id"] == "synthetic")
        request["max_decoded_bytes"] = 5
        status, _, body = self.call("POST", "/api/plan", request)
        self.assertEqual(status, 400)
        self.assertIn(b"Decoded ROI", body)
        self.assertFalse(any(self.workspace.root.glob("capsules/*/manifest.json")))

    def test_server_cannot_bind_other_interfaces(self):
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            LocalServer(("0.0.0.0", 0), self.workspace)

    def build_default(self):
        request = self.workspace.config()["sources"][0]["defaults"]
        status, _, body = self.call("POST", "/api/plan", request)
        self.assertEqual(status, 200, body)
        token = json.loads(body)["token"]
        status, _, body = self.call("POST", "/api/build", {"token": token})
        self.assertEqual(status, 200, body)
        return token, json.loads(body)

    def test_zip_extraction_and_trusted_replay_end_to_end(self):
        from bigbrain_scout.replay import compare_capsules, reproduce, verify_capsule
        _, built = self.build_default()
        directory = self.workspace.root / "capsules" / built["id"]
        (directory / "private-note.txt").write_text("must not be exported")
        status, _, payload = self.call("GET", f'/api/capsules/{built["id"]}/download')
        self.assertEqual(status, 200)
        extracted = self.workspace.root.parent / "extracted"
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(set(archive.namelist()), {"manifest.json", "data.raw", "preview.pgm", "fixture.json", "reproduce.py"})
            archive.extractall(extracted)
        self.assertEqual(verify_capsule(extracted), built["manifest"])
        replayed = extracted.parent / "replayed"
        result = reproduce(extracted, replayed)
        self.assertTrue(result["comparison"]["matched"])
        self.assertTrue(compare_capsules(extracted, replayed)["data_bytes_equal"])
        self.assertFalse((extracted / "private-note.txt").exists())

    def test_tampering_refuses_verify_manifest_and_export(self):
        _, built = self.build_default()
        directory = self.workspace.root / "capsules" / built["id"]
        (directory / "data.raw").write_bytes(b"tamper")
        for method, route, body in (
                ("POST", "/api/verify", {"id": built["id"]}),
                ("GET", f'/api/capsules/{built["id"]}/manifest', None),
                ("GET", f'/api/capsules/{built["id"]}/download', None)):
            with self.subTest(route=route):
                status, _, payload = self.call(method, route, body)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(payload)["code"], "verification_failed")

    def test_failed_collection_requires_fresh_plan_then_recovers(self):
        request = self.workspace.config()["sources"][0]["defaults"]
        _, _, body = self.call("POST", "/api/plan", request)
        token = json.loads(body)["token"]
        with patch("bigbrain_scout.workflow.core.create_demo", side_effect=ScoutError("simulated collection failure")):
            status, _, body = self.call("POST", "/api/build", {"token": token})
            self.assertEqual(status, 400)
            self.assertIn(b"simulated collection failure", body)
        self.assertEqual(self.call("POST", "/api/build", {"token": token})[0], 409)
        _, built = self.build_default()
        self.assertTrue(built["verification"]["verified"])

    def test_restart_preserves_capsules_but_not_pending_plans(self):
        _, built = self.build_default()
        request = self.workspace.config()["sources"][0]["defaults"]
        _, _, body = self.call("POST", "/api/plan", request)
        old_token = json.loads(body)["token"]
        self.server.workspace = Workspace(self.workspace.root)
        self.assertEqual(self.call("POST", "/api/build", {"token": old_token})[0], 409)
        status, _, body = self.call("POST", "/api/verify", {"id": built["id"]})
        self.assertEqual(status, 200, body)
        self.assertTrue(json.loads(body)["verified"])


if __name__ == "__main__":
    unittest.main()
