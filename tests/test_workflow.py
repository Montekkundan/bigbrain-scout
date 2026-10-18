"""Dependency-free local workflow checks; live adapters are replaced by mocks."""

import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from bigbrain_scout import core
from bigbrain_scout.workflow import MAX_BUDGET_BYTES, WorkflowError, Workspace, read_pgm


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.workspace = Workspace(self.root / "workspace")
        self.request = self.workspace.config()["sources"][0]["defaults"]

    def build(self):
        response = self.workspace.plan(self.request)
        return self.workspace.build(response["token"])

    def test_synthetic_full_flow(self):
        response = self.build()
        self.assertEqual(len(response["id"]), 32)
        self.assertEqual(response["verification"]["fixture_type"], "synthetic")
        self.assertTrue(response["verification"]["verified"])
        self.assertEqual(response["preview"]["width"], 2)
        self.assertEqual(response["preview"]["height"], 3)
        self.assertEqual(response["plan"]["source_id"], "synthetic")
        self.assertEqual(response["manifest"]["source"]["id"], "scout-nonbiological-section-v1")
        self.assertEqual(base64.b64decode(response["preview"]["pixels_base64"]), bytes([17, 34, 85, 102, 153, 170]))
        self.assertEqual(self.workspace.verify(response["id"]), response["verification"])
        self.assertEqual(self.workspace.capsule(response["id"]), response["manifest"])

    def test_config_is_independent_and_offline(self):
        config = self.workspace.config()
        self.assertFalse(config["allow_live"])
        self.assertEqual(config["max_budget_bytes"], MAX_BUDGET_BYTES)
        self.assertEqual(config["sources"][0]["label"], "Synthetic demo")
        self.assertEqual(config["sources"][0]["scales"][0]["label"], "2 µm")
        self.assertEqual(config["sources"][0]["defaults"]["max_decoded_bytes"], MAX_BUDGET_BYTES)
        self.assertEqual(config["sources"][0]["defaults"]["max_estimated_working_bytes"], MAX_BUDGET_BYTES)
        config["sources"][0]["defaults"]["scale"] = "invalid"
        self.assertEqual(self.workspace.config()["sources"][0]["defaults"]["scale"], "synthetic-2um")

    def test_plan_request_and_result_are_snapshots(self):
        response = self.workspace.plan(self.request)
        self.request["roi_mm"][0][0] = 100
        response["plan"]["shape"][0] = 100
        built = self.workspace.build(response["token"])
        self.assertEqual(built["plan"]["shape"], [2, 3, 1])
        self.assertEqual(built["manifest"]["request"]["bbox_mm"][0][0], 0.002)

    def test_token_replay_and_unknown(self):
        planned = self.workspace.plan(self.request)
        self.workspace.build(planned["token"])
        for token in (planned["token"], "0" * 32, "../outside", None):
            with self.subTest(token=token), self.assertRaises(WorkflowError) as failure:
                self.workspace.build(token)
            self.assertEqual(failure.exception.code, "unknown_plan")

    def test_expired_plan(self):
        with patch("bigbrain_scout.workflow.time.monotonic", return_value=1):
            planned = self.workspace.plan(self.request)
        with patch("bigbrain_scout.workflow.time.monotonic", return_value=1801):
            with self.assertRaises(WorkflowError) as failure:
                self.workspace.build(planned["token"])
        self.assertEqual(failure.exception.code, "expired_plan")

    def test_plan_limit_and_expiry_pruning(self):
        with patch("bigbrain_scout.workflow.time.monotonic", return_value=1):
            for _ in range(32):
                self.workspace.plan(self.request)
            with self.assertRaises(WorkflowError) as failure:
                self.workspace.plan(self.request)
        self.assertEqual(failure.exception.code, "plan_limit")
        with patch("bigbrain_scout.workflow.time.monotonic", return_value=1802):
            self.assertIn("token", self.workspace.plan(self.request))

    def test_capsule_limit(self):
        self.workspace._capsule_count = 64
        planned = self.workspace.plan(self.request)
        with self.assertRaises(WorkflowError) as failure:
            self.workspace.build(planned["token"])
        self.assertEqual(failure.exception.code, "capsule_limit")

    def test_server_owned_source_only(self):
        for source in ("https://example.org/secret", "../outside", {"url": "https://example.org"}):
            with self.subTest(source=source):
                request = deepcopy(self.request)
                request["source_id"] = source
                with self.assertRaises(WorkflowError):
                    self.workspace.plan(request)
        request = deepcopy(self.request)
        request["source"] = {"url": "https://example.org"}
        with self.assertRaises(WorkflowError):
            self.workspace.plan(request)

    def test_invalid_budgets_never_call_live(self):
        self.workspace.allow_live = True
        for value in (0, -1, MAX_BUDGET_BYTES + 1, True, 1.0, float("nan")):
            with self.subTest(value=value), patch.object(self.workspace, "_live") as adapter:
                request = deepcopy(self.request)
                request.update(source_id="section-3905", max_decoded_bytes=value)
                with self.assertRaises(WorkflowError):
                    self.workspace.plan(request)
                adapter.assert_not_called()

    def test_invalid_coordinates_never_call_live(self):
        self.workspace.allow_live = True
        for value in (float("nan"), float("inf"), 1001, 10**400, True, "0"):
            with self.subTest(value=value), patch.object(self.workspace, "_live") as adapter:
                request = deepcopy(self.request)
                request["source_id"] = "section-3905"
                request["roi_mm"][0][0] = value
                with self.assertRaises(WorkflowError):
                    self.workspace.plan(request)
                adapter.assert_not_called()

    def test_live_disabled(self):
        request = deepcopy(self.request)
        request["source_id"] = "section-3905"
        with patch.object(self.workspace, "_live") as adapter:
            with self.assertRaises(WorkflowError) as failure:
                self.workspace.plan(request)
            self.assertEqual(failure.exception.code, "live_disabled")
            adapter.assert_not_called()

    def test_no_scale_fallback_or_budget_bypass(self):
        for changes in ({"scale": "missing"}, {"max_decoded_bytes": 1}, {"max_estimated_working_bytes": 1}):
            with self.subTest(changes=changes):
                request = deepcopy(self.request)
                request.update(changes)
                with self.assertRaises(WorkflowError):
                    self.workspace.plan(request)

    def test_tamper_blocks_verify_manifest_and_archive(self):
        built = self.build()
        capsule = self.workspace.root / "capsules" / built["id"]
        (capsule / "data.raw").write_bytes(b"tamper")
        for action in (self.workspace.verify, self.workspace.capsule, self.workspace.archive):
            with self.subTest(action=action.__name__), self.assertRaises(WorkflowError) as failure:
                action(built["id"])
            self.assertEqual(failure.exception.code, "verification_failed")

    def test_archive_contains_only_recorded_files(self):
        built = self.build()
        capsule = self.workspace.root / "capsules" / built["id"]
        (capsule / "private-note.txt").write_text("not exported")
        with zipfile.ZipFile(io.BytesIO(self.workspace.archive(built["id"]))) as archive:
            self.assertEqual(set(archive.namelist()), {"manifest.json", "data.raw", "preview.pgm", "fixture.json", "reproduce.py"})
            self.assertEqual(archive.read("data.raw"), bytes([17, 34, 85, 102, 153, 170]))
            self.assertEqual(json.loads(archive.read("manifest.json")), built["manifest"])

    def test_bad_capsule_ids(self):
        for identifier in ("../outside", "/etc/passwd", "A" * 32, "0" * 32, None):
            with self.subTest(identifier=identifier), self.assertRaises(WorkflowError) as failure:
                self.workspace.capsule(identifier)
            self.assertEqual(failure.exception.code, "unknown_capsule")

    def test_unsafe_recorded_path_rejected_before_verify(self):
        built = self.build()
        capsule = self.workspace.root / "capsules" / built["id"]
        manifest = built["manifest"]
        manifest["artifacts"]["data"]["path"] = "../outside.raw"
        (capsule / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(WorkflowError) as failure:
            self.workspace.archive(built["id"])
        self.assertEqual(failure.exception.code, "unsafe_path")

    def test_symlink_artifact_refused(self):
        built = self.build()
        capsule = self.workspace.root / "capsules" / built["id"]
        path = capsule / "data.raw"
        outside = self.root / "outside.raw"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaises(WorkflowError) as failure:
            self.workspace.verify(built["id"])
        self.assertEqual(failure.exception.code, "unsafe_path")

    def test_symlink_workspace_refused(self):
        alias = self.root / "workspace-alias"
        alias.symlink_to(self.workspace.root, target_is_directory=True)
        with self.assertRaises(WorkflowError) as failure:
            Workspace(alias)
        self.assertEqual(failure.exception.code, "unsafe_path")

    def test_busy_prevents_concurrent_build(self):
        planned = self.workspace.plan(self.request)
        started, release = threading.Event(), threading.Event()
        actual = core.create_demo
        results = []

        def blocked(*args):
            started.set()
            self.assertTrue(release.wait(5))
            return actual(*args)

        def background():
            try:
                results.append(self.workspace.build(planned["token"]))
            except Exception as error:
                results.append(error)

        with patch("bigbrain_scout.workflow.core.create_demo", side_effect=blocked):
            thread = threading.Thread(target=background)
            thread.start()
            self.assertTrue(started.wait(5))
            try:
                with self.assertRaises(WorkflowError) as failure:
                    self.workspace.build(planned["token"])
                self.assertEqual(failure.exception.code, "busy")
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], dict)

    def test_failed_build_consumes_token(self):
        planned = self.workspace.plan(self.request)
        with patch("bigbrain_scout.workflow.core.create_demo", side_effect=core.ScoutError("failure")):
            with self.assertRaises(WorkflowError):
                self.workspace.build(planned["token"])
        with self.assertRaises(WorkflowError) as failure:
            self.workspace.build(planned["token"])
        self.assertEqual(failure.exception.code, "unknown_plan")

    def test_live_interface_with_isolated_mock(self):
        self.workspace.allow_live = True
        request = deepcopy(self.request)
        request["source_id"] = "section-3905"
        prepared = core.plan({**{key: value for key, value in self.request.items() if key != "source_id"},
                              "source": self.workspace._fixture["source"]})
        prepared["source_id"] = "section-3905"
        calls = []

        def prepare(canonical, cache):
            calls.append(("prepare", canonical["source_id"], Path(cache)))
            return {"request": canonical, "immutable_marker": [1, 2]}, deepcopy(prepared)

        def fetch(context, result, destination):
            calls.append(("fetch", context["immutable_marker"], result["source_id"]))
            with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                path = Path(temporary) / "request.json"
                augmented = {key: value for key, value in self.request.items() if key != "source_id"}
                augmented["source"] = self.workspace._fixture["source"]
                path.write_text(json.dumps(augmented))
                fixture = Path(temporary) / "fixture.json"
                fixture.write_text(json.dumps(self.workspace._fixture))
                manifest = core.create_demo(path, fixture, destination)
            manifest["fixture_type"] = "real_remote_data"
            (Path(destination) / "manifest.json").write_text(json.dumps(manifest))
            return manifest

        def verify(directory):
            calls.append(("verify", Path(directory)))
            return core.load_json(Path(directory) / "manifest.json")

        adapter = SimpleNamespace(defaults=lambda identifier: {**deepcopy(request), "source_id": identifier}, SCALES={"synthetic-2um": None},
                                  prepare=prepare, fetch=fetch, verify=verify)
        with patch.object(self.workspace, "_live", return_value=adapter):
            config = self.workspace.config()
            self.assertEqual(config["sources"][1]["id"], "section-3905")
            planned = self.workspace.plan(request)
            built = self.workspace.build(planned["token"])
            self.assertEqual(built["verification"]["fixture_type"], "real_remote_data")
            self.assertEqual([call[0] for call in calls], ["prepare", "fetch", "verify"])

    def test_all_catalog_presets_are_exposed_and_checked(self):
        from bigbrain_scout import live
        from bigbrain_scout.sources import source_ids
        self.workspace.allow_live = True
        config = self.workspace.config()
        self.assertEqual([item["id"] for item in config["sources"]], ["synthetic", *source_ids()])
        for source in config["sources"][1:]:
            with self.subTest(source=source["id"]):
                self.assertEqual(source["defaults"], live.defaults(source["id"]))
                self.assertEqual(len(source["scales"]), 8)
                with patch.object(live, "prepare", return_value=({}, {"source_id": source["id"]})) as prepare:
                    planned = self.workspace.plan(source["defaults"])
                    self.assertEqual(planned["plan"]["source_id"], source["id"])
                    prepare.assert_called_once()

    def test_all_live_presets_require_opt_in(self):
        from bigbrain_scout import live
        from bigbrain_scout.sources import source_ids
        for identifier in source_ids():
            with self.subTest(source=identifier), patch.object(live, "prepare") as prepare:
                with self.assertRaises(WorkflowError) as failure:
                    self.workspace.plan(live.defaults(identifier))
                self.assertEqual(failure.exception.code, "live_disabled")
                prepare.assert_not_called()

    def test_bad_manifest_json_is_workflow_error(self):
        built = self.build()
        path = self.workspace.root / "capsules" / built["id"] / "manifest.json"
        path.write_text("{")
        with self.assertRaises(WorkflowError) as failure:
            self.workspace.verify(built["id"])
        self.assertEqual(failure.exception.code, "verification_failed")

    def test_unhashable_manifest_fields_refused(self):
        built = self.build()
        path = self.workspace.root / "capsules" / built["id"] / "manifest.json"
        for field in ("path", "fixture_type"):
            with self.subTest(field=field):
                manifest = deepcopy(built["manifest"])
                if field == "path":
                    manifest["artifacts"]["data"]["path"] = []
                else:
                    manifest["fixture_type"] = []
                path.write_text(json.dumps(manifest))
                with self.assertRaises(WorkflowError):
                    self.workspace.verify(built["id"])


class PreviewTests(unittest.TestCase):
    def test_actual_pixels_whitespace_and_hash_are_preserved(self):
        pixels = bytes([10, 32, 35, 255])
        preview = read_pgm(b"P5\n# comment\n2 2\n255\n" + pixels)
        self.assertEqual(base64.b64decode(preview["pixels_base64"]), pixels)

    def test_crlf_header(self):
        preview = read_pgm(b"P5\r\n1 1\r\n255\r\n\x80")
        self.assertEqual(base64.b64decode(preview["pixels_base64"]), b"\x80")

    def test_malformed_and_oversized_preview(self):
        for data in (b"", b"P2\n1 1\n255\n\x00", b"P5\n-1 1\n255\n", b"P5\n1 1\n65535\n\x00", b"P5\n1 1\n255\n", b"P5\n99999999 1\n255\n", b"P5\n# unterminated"):
            with self.subTest(data=data), self.assertRaises(WorkflowError):
                read_pgm(data)


if __name__ == "__main__":
    unittest.main()
