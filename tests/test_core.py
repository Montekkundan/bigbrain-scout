"""Offline safety-contract tests; these do not validate any BigBrain service."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bigbrain_scout.core import ScoutError, create_demo, load_json, plan, verify_capsule

ROOT = Path(__file__).resolve().parent.parent
REQUEST = ROOT / "examples" / "safe-request.json"
FIXTURE = ROOT / "fixtures" / "synthetic-section.json"


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.request = load_json(REQUEST)

    def test_exact_request_geometry_and_estimates(self):
        result = plan(self.request)
        self.assertEqual(result["shape"], [2, 3, 1])
        self.assertEqual(result["indices"], {"start": [1, 0, 0], "stop": [3, 3, 1]})
        self.assertEqual(result["estimates"]["decoded_roi_bytes"], 6)
        self.assertEqual(result["estimates"]["chunk_mosaic_bytes"], 16)
        self.assertEqual(result["estimates"]["estimated_working_bytes"], 26)
        self.assertIsNone(result["estimates"]["network_bytes"])
        self.assertFalse(result["estimates"]["peak_memory_guarantee"])

    def test_no_scale_fallback(self):
        self.request["scale"] = "1um"
        with self.assertRaisesRegex(ScoutError, "no automatic fallback"):
            plan(self.request)

    def test_unknown_units_refused(self):
        self.request["source"]["units"] = "microns"
        with self.assertRaisesRegex(ScoutError, "millimetre"):
            plan(self.request)

    def test_ambiguous_scale_refused(self):
        self.request["source"]["scales"] *= 2
        with self.assertRaises(ScoutError):
            plan(self.request)

    def test_decoded_budget(self):
        self.request["max_decoded_bytes"] = 5
        with self.assertRaisesRegex(ScoutError, "Decoded ROI"):
            plan(self.request)

    def test_working_budget(self):
        self.request["max_estimated_working_bytes"] = 25
        with self.assertRaisesRegex(ScoutError, "Chunk assembly"):
            plan(self.request)

    def test_budget_boundary_accepted(self):
        self.request.update(max_decoded_bytes=6, max_estimated_working_bytes=26)
        self.assertEqual(plan(self.request)["shape"], [2, 3, 1])

    def test_invalid_budget_values(self):
        for value in (0, -1, True, 1.5, float("nan")):
            with self.subTest(value=value):
                request = copy.deepcopy(self.request)
                request["max_decoded_bytes"] = value
                with self.assertRaises(ScoutError):
                    plan(request)

    def test_nonfinite_roi(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.request["roi_mm"][0][0] = value
                with self.assertRaises(ScoutError):
                    plan(self.request)

    def test_negative_resolution(self):
        self.request["source"]["scales"][0]["resolution_mm"][0] = -1
        with self.assertRaises(ScoutError):
            plan(self.request)

    def test_outside_roi(self):
        self.request["roi_mm"] = [[1, 1, 1], [2, 2, 2]]
        with self.assertRaisesRegex(ScoutError, "does not intersect"):
            plan(self.request)

    def test_clips_to_source(self):
        self.request["roi_mm"] = [[-1, 0, 0], [0.004, 0.004, 0.002]]
        result = plan(self.request)
        self.assertTrue(result["clipped"])
        self.assertEqual(result["clipped_bbox_mm"], [[0, 0, 0], [0.004, 0.004, 0.002]])

    def test_subvoxel_roi_snaps_outward(self):
        self.request["roi_mm"] = [[0.0021, 0.0001, 0.0001], [0.0059, 0.0059, 0.0019]]
        result = plan(self.request)
        self.assertEqual(result["shape"], [2, 3, 1])
        self.assertNotEqual(result["requested_bbox_mm"], result["returned_bbox_mm"])

    def test_degenerate_roi(self):
        self.request["roi_mm"][1][0] = self.request["roi_mm"][0][0]
        with self.assertRaises(ScoutError):
            plan(self.request)

    def test_chunk_shape_integer_and_positive(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                self.request["source"]["scales"][0]["chunk_shape"][0] = value
                with self.assertRaises(ScoutError):
                    plan(self.request)

    def test_nonintegral_source_grid(self):
        self.request["source"]["bbox_mm"][1][0] = 0.007
        with self.assertRaisesRegex(ScoutError, "integral voxel grid"):
            plan(self.request)

    def test_malformed_dtype_refused(self):
        self.request["source"]["scales"][0]["dtype"] = []
        with self.assertRaisesRegex(ScoutError, "Unsupported dtype"):
            plan(self.request)

    def test_overflow_source_extent_refused(self):
        self.request["source"]["bbox_mm"] = [[-1e308, 0, 0], [1e308, 0.008, 0.002]]
        with self.assertRaisesRegex(ScoutError, "finite"):
            plan(self.request)

    def test_unsafe_example_cli_refuses(self):
        result = subprocess.run([sys.executable, "-m", "bigbrain_scout", "plan", "--request", "examples/unsafe-request.json"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Decoded ROI estimate 6 exceeds", result.stderr)
        self.assertEqual(result.stdout, "")


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "capsule"
        self.manifest = create_demo(REQUEST, FIXTURE, self.output)

    def update_manifest(self):
        (self.output / "manifest.json").write_text(json.dumps(self.manifest))

    def test_generated_capsule(self):
        manifest = verify_capsule(self.output)
        self.assertEqual(manifest["fixture_type"], "synthetic")
        self.assertEqual(manifest["returned"]["shape"], [2, 3, 1])
        self.assertEqual((self.output / "data.raw").read_bytes(), bytes([17, 34, 85, 102, 153, 170]))
        self.assertEqual(manifest["returned"]["affine"][0], [0.002, 0, 0, 0.002])

    def test_deterministic_data(self):
        second = create_demo(REQUEST, FIXTURE, self.root / "second")
        self.assertEqual(second["artifacts"], self.manifest["artifacts"])

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ScoutError, "refusing overwrite"):
            create_demo(REQUEST, FIXTURE, self.output)

    def test_tampered_hash(self):
        (self.output / "data.raw").write_bytes(b"tamper")
        with self.assertRaisesRegex(ScoutError, "checksum/size mismatch"):
            verify_capsule(self.output)

    def test_missing_required_field(self):
        del self.manifest["source"]["license"]
        self.update_manifest()
        with self.assertRaisesRegex(ScoutError, "missing required fields"):
            verify_capsule(self.output)

    def test_traversal_refused(self):
        for value in ("../outside", "/etc/passwd", "..\\outside"):
            with self.subTest(value=value):
                self.manifest["artifacts"]["data"]["path"] = value
                self.update_manifest()
                with self.assertRaises(ScoutError):
                    verify_capsule(self.output)

    def test_symlink_escape_refused(self):
        outside = self.root / "outside.raw"
        outside.write_bytes((self.output / "data.raw").read_bytes())
        (self.output / "escape.raw").symlink_to(outside)
        self.manifest["artifacts"]["data"]["path"] = "escape.raw"
        self.update_manifest()
        with self.assertRaisesRegex(ScoutError, "escapes"):
            verify_capsule(self.output)

    def test_bad_geometry_refused(self):
        self.manifest["returned"]["affine"][0][0] = 2
        self.update_manifest()
        with self.assertRaisesRegex(ScoutError, "inconsistent"):
            verify_capsule(self.output)

    def test_shape_byte_count_refused(self):
        self.manifest["returned"]["shape"][0] = 3
        self.manifest["returned"]["bbox_mm"][1][0] = 0.008
        self.manifest["returned"]["indices"]["stop"][0] = 4
        self.update_manifest()
        with self.assertRaisesRegex(ScoutError, "byte count"):
            verify_capsule(self.output)

    def test_bad_indices_refused(self):
        self.manifest["returned"]["indices"]["start"][0] = -1
        self.update_manifest()
        with self.assertRaisesRegex(ScoutError, "nonnegative"):
            verify_capsule(self.output)

    def test_live_claim_refused(self):
        self.manifest["provenance"]["offline_only"] = False
        self.update_manifest()
        with self.assertRaisesRegex(ScoutError, "offline-only"):
            verify_capsule(self.output)

    def test_offline_reproduction(self):
        replay = self.root / "replayed.raw"
        result = subprocess.run([sys.executable, str(self.output / "reproduce.py"), "--output", str(replay)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(hashlib.sha256(replay.read_bytes()).hexdigest(), self.manifest["artifacts"]["data"]["sha256"])
        self.assertIn("no network", result.stdout)
        repeated = subprocess.run([sys.executable, str(self.output / "reproduce.py"), "--output", str(replay)], capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)

    def test_fixture_metadata_mismatch(self):
        fixture = load_json(FIXTURE)
        fixture["source"]["id"] = "other"
        altered = self.root / "altered.json"
        altered.write_text(json.dumps(fixture))
        with self.assertRaisesRegex(ScoutError, "does not match"):
            create_demo(REQUEST, altered, self.root / "bad")

    def test_schema_required_fields_match(self):
        schema = load_json(ROOT / "schemas" / "manifest.schema.json")
        self.assertTrue(set(schema["required"]).issubset(self.manifest))
        for field in ("source", "request", "returned", "provenance"):
            self.assertTrue(set(schema["properties"][field]["required"]).issubset(self.manifest[field]))


if __name__ == "__main__":
    unittest.main()
