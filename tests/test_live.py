"""Offline live-adapter contract tests using captured-style metadata and mocks."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bigbrain_scout import live
from bigbrain_scout.core import ScoutError

try:
    import numpy as np
    import nibabel as nib
except ImportError:
    np = nib = None


class RequestTests(unittest.TestCase):
    def test_defaults_are_independent(self):
        first = live.defaults()
        first["roi_mm"][0][0] = 100
        self.assertEqual(live.defaults()["roi_mm"][0][0], 7.6)

    def test_invalid_requests_stop_before_imports_or_network(self):
        cases = []
        for key, value in (("source_id", "other-source"), ("source_id", None),
                           ("source_id", []), ("source_id", True), ("scale", "3um"),
                           ("scale", []), ("max_decoded_bytes", live.LIMIT + 1),
                           ("max_decoded_bytes", True), ("max_decoded_bytes", 0),
                           ("max_estimated_working_bytes", 1.5)):
            request = live.defaults()
            request[key] = value
            cases.append(request)
        for value in (float("nan"), float("inf"), True, "7.6"):
            request = live.defaults()
            request["roi_mm"][0][0] = value
            cases.append(request)
        request = live.defaults()
        request["roi_mm"] = [[-1e308, 8, 7.6], [1e308, 10, 8]]
        cases.append(request)
        request = live.defaults()
        request["roi_mm"][1][0] = request["roi_mm"][0][0]
        cases.append(request)
        request = live.defaults()
        request["source_url"] = "https://example.org/"
        cases.append(request)
        for request in cases:
            with self.subTest(request=request), patch.object(live, "_runtime") as runtime:
                with self.assertRaises(ScoutError):
                    live.prepare(request, "/not-used")
                runtime.assert_not_called()


@unittest.skipIf(np is None or nib is None, "existing live extra is optional")
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.info = {"data_type": "uint8", "num_channels": 1, "scales": [
            {"key": "1,1,1", "resolution": [1000, 20000, 1000], "size": [120900, 1, 139120],
             "voxel_offset": [0, 0, 0], "chunk_sizes": [[256, 1, 256]], "encoding": "raw"},
            {"key": "4,1,4", "resolution": [4000, 20000, 4000], "size": [30225, 1, 34780],
             "voxel_offset": [0, 0, 0], "chunk_sizes": [[256, 1, 256]], "encoding": "raw"}]}
        self.transform = np.asarray([[0, 0, 1, -70677183.3333333],
                                     [0, 1, 0, 8069999.999999998],
                                     [1, 0, 0, -58788283.3333333], [0, 0, 0, 1]], dtype=float)
        self.scales = []
        for item in self.info["scales"]:
            affine = self.transform @ np.diag([*item["resolution"], 1.0])
            affine[:3, 3] += 500
            affine[:3] /= 1e6
            self.scales.append(SimpleNamespace(key=item["key"], res_mm=np.asarray(item["resolution"]) / 1e6,
                                               affine=affine))
        self.finest_bbox = live._image_bbox(np, self.scales[0].affine, self.info["scales"][0]["size"])
        self.section = SimpleNamespace(name="#3905: mocked section", id="mocked-section",
                                       providers={"neuroglancer/precomputed": live.SOURCE_URL})
        self.fetch_calls = []

        def boundingbox(*, clip):
            self.assertFalse(clip)
            return SimpleNamespace(minpoint=self.finest_bbox[0], maxpoint=self.finest_bbox[1])

        def fetch(**kwargs):
            self.fetch_calls.append(kwargs)
            return self.image

        self.section.get_boundingbox = boundingbox
        self.section.fetch = fetch
        self.volume = SimpleNamespace(scales=self.scales, _info=self.info, transform_nm=self.transform)
        self.siibra = SimpleNamespace(
            locations=SimpleNamespace(BoundingBox=lambda lower, upper, space: (lower, upper, space)),
            features=SimpleNamespace(get=lambda *args: [self.section]))
        self.runtime = patch.object(live, "_runtime", return_value=(np, self.siibra, lambda url: self.volume))
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def prepare(self, request=None):
        return live.prepare(request or live.defaults(), self.root / "cache")

    def select_source(self, source_id):
        """Use the three officially observed plane positions, without network."""
        source = live.catalog(source_id)
        self.section.name = f"#{source['section_number']}: mocked section"
        self.section.providers["neuroglancer/precomputed"] = source["source_url"]
        self.transform[1, 3] = {"section-3905": 8070000.0, "section-3954": 9050000.0,
                                "section-4003": 10030000.0}[source_id]
        for scale, item in zip(self.scales, self.info["scales"]):
            affine = self.transform @ np.diag([*item["resolution"], 1.0])
            affine[:3, 3] += 500
            affine[:3] /= 1e6
            scale.affine = affine
        self.finest_bbox = live._image_bbox(np, self.scales[0].affine, self.info["scales"][0]["size"])

    def make_capsule(self, source_id=live.SOURCE_ID):
        context, plan = self.prepare(live.defaults(source_id))
        data = np.arange(np.prod(plan["shape"]), dtype=np.uint8).reshape(plan["shape"])
        self.image = nib.Nifti1Image(data, np.asarray(plan["returned_affine_mm"]))
        output = self.root / source_id
        manifest = live.fetch(context, plan, output)
        return context, plan, output, manifest

    def replace_manifest(self, output, manifest):
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def refresh_artifact(self, output, manifest, key):
        path = output / manifest["artifacts"][key]["path"]
        content = path.read_bytes()
        manifest["artifacts"][key].update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.replace_manifest(output, manifest)

    def test_singleton_spatial_axes_preserve_two_dimensional_preview(self):
        for index, axes in enumerate(((0,), (2,), (0, 2))):
            request = live.defaults()
            for axis in axes:
                request["roi_mm"][1][axis] = request["roi_mm"][0][axis] + 0.0001
            context, plan = self.prepare(request)
            self.assertEqual(plan["shape"][1], 1)
            self.assertIn(1, (plan["shape"][0], plan["shape"][2]))
            data = np.arange(np.prod(plan["shape"]), dtype=np.uint8).reshape(plan["shape"])
            self.image = nib.Nifti1Image(data, np.asarray(plan["returned_affine_mm"]))
            output = self.root / f"narrow-{index}"
            manifest = live.fetch(context, plan, output)
            self.assertEqual(live.verify(output), manifest)
            plane = data[:, 0, :]
            expected = f"P5\n{plane.shape[1]} {plane.shape[0]}\n255\n".encode() + plane.tobytes()
            self.assertEqual((output / "preview.pgm").read_bytes(), expected)

    def test_every_catalog_section_roundtrips_offline(self):
        for source_id in live.source_ids():
            with self.subTest(source_id=source_id):
                self.select_source(source_id)
                context, plan, output, manifest = self.make_capsule(source_id)
                self.assertEqual(plan["source_id"], source_id)
                self.assertEqual(manifest["source"]["id"], source_id)
                self.assertEqual(manifest["source"]["url"], live.catalog(source_id)["source_url"])
                self.assertEqual(manifest["artifacts"]["data"]["path"], f"{source_id}.nii")
                self.assertEqual(live.verify(output), manifest)
                self.assertLess(plan["estimates"]["estimated_working_bytes"], live.LIMIT)
                self.assertEqual(context["request"], live.defaults(source_id))
                instructions = (output / "README.md").read_text()
                self.assertIn("reproduce --capsule", instructions)
                self.assertIn("--allow-live", instructions)
                self.assertNotIn("from bigbrain_scout.live", instructions)

    def test_every_catalog_section_refuses_provider_drift_before_fetch(self):
        for source_id in live.source_ids():
            with self.subTest(source_id=source_id):
                self.select_source(source_id)
                context, plan = self.prepare(live.defaults(source_id))
                self.section.providers["neuroglancer/precomputed"] = "https://example.org/"
                with self.assertRaisesRegex(ScoutError, "source changed"):
                    live.fetch(context, plan, self.root / source_id)
                self.assertEqual(self.fetch_calls, [])

    def test_every_catalog_section_refuses_metadata_and_geometry_drift(self):
        for source_id in live.source_ids():
            with self.subTest(source_id=source_id):
                self.select_source(source_id)
                context, plan = self.prepare(live.defaults(source_id))
                for key, value in (("source_url", live.catalog("section-4003" if source_id != "section-4003"
                                                               else "section-3905")["source_url"]),
                                   ("configuration_url", "https://example.org/config.json"),
                                   ("feature_name", "#9999: different section")):
                    altered = copy.deepcopy(context)
                    altered["metadata"][key] = value
                    with self.assertRaises(ScoutError):
                        live.fetch(altered, plan, self.root / f"{source_id}-{key}")
                affine = np.asarray(plan["returned_affine_mm"])
                affine[1, 3] += 0.001
                self.image = nib.Nifti1Image(np.zeros(plan["shape"], dtype=np.uint8), affine)
                with self.assertRaisesRegex(ScoutError, "affine differs"):
                    live.fetch(context, plan, self.root / f"{source_id}-affine")

    def test_cross_section_artifact_names_are_refused(self):
        for source_id in live.source_ids():
            with self.subTest(source_id=source_id):
                self.select_source(source_id)
                _, _, output, manifest = self.make_capsule(source_id)
                alternate = "section-3905" if source_id != "section-3905" else "section-3954"
                manifest["artifacts"]["data"]["path"] = live.artifact_names(alternate)["data"]
                self.replace_manifest(output, manifest)
                with self.assertRaisesRegex(ScoutError, "unsafe"):
                    live.verify(output)

    def test_unknown_and_cross_section_manifest_sources_are_refused(self):
        for source_id in live.source_ids():
            with self.subTest(source_id=source_id):
                self.select_source(source_id)
                _, _, output, original = self.make_capsule(source_id)
                for identity in (None, [], "section-9999", "section-3905" if source_id != "section-3905"
                                 else "section-3954"):
                    manifest = copy.deepcopy(original)
                    manifest["source"]["id"] = identity
                    self.replace_manifest(output, manifest)
                    with self.assertRaisesRegex(ScoutError, "whitelisted"):
                        live.verify(output)

    def test_metadata_only_prepare_and_transform_aware_plan(self):
        context, plan = self.prepare()
        self.assertEqual(self.fetch_calls, [])
        self.assertEqual(plan["shape"], plan["shape_provider_xyz"][::-1])
        self.assertEqual(plan["scale_key"], "4um")
        self.assertEqual(plan["source_scale_key"], "4,1,4")
        self.assertEqual(plan["dtype"], "uint8")
        self.assertTrue(plan["clipped"])
        self.assertEqual(plan["estimates"]["decoded_roi_bytes"], np.prod(plan["shape"]))
        self.assertEqual(plan["estimates"]["estimated_working_bytes"], sum(
            plan["estimates"][key] for key in ("decoded_roi_bytes", "chunk_mosaic_bytes", "one_chunk_bytes")))
        self.assertIsNone(plan["estimates"]["network_bytes"])
        self.assertFalse(plan["estimates"]["peak_memory_guarantee"])
        self.assertEqual(context["request"], live.defaults())

    def test_asymmetric_shape_preserves_array_axis_reversal(self):
        request = live.defaults()
        request["roi_mm"][1][0] = 8.2
        _, plan = self.prepare(request)
        self.assertNotEqual(plan["shape"][0], plan["shape_provider_xyz"][0])
        self.assertEqual(plan["shape"], plan["shape_provider_xyz"][::-1])
        self.assertNotEqual(plan["returned_bbox_mm"], plan["clipped_bbox_mm"])

    def test_roi_budget_refuses_before_image_fetch(self):
        request = live.defaults()
        request["max_decoded_bytes"] = 10
        with self.assertRaisesRegex(ScoutError, "Decoded ROI"):
            self.prepare(request)
        self.assertEqual(self.fetch_calls, [])

    def test_working_budget_includes_roi_mosaic_and_one_chunk(self):
        _, plan = self.prepare()
        request = live.defaults()
        request["max_estimated_working_bytes"] = plan["estimates"]["estimated_working_bytes"] - 1
        with self.assertRaisesRegex(ScoutError, "Chunk assembly"):
            self.prepare(request)
        request["max_estimated_working_bytes"] += 1
        self.prepare(request)

    def test_nonintersecting_request(self):
        request = live.defaults()
        request["roi_mm"] = [[7.6, 100, 7.6], [8, 101, 8]]
        with self.assertRaisesRegex(ScoutError, "does not intersect"):
            self.prepare(request)

    def test_source_change_refused(self):
        self.section.providers["neuroglancer/precomputed"] = "https://example.org/"
        with self.assertRaisesRegex(ScoutError, "source changed"):
            self.prepare()

    def test_exact_scale_missing_refused(self):
        self.volume.scales = self.scales[:1]
        with self.assertRaisesRegex(ScoutError, "no automatic fallback"):
            self.prepare()

    def test_multichannel_metadata_refused(self):
        self.info["num_channels"] = 3
        with self.assertRaisesRegex(ScoutError, "single-channel"):
            self.prepare()

    def test_oversized_nifti_dimension_refused(self):
        request = live.defaults()
        request["scale"] = "1um"
        request["roi_mm"] = [[-50, 8, -40], [50, 10, 50]]
        with self.assertRaisesRegex(ScoutError, "NIfTI-1"):
            self.prepare(request)

    def test_capsule_fetch_and_offline_verify(self):
        _, plan, output, manifest = self.make_capsule()
        self.assertEqual(manifest["schema_version"], "live-1.0")
        self.assertEqual(manifest["fixture_type"], "real_remote_data")
        self.assertEqual(manifest["returned"]["shape"], plan["shape"])
        self.assertEqual(manifest["returned"]["decoded_bytes"], plan["estimates"]["decoded_roi_bytes"])
        self.assertEqual(set(manifest["artifacts"]), {"data", "preview", "metadata", "reproduce"})
        self.assertEqual(len(self.fetch_calls), 1)
        self.assertEqual(self.fetch_calls[0]["resolution_mm"], 0.004)
        self.assertEqual(live.verify(output), manifest)
        self.assertEqual(len(self.fetch_calls), 1)
        self.assertIn("not verified", manifest["source"]["license"])
        self.assertIn("does not prove", (output / "README.md").read_text())

    def test_existing_output_refused_before_fetch(self):
        context, plan = self.prepare()
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(ScoutError, "refusing overwrite"):
            live.fetch(context, plan, output)
        self.assertEqual(self.fetch_calls, [])

    def test_changed_plan_refused_before_fetch(self):
        context, plan = self.prepare()
        plan["shape"][0] += 1
        with self.assertRaisesRegex(ScoutError, "plan changed"):
            live.fetch(context, plan, self.root / "changed")
        self.assertEqual(self.fetch_calls, [])

    def test_returned_resolution_fallback_refused(self):
        context, plan = self.prepare()
        affine = np.asarray(plan["returned_affine_mm"])
        affine[:3, 0] *= 2
        self.image = nib.Nifti1Image(np.zeros(plan["shape"], dtype=np.uint8), affine)
        with self.assertRaisesRegex(ScoutError, "resolution changed"):
            live.fetch(context, plan, self.root / "fallback")
        self.assertFalse((self.root / "fallback").exists())

    def test_same_shape_wrong_spatial_bounds_refused(self):
        context, plan = self.prepare()
        affine = np.asarray(plan["returned_affine_mm"])
        affine[0, 3] += 0.1
        self.image = nib.Nifti1Image(np.zeros(plan["shape"], dtype=np.uint8), affine)
        with self.assertRaisesRegex(ScoutError, "affine differs"):
            live.fetch(context, plan, self.root / "wrong-bounds")

    def test_missing_fetch_result_refused(self):
        context, plan = self.prepare()
        self.image = None
        with self.assertRaisesRegex(ScoutError, "no image"):
            live.fetch(context, plan, self.root / "missing")

    def test_data_tampering_detected(self):
        _, _, output, _ = self.make_capsule()
        path = output / "section-3905.nii"
        content = bytearray(path.read_bytes())
        content[-1] ^= 255
        path.write_bytes(content)
        with self.assertRaisesRegex(ScoutError, "checksum"):
            live.verify(output)

    def test_unsafe_artifact_path_detected(self):
        _, _, output, manifest = self.make_capsule()
        manifest["artifacts"]["data"]["path"] = "../section-3905.nii"
        self.replace_manifest(output, manifest)
        with self.assertRaisesRegex(ScoutError, "unsafe"):
            live.verify(output)

    def test_symlink_artifact_refused(self):
        _, _, output, _ = self.make_capsule()
        path = output / "README.md"
        alternate = self.root / "readme-copy"
        path.rename(alternate)
        path.symlink_to(alternate)
        with self.assertRaisesRegex(ScoutError, "linked"):
            live.verify(output)

    def test_changed_source_and_schema_refused(self):
        _, _, output, manifest = self.make_capsule()
        manifest["source"]["url"] = "https://example.org/"
        self.replace_manifest(output, manifest)
        with self.assertRaisesRegex(ScoutError, "whitelisted"):
            live.verify(output)
        manifest["source"]["url"] = live.SOURCE_URL
        manifest["schema_version"] = "other"
        self.replace_manifest(output, manifest)
        with self.assertRaisesRegex(ScoutError, "live-1.0"):
            live.verify(output)

    def test_recorded_geometry_tampering_detected(self):
        _, _, output, manifest = self.make_capsule()
        manifest["returned"]["bbox_mm"][0][0] += 0.01
        self.replace_manifest(output, manifest)
        with self.assertRaisesRegex(ScoutError, "saved NIfTI"):
            live.verify(output)

    def test_source_metadata_tampering_detected_even_with_new_file_checksum(self):
        _, _, output, manifest = self.make_capsule()
        path = output / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["source_transform_nm"][0][3] += 1000
        path.write_text(json.dumps(metadata))
        self.refresh_artifact(output, manifest, "metadata")
        with self.assertRaisesRegex(ScoutError, "affine does not match"):
            live.verify(output)

    def test_preview_is_checked_against_data_after_its_checksum(self):
        _, _, output, manifest = self.make_capsule()
        preview = output / "preview.pgm"
        content = bytearray(preview.read_bytes())
        content[-1] ^= 1
        preview.write_bytes(content)
        self.refresh_artifact(output, manifest, "preview")
        with self.assertRaisesRegex(ScoutError, "Preview differs"):
            live.verify(output)

    def test_nonfinite_manifest_json_refused(self):
        _, _, output, manifest = self.make_capsule()
        manifest["returned"]["decoded_bytes"] = float("nan")
        self.replace_manifest(output, manifest)
        with self.assertRaisesRegex(ScoutError, "strict capsule JSON"):
            live.verify(output)

    def test_malformed_manifest_fields_raise_scout_errors(self):
        _, _, output, original = self.make_capsule()
        for section, key, value in (("returned", "shape", [101, True, 101]),
                                    ("returned", "decoded_bytes", True),
                                    ("returned", "resolution_mm", ["0.004", 0.02, 0.004]),
                                    ("request", "resolution_mm", 0.004),
                                    ("source", "coordinate_space", "unknown")):
            with self.subTest(section=section, key=key):
                manifest = copy.deepcopy(original)
                manifest[section][key] = value
                self.replace_manifest(output, manifest)
                with self.assertRaises(ScoutError):
                    live.verify(output)

    def test_malformed_metadata_environment_raises_scout_error(self):
        _, _, output, manifest = self.make_capsule()
        path = output / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["environment"] = []
        path.write_text(json.dumps(metadata))
        self.refresh_artifact(output, manifest, "metadata")
        with self.assertRaisesRegex(ScoutError, "metadata environment"):
            live.verify(output)


if __name__ == "__main__":
    unittest.main()
