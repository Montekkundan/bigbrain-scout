"""Offline audit contracts. Numerical fixtures are NONBIOLOGICAL, not brains.

The mocked adapter capsule exercises trusted-format verification only. It does
not constitute an acquired section, annotation, or biological validation.
"""

import copy
import hashlib
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from bigbrain_scout import audit, core, live

try:
    import numpy as np
    import nibabel as nib
    import scipy.ndimage
except ImportError:
    np = nib = None


class SettingsTests(unittest.TestCase):
    def test_defaults_are_independent_and_partial_settings_are_normalized(self):
        first = audit._settings({"dark_threshold_uint8": 120})
        first["smoothing_sigma_um"] = 7
        self.assertEqual(audit._settings()["smoothing_sigma_um"], 1.5)
        self.assertEqual(audit._settings({"dark_threshold_uint8": 120})["dark_threshold_uint8"], 120.0)

    def test_unknown_nonfinite_boolean_and_unbounded_settings_refused(self):
        cases = [[], {"unknown": 1}, {"smoothing_sigma_um": -1}, {"smoothing_sigma_um": 11},
                 {"dark_threshold_uint8": 256}, {"min_area_um2": 0}, {"max_area_um2": 16385},
                 {"min_area_um2": 200, "max_area_um2": 100}]
        cases.extend({key: value} for key in audit.DEFAULT_SETTINGS
                     for value in (True, float("nan"), float("inf"), "1"))
        for value in cases:
            with self.subTest(value=value), self.assertRaises(core.ScoutError):
                audit._settings(value)


@unittest.skipIf(np is None, "existing NumPy/SciPy/nibabel environment is optional")
class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.settings = audit._settings({"smoothing_sigma_um": 0.1, "dark_threshold_uint8": 127,
                                         "min_area_um2": 1, "max_area_um2": 1000})

    def test_block_mean_preserves_float64_fractional_intensity(self):
        plane = np.asarray([[0, 0, 0, 1], [0, 1, 1, 1], [2, 2, 2, 3], [2, 3, 3, 3]], dtype=np.uint8)
        self.assertEqual(audit._block_mean(plane, 2).dtype, np.float64)
        np.testing.assert_array_equal(audit._block_mean(plane, 2), [[0.25, 0.75], [2.25, 2.75]])
        np.testing.assert_array_equal(audit._block_mean(plane, 1), plane)
        self.assertEqual(audit._block_mean(plane, 4)[0, 0], 1.5)

    def test_bad_source_plane_or_factor_refused(self):
        cases = [(np.zeros((4, 4), dtype=np.uint8), value) for value in (True, 2.0, 3, 0)]
        cases.extend((plane, factor) for plane, factor in ((np.zeros((3, 4), dtype=np.uint8), 2),
                     (np.zeros((4, 4), dtype=np.float64), 1), (np.zeros((0, 4), dtype=np.uint8), 1),
                     (np.zeros((1028, 1028), dtype=np.uint8), 1), (np.zeros(4, dtype=np.uint8), 1)))
        for plane, factor in cases:
            with self.subTest(shape=plane.shape, factor=factor), self.assertRaises(core.ScoutError):
                audit._block_mean(plane, factor)

    def test_same_physical_component_area_and_fixed_settings_across_scales(self):
        plane = np.full((64, 64), 255, dtype=np.uint8)
        plane[16:32, 16:32] = 0
        settings = {**self.settings, "min_area_um2": 128, "max_area_um2": 300}
        for resolution in audit.RESOLUTIONS:
            summary, mask = audit._candidates(audit._block_mean(plane, resolution), resolution, settings)
            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["total_candidate_area_um2"], 256)
            self.assertEqual(summary["candidates"][0]["area_pixels"], 256 // resolution**2)
            self.assertEqual(summary["smoothing_sigma_pixels"], 0.1 / resolution)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_edge_touching_components_excluded_at_every_scale(self):
        plane = np.full((64, 64), 255, dtype=np.uint8)
        plane[:8, 8:16] = 0
        plane[32:40, 32:40] = 0
        for resolution in audit.RESOLUTIONS:
            summary, mask = audit._candidates(audit._block_mean(plane, resolution), resolution, self.settings)
            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["excluded_border_components"], 1)
            self.assertFalse(mask[0].any())
            self.assertEqual(summary["total_candidate_area_um2"], 64)

    def test_eight_connected_diagonal_pixels_are_one_component(self):
        values = np.full((8, 8), 255.0, dtype=np.float64)
        values[2, 2] = values[3, 3] = 0
        summary, _ = audit._candidates(values, 1, self.settings)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["candidates"][0]["area_pixels"], 2)
        self.assertEqual(summary["candidates"][0]["bbox_row_col_exclusive"], [[2, 2], [4, 4]])
        self.assertEqual(summary["candidates"][0]["centroid_row_col"], [2.5, 2.5])

    def test_physical_area_bounds_are_inclusive(self):
        values = np.full((16, 16), 255.0, dtype=np.float64)
        values[2:4, 2:4] = 0
        values[8:11, 8:11] = 0
        summary, _ = audit._candidates(values, 1, {**self.settings, "min_area_um2": 4, "max_area_um2": 9})
        self.assertEqual([item["area_um2"] for item in summary["candidates"]], [4, 9])
        summary, _ = audit._candidates(values, 1, {**self.settings, "min_area_um2": 5, "max_area_um2": 8})
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(summary["rejected_area_components"], 2)

    def test_zero_native_count_has_no_undefined_percentage(self):
        levels, _ = audit._calculate(np.full((16, 16), 255, dtype=np.uint8), self.settings)
        for level in levels:
            self.assertEqual(level["candidate_count"], 0)
            self.assertEqual(level["count_drift_from_native"], 0)
            self.assertIsNone(level["count_drift_percent_from_native"])
            self.assertIsNone(level["mean_candidate_area_um2"])
            self.assertIsNone(level["min_candidate_area_um2"])

    def test_count_drift_uses_native_count_without_accuracy_claims(self):
        plane = np.full((16, 16), 255, dtype=np.uint8)
        plane[2, 2] = plane[6, 6] = 0
        levels, _ = audit._calculate(plane, self.settings)
        self.assertEqual([level["candidate_count"] for level in levels], [2, 0, 0])
        self.assertEqual([level["count_drift_from_native"] for level in levels], [0, -2, -2])
        self.assertEqual([level["count_drift_percent_from_native"] for level in levels], [0, -100, -100])

    def test_invalid_candidate_inputs_refused(self):
        values = np.full((8, 8), 255.0, dtype=np.float64)
        cases = [(values, size) for size in (True, 1.0, 3)]
        cases.extend((plane, 1) for plane in (values.astype(np.uint8), np.zeros((1, 8), dtype=np.float64),
                                              np.full((8, 8), float("nan")), np.full((8, 8), 256.0)))
        for plane, size in cases:
            with self.subTest(size=size), self.assertRaises(core.ScoutError):
                audit._candidates(plane, size, self.settings)

    def test_png_is_grayscale_exact_mask_with_valid_chunk_crcs(self):
        mask = np.asarray([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        content = audit._png(mask)
        self.assertEqual(content[:8], b"\x89PNG\r\n\x1a\n")
        offset, compressed = 8, b""
        while offset < len(content):
            size = struct.unpack(">I", content[offset:offset + 4])[0]
            kind = content[offset + 4:offset + 8]
            data = content[offset + 8:offset + 8 + size]
            crc = struct.unpack(">I", content[offset + 8 + size:offset + 12 + size])[0]
            self.assertEqual(crc, zlib.crc32(kind + data) & 0xffffffff)
            if kind == b"IHDR":
                self.assertEqual(struct.unpack(">IIBBBBB", data), (3, 2, 8, 0, 0, 0, 0))
            if kind == b"IDAT":
                compressed += data
            offset += 12 + size
        self.assertEqual(zlib.decompress(compressed), b"\0\0\xff\0\0\xff\0\xff")

    def test_component_and_total_candidate_report_limits(self):
        values = np.full((64, 64), 255.0, dtype=np.float64)
        values[2::4, 2::4] = 0
        with patch.object(audit, "MAX_COMPONENTS", 2), self.assertRaisesRegex(core.ScoutError, "threshold components"):
            audit._candidates(values, 1, self.settings)
        with patch.object(audit, "MAX_CANDIDATES", 2), self.assertRaisesRegex(core.ScoutError, "Too many candidates"):
            audit._candidates(values, 1, self.settings)


@unittest.skipIf(np is None, "existing NumPy/SciPy/nibabel environment is optional")
class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "audit"
        self.plane = np.full((400, 400), 255, dtype=np.uint8)
        self.plane[40:56, 80:96] = 0
        trusted = live.catalog("section-3905")
        self.source = {"id": trusted["id"], "url": trusted["source_url"], "citation": trusted["configuration_url"],
                       "license": live.LICENSE_STATUS, "coordinate_space": "BigBrain histological reference space",
                       "units": "mm", "native_resolution_um": 1, "through_section_um": 20,
                       "source_shape": [401, 1, 401],
                       "source_affine_mm": [[0.001, 0, 0, 7.6], [0, 0.02, 0, 8.0705],
                                            [0, 0, 0.001, 7.6], [0, 0, 0, 1]],
                       "request_bbox_mm": live.defaults()["roi_mm"], "width": 400, "height": 400,
                       "fov_um": [400, 400], "nifti_sha256": "a" * 64, "decoded_sha256": "b" * 64,
                       "crop_sha256": hashlib.sha256(self.plane.tobytes()).hexdigest(), "orientation": audit.ORIENTATION,
                       "crop_index_start_xyz": [0, 0, 0], "crop_index_stop_xyz_exclusive": [400, 1, 400]}

    def build(self, output=None):
        # Mock only the already-verified acquisition boundary. This is NOT brain data.
        with patch.object(audit, "_source", return_value=(copy.deepcopy(self.source), self.plane.copy())):
            return audit.run_audit(self.root / "nonbiological-source", output or self.output)

    def replace_report(self, report):
        (self.output / "report.json").write_bytes(core._json_bytes(report))

    def test_full_flow_recomputed_and_all_outputs_repeatable_except_runtime(self):
        first = self.build()
        second = self.build(self.root / "second")
        self.assertEqual(audit.verify_audit(self.output), first)
        self.assertEqual(audit._without_runtime(first["levels"]), audit._without_runtime(second["levels"]))
        self.assertEqual(first["artifacts"], second["artifacts"])
        self.assertEqual(first["source"], second["source"])
        self.assertEqual(first["environment"]["scipy"], audit.importlib.metadata.version("scipy"))
        self.assertFalse(first["accuracy_established"])
        self.assertFalse(first["gpu_used"])
        self.assertEqual(first["runtime_seconds_total"], sum(item["runtime_seconds"] for item in first["levels"]))
        self.assertEqual([(item["width"], item["height"]) for item in first["levels"]], [(400, 400), (200, 200), (100, 100)])
        for name in audit.ARTIFACT_PATHS.values():
            self.assertEqual((self.output / name).read_bytes(), (self.root / "second" / name).read_bytes())

    def test_existing_output_refused_before_source_access(self):
        self.output.mkdir()
        with patch.object(audit, "_source") as source, self.assertRaisesRegex(core.ScoutError, "overwrite"):
            audit.run_audit(self.root / "not-read", self.output)
        source.assert_not_called()

    def test_oversized_report_refused_before_output_creation(self):
        with patch.object(audit, "MAX_REPORT", 1), self.assertRaisesRegex(core.ScoutError, "size limit"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_output_symlink_parent_refused(self):
        (self.root / "alias").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(core.ScoutError, "symlink"):
            audit.run_audit(self.root / "not-read", self.root / "alias" / "new")

    def test_hash_tampering_refused(self):
        self.build()
        (self.output / "candidate-1um.raw").write_bytes(b"wrong")
        with self.assertRaisesRegex(core.ScoutError, "checksum"):
            audit.verify_audit(self.output)

    def test_rehashed_mask_still_refused_by_recomputation(self):
        report = self.build()
        name = audit.ARTIFACT_PATHS["mask_1um"]
        data = b"\0" * (400 * 400)
        (self.output / name).write_bytes(data)
        report["artifacts"]["mask_1um"].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.replace_report(report)
        with self.assertRaisesRegex(core.ScoutError, "recomputation"):
            audit.verify_audit(self.output)

    def test_rehashed_source_cannot_keep_old_candidate_metrics(self):
        report = self.build()
        data = b"\xff" * (400 * 400)
        digest = hashlib.sha256(data).hexdigest()
        (self.output / "source.raw").write_bytes(data)
        report["artifacts"]["source"].update(sha256=digest)
        report["source"]["crop_sha256"] = digest
        self.replace_report(report)
        with self.assertRaisesRegex(core.ScoutError, "recomputation"):
            audit.verify_audit(self.output)

    def test_forged_numeric_and_boolean_metrics_refused(self):
        original = self.build()
        for value in (999, True):
            report = copy.deepcopy(original)
            report["levels"][0]["candidate_count"] = value
            self.replace_report(report)
            with self.subTest(value=value), self.assertRaisesRegex(core.ScoutError, "recomputation"):
                audit.verify_audit(self.output)

    def test_traversal_unknown_schema_extra_fields_and_files_refused(self):
        original = self.build()
        cases = [copy.deepcopy(original) for _ in range(4)]
        cases[0]["artifacts"]["source"]["path"] = "../../escaped.raw"
        cases[1]["schema_version"] = "audit-unknown"
        cases[2]["levels"][0]["unknown"] = 1
        del cases[3]["settings"]["dark_threshold_uint8"]
        for index, report in enumerate(cases):
            self.replace_report(report)
            with self.subTest(index=index), self.assertRaises(core.ScoutError):
                audit.verify_audit(self.output)
        self.replace_report(original)
        (self.output / "extra.txt").write_text("not allowed", encoding="utf-8")
        with self.assertRaisesRegex(core.ScoutError, "unexpected"):
            audit.verify_audit(self.output)
        self.assertFalse((self.root / "escaped.raw").exists())

    def test_artifact_symlink_refused(self):
        self.build()
        path = self.output / "source.raw"
        stored = self.root / "stored.raw"
        path.rename(stored)
        path.symlink_to(stored)
        with self.assertRaisesRegex(core.ScoutError, "symlink"):
            audit.verify_audit(self.output)

    def test_version_drift_and_bad_runtime_refused(self):
        original = self.build()
        cases = [copy.deepcopy(original) for _ in range(3)]
        cases[0]["environment"]["scipy"] = "0.0"
        cases[1]["levels"][0]["runtime_seconds"] = -1
        cases[2]["runtime_seconds_total"] = 999
        for index, report in enumerate(cases):
            self.replace_report(report)
            with self.subTest(index=index), self.assertRaises(core.ScoutError):
                audit.verify_audit(self.output)

    def test_strict_finite_json_and_top_level_object_required(self):
        self.build()
        for content in (b"[]", b'{"schema_version":"audit-1.0","schema_version":"audit-1.0"}',
                        b'{"schema_version":"audit-1.0","value":NaN}'):
            (self.output / "report.json").write_bytes(content)
            with self.subTest(content=content), self.assertRaises(core.ScoutError):
                audit.verify_audit(self.output)


@unittest.skipIf(np is None, "existing NumPy/SciPy/nibabel environment is optional")
class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def capsule(self):
        """Captured-style source metadata plus a clearly NONBIOLOGICAL fake provider."""
        info = {"data_type": "uint8", "num_channels": 1, "scales": [
            {"key": "1,1,1", "resolution": [1000, 20000, 1000], "size": [120900, 1, 139120],
             "voxel_offset": [0, 0, 0], "chunk_sizes": [[256, 1, 256]], "encoding": "raw"}]}
        transform = np.asarray([[0, 0, 1, -70677183.3333333], [0, 1, 0, 8070000],
                                [1, 0, 0, -58788283.3333333], [0, 0, 0, 1]], dtype=float)
        affine = transform @ np.diag([1000, 20000, 1000, 1.0])
        affine[:3, 3] += 500
        affine[:3] /= 1e6
        bbox = live._image_bbox(np, affine, info["scales"][0]["size"])
        section = SimpleNamespace(name="#3905: NONBIOLOGICAL mocked test section", id="NONBIOLOGICAL-test",
                                  providers={"neuroglancer/precomputed": live.SOURCE_URL},
                                  get_boundingbox=lambda **kwargs: SimpleNamespace(minpoint=bbox[0], maxpoint=bbox[1]))
        volume = SimpleNamespace(_info=info, transform_nm=transform,
                                 scales=[SimpleNamespace(key="1,1,1", res_mm=np.asarray([0.001, 0.020, 0.001]), affine=affine)])
        siibra = SimpleNamespace(locations=SimpleNamespace(BoundingBox=lambda *args, **kwargs: (args, kwargs)),
                                 features=SimpleNamespace(get=lambda *args: [section]))
        request = live.defaults()
        request["scale"] = "1um"
        with patch.object(live, "_runtime", return_value=(np, siibra, lambda url: volume)):
            context, plan = live.prepare(request, self.root / "cache")
        data = np.arange(np.prod(plan["shape"]), dtype=np.uint8).reshape(plan["shape"])
        section.fetch = lambda **kwargs: nib.Nifti1Image(data, np.asarray(plan["returned_affine_mm"]))
        output = self.root / "capsule"
        manifest = live.fetch(context, plan, output)
        return output, manifest, data

    def test_native_capsule_verified_transposed_and_high_edges_trimmed(self):
        output, manifest, data = self.capsule()
        record, plane = audit._source(output)
        self.assertEqual(data.shape, (401, 1, 401))
        self.assertEqual(plane.shape, (400, 400))
        np.testing.assert_array_equal(plane, data[:400, 0, :400].T)
        self.assertEqual(record["crop_sha256"], hashlib.sha256(plane.tobytes()).hexdigest())
        self.assertEqual(record["decoded_sha256"], hashlib.sha256(data.tobytes()).hexdigest())
        self.assertEqual(record["nifti_sha256"], manifest["artifacts"]["data"]["sha256"])
        self.assertEqual(record["fov_um"], [400, 400])

    def test_source_hash_tamper_stops_before_audit_decode(self):
        output, manifest, _ = self.capsule()
        path = output / manifest["artifacts"]["data"]["path"]
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with patch.object(audit.replay, "_decoded") as decoded, self.assertRaises(core.ScoutError):
            audit._source(output)
        decoded.assert_not_called()

    def test_unsupported_source_resolution_geometry_dtype_and_size_prechecked(self):
        output, original, _ = self.capsule()
        cases = [copy.deepcopy(original) for _ in range(5)]
        cases[0]["request"]["resolution_key"] = "4um"
        cases[1]["source"]["id"] = "section-3954"
        cases[2]["returned"]["affine"][0][0] = 0.004
        cases[3]["returned"]["dtype"] = "float64"
        cases[4]["returned"].update(shape=[1028, 1, 1028], decoded_bytes=1028**2)
        for index, manifest in enumerate(cases):
            (output / "manifest.json").write_bytes(core._json_bytes(manifest))
            with self.subTest(index=index), patch.object(audit.replay, "_inspect") as inspect:
                with self.assertRaises(core.ScoutError):
                    audit._source(output)
                inspect.assert_not_called()

    def test_source_symlink_refused(self):
        output, _, _ = self.capsule()
        (self.root / "alias").symlink_to(output, target_is_directory=True)
        with self.assertRaisesRegex(core.ScoutError, "symlink"):
            audit._source(self.root / "alias")
