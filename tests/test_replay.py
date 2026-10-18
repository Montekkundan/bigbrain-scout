"""Trusted offline replay, mocked live replay and optional decoded-NIfTI checks."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bigbrain_scout import core, replay

ROOT = Path(__file__).resolve().parent.parent

try:
    import numpy as np
    import nibabel as nib
except ImportError:
    np = nib = None


class OfflineReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.capsule = self.root / "original"
        self.manifest = core.create_demo(ROOT / "examples" / "safe-request.json",
                                         ROOT / "fixtures" / "synthetic-section.json", self.capsule)

    def save(self, directory=None, manifest=None):
        path = directory or self.capsule
        (path / "manifest.json").write_bytes(core._json_bytes(manifest or self.manifest))

    def refresh(self, role, directory=None, manifest=None):
        directory, manifest = directory or self.capsule, manifest or self.manifest
        record = manifest["artifacts"][role]
        data = (directory / record["path"]).read_bytes()
        record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.save(directory, manifest)

    def test_offline_end_to_end(self):
        output = self.root / "reproduced"
        result = replay.reproduce(self.capsule, output)
        self.assertTrue(result["comparison"]["matched"])
        self.assertTrue(result["comparison"]["data_bytes_equal"])
        self.assertTrue(result["comparison"]["geometry_match"])
        self.assertEqual(result["manifest"], replay.verify_capsule(output))
        self.assertEqual((output / "data.raw").read_bytes(), (self.capsule / "data.raw").read_bytes())
        self.assertIsNone(result["cache"])
        self.assertIsNone(result["network_bytes"])
        self.assertFalse(result["independent_download_proven"])

    def test_capsule_python_is_never_executed(self):
        marker = self.root / "unexpected-execution"
        payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        (self.capsule / "reproduce.py").write_text(payload)
        self.refresh("reproduce")
        result = replay.reproduce(self.capsule, self.root / "trusted-replay")
        self.assertFalse(marker.exists())
        self.assertTrue(result["comparison"]["matched"])
        self.assertIn("reproduce", result["comparison"]["file_hash_differences"])
        self.assertEqual((self.root / "trusted-replay" / "reproduce.py").read_text(), core.REPRODUCE)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(core.ScoutError, "refusing overwrite"):
            replay.reproduce(self.capsule, self.capsule)

    def test_symlink_output_parent_and_capsule_refused(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(core.ScoutError, "symlinks"):
            replay.reproduce(self.capsule, alias / "output")
        with self.assertRaisesRegex(core.ScoutError, "symlinks"):
            replay.verify_capsule(alias / "original")

    def test_symlink_artifact_refused(self):
        data = self.capsule / "data.raw"
        saved = self.root / "data.raw"
        data.rename(saved)
        data.symlink_to(saved)
        with self.assertRaisesRegex(core.ScoutError, "symlinks"):
            replay.verify_capsule(self.capsule)

    def test_tamper_refused_before_replay(self):
        (self.capsule / "data.raw").write_bytes(b"tamper")
        output = self.root / "should-not-exist"
        with self.assertRaisesRegex(core.ScoutError, "checksum"):
            replay.reproduce(self.capsule, output)
        self.assertFalse(output.exists())

    def test_fixture_consistency_checked_even_with_new_data_checksum(self):
        (self.capsule / "data.raw").write_bytes(bytes([1, 2, 3, 4, 5, 6]))
        self.refresh("data")
        with self.assertRaisesRegex(core.ScoutError, "preserved fixture"):
            replay.verify_capsule(self.capsule)

    def test_forged_clipped_bounds_refused(self):
        self.manifest["returned"]["clipped_bbox_mm"] = [[900, 900, 900], [901, 901, 901]]
        self.save()
        with self.assertRaisesRegex(core.ScoutError, "returned geometry"):
            replay.verify_capsule(self.capsule)
        output = self.root / "should-not-exist"
        with self.assertRaisesRegex(core.ScoutError, "returned geometry"):
            replay.reproduce(self.capsule, output)
        self.assertFalse(output.exists())

    def test_unknown_source_and_schema(self):
        for changed in ("source", "schema"):
            with self.subTest(changed=changed):
                manifest = deepcopy(self.manifest)
                if changed == "source":
                    manifest["source"]["id"] = "unknown"
                else:
                    manifest["schema_version"] = "future-version"
                self.save(manifest=manifest)
                with self.assertRaises(core.ScoutError):
                    replay.verify_capsule(self.capsule)

    def test_strict_json_types_duplicates_and_nonfinite(self):
        for data in (b"[]", b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.subTest(data=data):
                (self.capsule / "manifest.json").write_bytes(data)
                with self.assertRaises(core.ScoutError):
                    replay.verify_capsule(self.capsule)

    def test_manifest_and_artifact_limits(self):
        self.manifest["request"]["max_decoded_bytes"] = replay.MAX_BYTES + 1
        self.save()
        with self.assertRaisesRegex(core.ScoutError, "8 MiB"):
            replay.verify_capsule(self.capsule)
        self.manifest["request"]["max_decoded_bytes"] = 64
        self.manifest["artifacts"]["fixture"]["bytes"] = replay.MAX_METADATA_BYTES + 1
        self.save()
        with self.assertRaisesRegex(core.ScoutError, "size limit"):
            replay.verify_capsule(self.capsule)

    def test_traversal_or_unknown_artifact_field_refused(self):
        for changed in ("path", "field"):
            with self.subTest(changed=changed):
                manifest = deepcopy(self.manifest)
                if changed == "path":
                    manifest["artifacts"]["data"]["path"] = "../outside"
                else:
                    manifest["artifacts"]["data"]["url"] = "https://example.invalid/"
                self.save(manifest=manifest)
                with self.assertRaises(core.ScoutError):
                    replay.verify_capsule(self.capsule)

    def test_environment_and_instructions_are_not_data_differences(self):
        output = self.root / "other-environment"
        shutil.copytree(self.capsule, output)
        manifest = deepcopy(self.manifest)
        manifest["provenance"]["environment"]["python"] = "different recorded environment"
        (output / "reproduce.py").write_text("# Different instructions; not executed\n")
        self.refresh("reproduce", output, manifest)
        result = replay.compare_capsules(self.capsule, output)
        self.assertTrue(result["matched"])
        self.assertTrue(result["data_bytes_equal"])
        self.assertFalse(result["environment_match"])
        self.assertFalse(result["artifact_hashes_match"])
        self.assertFalse(result["manifest_file_hash_match"])

    def test_request_mismatch_is_separate_from_data(self):
        output = self.root / "different-budget"
        shutil.copytree(self.capsule, output)
        manifest = deepcopy(self.manifest)
        manifest["request"]["max_decoded_bytes"] = 128
        self.save(output, manifest)
        result = replay.compare_capsules(self.capsule, output)
        self.assertFalse(result["matched"])
        self.assertFalse(result["request_match"])
        self.assertTrue(result["data_bytes_equal"])
        self.assertTrue(result["geometry_match"])

    def test_verified_different_fixture_values_are_data_mismatch(self):
        output = self.root / "different-values"
        fixture = core.load_json(ROOT / "fixtures" / "synthetic-section.json")
        fixture["values"][1] = 18
        altered = self.root / "altered-fixture.json"
        altered.write_bytes(core._json_bytes(fixture))
        core.create_demo(ROOT / "examples" / "safe-request.json", altered, output)
        result = replay.compare_capsules(self.capsule, output)
        self.assertFalse(result["matched"])
        self.assertFalse(result["data_bytes_equal"])
        self.assertTrue(result["geometry_match"])
        self.assertTrue(result["metadata_snapshots_match"])

    def test_mismatch_exception_preserves_output_and_report(self):
        with patch.object(replay, "compare_capsules", return_value={"matched": False, "data_bytes_equal": False}):
            with self.assertRaises(replay.ReplayMismatch) as failure:
                replay.reproduce(self.capsule, self.root / "preserved")
        self.assertEqual(failure.exception.output, str(self.root / "preserved"))
        self.assertFalse(failure.exception.comparison["data_bytes_equal"])
        self.assertTrue((self.root / "preserved" / "manifest.json").exists())

    def test_offline_cache_option_refused(self):
        with self.assertRaisesRegex(core.ScoutError, "does not use a live cache"):
            replay.reproduce(self.capsule, self.root / "output", cache_dir=self.root / "cache")

    def test_trusted_metadata_comes_from_packaged_resource(self):
        with patch.object(replay.resources, "load", wraps=replay.resources.load) as bundled:
            replay.verify_capsule(self.capsule)
        bundled.assert_called_once_with("fixture")

    def test_geometry_tolerance_has_no_relative_component(self):
        self.assertTrue(replay._close([7.5, 8.0], [7.5 + 5e-6, 8.0]))
        self.assertFalse(replay._close([7.5, 8.0], [7.5 + 2e-5, 8.0]))
        self.assertFalse(replay._close([1000.0], [1000.0 + 2e-5]))


class MockLiveReplayTests(unittest.TestCase):
    """No remote request or image-library import is required for these tests."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.capsule = self.root / "original-live"
        self.names = {"data": "section-3905.nii", "preview": "preview.pgm", "metadata": "metadata.json", "reproduce": "README.md"}
        self.descriptor = {"id": "section-3905", "source_url": "https://example.invalid/fixed-trusted-source"}
        self.calls = []
        self.make(self.capsule)

        def prepare(request, cache):
            self.calls.append(("prepare", deepcopy(request), Path(cache)))
            return {"request": request}, {"source_id": "section-3905"}

        def fetch(context, planned, output):
            self.calls.append(("fetch", deepcopy(context), deepcopy(planned)))
            return self.make(Path(output))

        self.adapter = SimpleNamespace(source_ids=lambda: ("section-3905",),
                                       catalog=lambda identifier: deepcopy(self.descriptor),
                                       artifact_names=lambda identifier: deepcopy(self.names),
                                       verify=lambda directory: core.load_json(Path(directory) / "manifest.json"),
                                       prepare=prepare, fetch=fetch)
        self.patcher = patch.object(replay, "_live", return_value=self.adapter)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def make(self, path):
        path.mkdir()
        contents = {"data": b"MOCK_NIFTI_NOT_SCIENTIFIC_DATA", "preview": b"P5\n1 1\n255\n\x01",
                    "metadata": b'{"mock":true}\n', "reproduce": b"Mock instructions are never executed.\n"}
        for role, content in contents.items():
            (path / self.names[role]).write_bytes(content)
        manifest = {"schema_version": "live-1.0", "fixture_type": "real_remote_data",
                    "source": {"id": "section-3905", "url": self.descriptor["source_url"],
                               "coordinate_space": "BigBrain histological reference space", "units": "mm",
                               "license": "Mock only", "citation": "Mock only"},
                    "request": {"bbox_mm": [[7.6, 8, 7.6], [8, 10, 8]], "resolution_key": "4um",
                                "resolution_mm": [0.004, 0.020, 0.004], "resolution_axes": "provider xyz",
                                "max_decoded_bytes": replay.MAX_BYTES, "max_estimated_working_bytes": replay.MAX_BYTES},
                    "returned": {"shape": [1, 1, 1], "dtype": "uint8"},
                    "provenance": {"environment": {"mock": True}, "metadata_snapshots": {}, "warning": "Mock only"},
                    "artifacts": {role: {"path": self.names[role], "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                                  for role, content in contents.items()}}
        (path / "manifest.json").write_bytes(core._json_bytes(manifest))
        return manifest

    def reproduction(self, output, **options):
        with patch.object(replay, "compare_capsules", return_value={"matched": True, "data_bytes_equal": True}):
            return replay.reproduce(self.capsule, output, **options)

    def test_live_requires_explicit_opt_in_before_prepare(self):
        for flag in (False, "yes"):
            with self.subTest(flag=flag), self.assertRaisesRegex(core.ScoutError, "explicit"):
                self.reproduction(self.root / "output", allow_live=flag)
        self.assertEqual(self.calls, [])

    def test_live_request_reconstructed_and_default_cache_fresh(self):
        output = self.root / "reproduced"
        result = self.reproduction(output, allow_live=True)
        self.assertEqual([call[0] for call in self.calls], ["prepare", "fetch"])
        request = self.calls[0][1]
        self.assertEqual(set(request), {"source_id", "roi_mm", "scale", "max_decoded_bytes", "max_estimated_working_bytes"})
        self.assertEqual(request["source_id"], "section-3905")
        self.assertEqual(request["scale"], "4um")
        self.assertTrue(result["cache"]["fresh_directory"])
        self.assertFalse(result["cache"]["explicit_cache"])
        self.assertTrue(Path(result["cache"]["path"]).is_dir())
        self.assertFalse(result["independent_download_proven"])

    def test_reused_cache_reported_and_preserved(self):
        cache = self.root / "existing-cache"
        cache.mkdir()
        marker = cache / "preserve.txt"
        marker.write_text("preserve")
        result = self.reproduction(self.root / "reproduced", allow_live=True, cache_dir=cache)
        self.assertFalse(result["cache"]["fresh_directory"])
        self.assertTrue(result["cache"]["explicit_cache"])
        self.assertEqual(marker.read_text(), "preserve")

    def test_loaded_siibra_refused_before_prepare(self):
        with patch.dict(sys.modules, {"siibra": object()}):
            with self.assertRaisesRegex(core.ScoutError, "fresh process"):
                self.reproduction(self.root / "output", allow_live=True)
        self.assertEqual(self.calls, [])

    def test_unknown_url_refused_before_prepare(self):
        manifest = core.load_json(self.capsule / "manifest.json")
        manifest["source"]["url"] = "https://example.invalid/not-whitelisted"
        (self.capsule / "manifest.json").write_bytes(core._json_bytes(manifest))
        with self.assertRaisesRegex(core.ScoutError, "trusted catalog"):
            self.reproduction(self.root / "output", allow_live=True)
        self.assertEqual(self.calls, [])

    def test_unknown_source_refused(self):
        manifest = core.load_json(self.capsule / "manifest.json")
        manifest["source"]["id"] = "section-0000"
        (self.capsule / "manifest.json").write_bytes(core._json_bytes(manifest))
        with self.assertRaisesRegex(core.ScoutError, "Unknown live source"):
            replay.verify_capsule(self.capsule)

    def test_cache_symlink_overlap_and_file_refused(self):
        real = self.root / "real-cache"
        real.mkdir()
        alias = self.root / "cache-alias"
        alias.symlink_to(real, target_is_directory=True)
        existing_file = self.root / "cache-file"
        existing_file.write_text("not a directory")
        for cache, message in ((alias, "symlinks"), (self.root / "output", "overlap"), (existing_file, "directory")):
            with self.subTest(cache=cache), self.assertRaisesRegex(core.ScoutError, message):
                self.reproduction(self.root / "output", allow_live=True, cache_dir=cache)
        self.assertEqual(self.calls, [])

    def test_prepare_error_propagates_without_output(self):
        self.adapter.prepare = lambda *args: (_ for _ in ()).throw(core.ScoutError("metadata unavailable"))
        with self.assertRaisesRegex(core.ScoutError, "metadata unavailable"):
            self.reproduction(self.root / "output", allow_live=True)
        self.assertFalse((self.root / "output").exists())

    def test_duplicate_metadata_json_refused_before_prepare(self):
        content = b'{"mock":true,"mock":false}\n'
        (self.capsule / "metadata.json").write_bytes(content)
        manifest = core.load_json(self.capsule / "manifest.json")
        manifest["artifacts"]["metadata"].update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        (self.capsule / "manifest.json").write_bytes(core._json_bytes(manifest))
        with self.assertRaisesRegex(core.ScoutError, "unambiguous JSON"):
            self.reproduction(self.root / "output", allow_live=True)
        self.assertEqual(self.calls, [])


@unittest.skipIf(np is None or nib is None, "existing optional live image dependencies are unavailable")
class DecodedNiftiTests(unittest.TestCase):
    def setUp(self):
        # Existing test fixture uses mocked metadata and fetching, not the network.
        from test_live import AdapterTests
        self.fixture = AdapterTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        # macOS /var is a system alias; replay intentionally requires canonical paths.
        self.fixture.root = self.fixture.root.resolve()
        _, _, self.original, self.manifest = self.fixture.make_capsule()
        self.second = self.fixture.root / "other-header"
        shutil.copytree(self.original, self.second)

    def test_nifti_header_difference_is_not_image_byte_mismatch(self):
        image = nib.load(self.second / "section-3905.nii")
        image.header["cal_max"] = 123
        nib.save(image, self.second / "section-3905.nii")
        manifest = deepcopy(self.manifest)
        self.fixture.refresh_artifact(self.second, manifest, "data")
        comparison = replay.compare_capsules(self.original, self.second)
        self.assertTrue(comparison["matched"])
        self.assertTrue(comparison["data_bytes_equal"])
        self.assertFalse(comparison["artifact_hashes_match"])
        self.assertIn("data", comparison["file_hash_differences"])

    def test_metadata_environment_difference_not_image_mismatch(self):
        manifest = deepcopy(self.manifest)
        metadata = core.load_json(self.second / "metadata.json")
        metadata["environment"]["python"] = "different recorded environment"
        manifest["provenance"]["environment"] = deepcopy(metadata["environment"])
        (self.second / "metadata.json").write_bytes(core._json_bytes(metadata))
        self.fixture.refresh_artifact(self.second, manifest, "metadata")
        comparison = replay.compare_capsules(self.original, self.second)
        self.assertTrue(comparison["matched"])
        self.assertTrue(comparison["data_bytes_equal"])
        self.assertTrue(comparison["metadata_snapshots_match"])
        self.assertFalse(comparison["environment_match"])
        self.assertIn("metadata", comparison["file_hash_differences"])

    def test_one_decoded_uint8_value_is_an_explicit_mismatch(self):
        image = nib.load(self.second / "section-3905.nii")
        data = np.asanyarray(image.dataobj).copy()
        data[0, 0, 0] = int(data[0, 0, 0]) + 1
        changed = nib.Nifti1Image(data, image.affine)
        nib.save(changed, self.second / "section-3905.nii")
        plane = data[:, 0, :]
        preview = f"P5\n{plane.shape[1]} {plane.shape[0]}\n255\n".encode() + plane.tobytes()
        (self.second / "preview.pgm").write_bytes(preview)
        manifest = deepcopy(self.manifest)
        self.fixture.refresh_artifact(self.second, manifest, "data")
        self.fixture.refresh_artifact(self.second, manifest, "preview")
        comparison = replay.compare_capsules(self.original, self.second)
        self.assertFalse(comparison["matched"])
        self.assertFalse(comparison["data_bytes_equal"])
        self.assertTrue(comparison["geometry_match"])
        self.assertTrue(comparison["request_match"])
        self.assertNotEqual(comparison["decoded_data_sha256"]["first"], comparison["decoded_data_sha256"]["second"])


if __name__ == "__main__":
    unittest.main()
