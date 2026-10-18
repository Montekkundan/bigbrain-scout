"""Teammate checks exercise trusted code, never silently collect remote images."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bigbrain_scout import check, core, live, resources


def reference_report():
    """Aggregate signatures only, not a synthetic imitation of real tissue."""
    reference = resources.load("audit-reference")
    return {"source": deepcopy(reference["source"]),
            "settings": deepcopy(reference["settings"]),
            "environment": deepcopy(reference["environment"]),
            "levels": deepcopy(reference["levels"]),
            "artifacts": {key: {"sha256": value} for key, value in reference["mask_sha256"].items()}}


class CheckTests(unittest.TestCase):
    def test_default_check_is_offline_and_records_real_synthetic_replay(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(live, "prepare") as prepare:
            output = Path(temporary).resolve() / "check"
            result = check.run_check(output)
            self.assertTrue(result["passed"])
            self.assertTrue(result["synthetic"]["matched"])
            self.assertTrue(result["synthetic"]["unsafe_request_refused"])
            self.assertFalse(result["real_audit"]["executed"])
            self.assertFalse(result["gpu_used"])
            self.assertFalse(result["accuracy_claimed"])
            self.assertEqual(json.loads((output / "check.json").read_bytes()), result)
            self.assertTrue((output / "synthetic" / "manifest.json").is_file())
            self.assertTrue((output / "synthetic-replayed" / "manifest.json").is_file())
            prepare.assert_not_called()

    def test_existing_output_and_symlink_refused_before_work(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(live, "prepare") as prepare:
            output = Path(temporary).resolve()
            with self.assertRaisesRegex(core.ScoutError, "exists"):
                check.run_check(output, allow_live=True)
            link = output / "link"
            link.symlink_to(output, target_is_directory=True)
            with self.assertRaisesRegex(core.ScoutError, "symlink"):
                check.run_check(link / "check", allow_live=True)
            prepare.assert_not_called()

    def test_optin_must_be_boolean_and_local_input_excludes_network(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(live, "prepare") as prepare:
            for value in (1, "yes", None):
                with self.subTest(value=value), self.assertRaises(core.ScoutError):
                    check.run_check(Path(temporary) / "check", allow_live=value)
            with self.assertRaises(core.ScoutError):
                check.run_check(Path(temporary) / "check", allow_live=True, capsule="existing")
            prepare.assert_not_called()

    @patch.dict("sys.modules", {"nibabel": SimpleNamespace()})
    @patch("bigbrain_scout.audit._runtime")
    def test_local_capsule_audit_reuses_trusted_code_without_network(self, runtime):
        from bigbrain_scout import audit
        with tempfile.TemporaryDirectory() as temporary, patch.object(live, "prepare") as prepare:
            output = Path(temporary).resolve() / "check"
            with patch.object(audit, "run_audit", return_value=reference_report()) as run:
                result = check.run_check(output, capsule="existing")
            self.assertTrue(result["passed"])
            self.assertTrue(result["real_audit"]["reference_matched"])
            self.assertEqual(result["real_audit"]["acquisition"], "existing_local_capsule")
            run.assert_called_once_with("existing", output / "audit")
            prepare.assert_not_called()

    @patch.dict("sys.modules", {"nibabel": SimpleNamespace()})
    @patch("bigbrain_scout.audit._runtime")
    def test_live_check_fetches_exact_native_preset_then_audits(self, runtime):
        from bigbrain_scout import audit
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "check"
            with patch.object(live, "prepare", return_value=("context", {"plan": True})) as prepare, \
                    patch.object(live, "fetch") as fetch, \
                    patch.object(audit, "run_audit", return_value=reference_report()):
                result = check.run_check(output, allow_live=True)
            request = {**live.defaults(), "scale": "1um"}
            prepare.assert_called_once_with(request, output / "cache")
            fetch.assert_called_once_with("context", {"plan": True}, output / "native")
            self.assertTrue(result["passed"])
            self.assertEqual(result["real_audit"]["acquisition"], "explicit_live_fetch")
            self.assertFalse(result["real_audit"]["independent_download_proven"])

    def test_reference_compares_source_settings_versions_counts_and_each_mask(self):
        original = reference_report()
        self.assertTrue(check.compare_reference(original)["matched"])
        changes = (
            lambda r: r["source"].update(crop_sha256="0" * 64),
            lambda r: r["source"].update(decoded_sha256="0" * 64),
            lambda r: r["source"].update(width=396),
            lambda r: r["settings"].update(dark_threshold_uint8=91.0),
            lambda r: r["environment"].update(numpy="other"),
            lambda r: r["levels"][0].update(candidate_count=70),
            *(lambda r, key=key: r["artifacts"][key].update(sha256="0" * 64)
              for key in ("mask_1um", "mask_2um", "mask_4um")),
        )
        for change in changes:
            report = deepcopy(original)
            change(report)
            with self.subTest(report=report):
                comparison = check.compare_reference(report)
                self.assertFalse(comparison["matched"])
                self.assertTrue(comparison["differences"])
        # Python patch versions and wall-clock time need not be identical.
        original["environment"]["python"] = "different supported Python"
        original["runtime_seconds_total"] = 99.0
        self.assertTrue(check.compare_reference(original)["matched"])

    @patch.dict("sys.modules", {"nibabel": SimpleNamespace()})
    @patch("bigbrain_scout.audit._runtime")
    def test_changed_reference_is_failed_check_with_evidence_retained(self, runtime):
        from bigbrain_scout import audit
        report = reference_report()
        report["levels"][0]["candidate_count"] += 1
        with tempfile.TemporaryDirectory() as temporary, patch.object(audit, "run_audit", return_value=report):
            output = Path(temporary).resolve() / "check"
            result = check.run_check(output, capsule="existing")
            self.assertFalse(result["passed"])
            self.assertTrue((output / "check.json").is_file())
            self.assertIn("levels", result["real_audit"]["comparison"]["differences"])

    def test_missing_scientific_dependencies_refused_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "check"
            with patch("bigbrain_scout.audit._runtime", side_effect=core.ScoutError("requires NumPy")):
                with self.assertRaisesRegex(core.ScoutError, "NumPy"):
                    check.run_check(output, capsule="existing")
            self.assertFalse(output.exists())
            with patch("bigbrain_scout.audit._runtime"), patch.dict("sys.modules", {"nibabel": None}):
                with self.assertRaisesRegex(core.ScoutError, "audit extra"):
                    check.run_check(output, capsule="existing")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
