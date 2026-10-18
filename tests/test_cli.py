"""CLI refusal, opt-in, trusted replay and explicit mismatch exit contracts."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bigbrain_scout import __main__, live


class CliTests(unittest.TestCase):
    def call(self, *arguments):
        output, error = io.StringIO(), io.StringIO()
        code = 0
        with patch("sys.argv", ["bigbrain-scout", *map(str, arguments)]), redirect_stdout(output), redirect_stderr(error):
            try:
                __main__.main()
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue(), error.getvalue()

    def test_source_catalog_is_metadata_only(self):
        with patch.object(live, "_runtime") as runtime:
            code, output, error = self.call("sources")
            self.assertEqual(code, 0, error)
            self.assertEqual([item["id"] for item in json.loads(output)["sources"]], ["section-3905", "section-3954", "section-4003"])
            runtime.assert_not_called()

    def test_live_plan_and_fetch_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(live, "prepare") as prepare:
            for arguments in (("plan", "--source", "section-3954"),
                              ("fetch", "--source", "section-3954", "--output", str(Path(temporary) / "capsule"))):
                with self.subTest(arguments=arguments):
                    code, _, error = self.call(*arguments)
                    self.assertEqual(code, 2)
                    self.assertIn("--allow-live", error)
            prepare.assert_not_called()

    def test_unknown_source_refused_before_network(self):
        with patch.object(live, "_runtime") as runtime:
            code, _, error = self.call("plan", "--source", "https://example.org/", "--allow-live")
            self.assertEqual(code, 2)
            self.assertIn("whitelisted", error)
            runtime.assert_not_called()

    def test_fetch_existing_output_refused_before_metadata(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(live, "prepare") as prepare:
            code, _, error = self.call("fetch", "--source", "section-3954", "--output", temporary, "--allow-live")
            self.assertEqual(code, 2)
            self.assertIn("overwrite", error)
            prepare.assert_not_called()

    def test_replay_compare_and_overwrite_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first, second = root / "first", root / "second"
            self.assertEqual(self.call("demo", "--output", first)[0], 0)
            code, output, error = self.call("reproduce", "--capsule", first, "--output", second)
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["comparison"]["matched"])
            code, output, error = self.call("compare", "--first", first, "--second", second)
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["data_bytes_equal"])
            self.assertEqual(self.call("reproduce", "--capsule", first, "--output", second)[0], 2)

    def test_comparison_mismatch_has_machine_readable_output_and_exit_three(self):
        from bigbrain_scout import replay
        with patch.object(replay, "compare_capsules", return_value={"matched": False, "data_bytes_equal": False}):
            code, output, error = self.call("compare", "--first", "a", "--second", "b")
            self.assertEqual(code, 3)
            self.assertFalse(json.loads(output)["matched"])
            self.assertIn("differs", error)

    def test_replay_mismatch_retains_output_and_reports_exit_three(self):
        from bigbrain_scout import replay
        mismatch = replay.ReplayMismatch(Path("retained"), {"matched": False})
        with patch.object(replay, "reproduce", side_effect=mismatch):
            code, output, error = self.call("reproduce", "--capsule", "a", "--output", "retained", "--allow-live")
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(output)["output"], "retained")
            self.assertIn("output was preserved", error)

    def test_audit_dispatch_is_offline_and_machine_readable(self):
        from bigbrain_scout import audit
        with patch.object(audit, "run_audit", return_value={"accuracy_claimed": False}) as run:
            code, output, error = self.call("audit", "--capsule", "input", "--output", "output")
            self.assertEqual(code, 0, error)
            self.assertFalse(json.loads(output)["accuracy_claimed"])
            run.assert_called_once_with("input", "output")
        with patch.object(audit, "verify_audit", return_value={"verified": True}) as verify:
            code, output, error = self.call("verify-audit", "--audit", "output")
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["verified"])
            verify.assert_called_once_with("output")

    def test_teammate_check_dispatch_and_mismatch_exit_code(self):
        from bigbrain_scout import check
        for passed, expected in ((True, 0), (False, 3)):
            with self.subTest(passed=passed), patch.object(check, "run_check", return_value={"passed": passed}) as run:
                code, output, error = self.call("check", "--output", "new", "--capsule", "existing")
                self.assertEqual(code, expected, error)
                self.assertEqual(json.loads(output)["passed"], passed)
                run.assert_called_once_with("new", allow_live=False, capsule="existing")
        with patch.object(check, "run_check", return_value={"passed": True}) as run:
            self.assertEqual(self.call("check", "--output", "new", "--allow-live")[0], 0)
            run.assert_called_once_with("new", allow_live=True, capsule=None)
        with patch.object(check, "run_check") as run:
            self.assertEqual(self.call("check", "--output", "new", "--allow-live", "--capsule", "existing")[0], 2)
            run.assert_not_called()

    def test_evaluation_requires_declared_reference_and_threshold(self):
        from bigbrain_scout import evaluation
        with patch.object(evaluation, "evaluate_label_files", return_value={"f1": 0.5}) as evaluate:
            code, output, error = self.call("evaluate", "--prediction", "p", "--reference", "r", "--iou-threshold", "0.7")
            self.assertEqual(code, 0, error)
            self.assertEqual(json.loads(output)["f1"], 0.5)
            evaluate.assert_called_once_with("p", "r", iou_threshold=0.7)

    def test_model_metadata_stays_offline_and_execution_is_held(self):
        from bigbrain_scout import neural
        with patch.object(neural, "_runtime") as runtime:
            code, output, error = self.call("model-plan")
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["on_hold"])
            self.assertFalse(json.loads(output)["inference_executed"])
            code, _, error = self.call("infer", "--audit", "unused", "--checkpoint", "unused", "--output", "unused")
            self.assertEqual(code, 2)
            self.assertIn("on hold", error)
            runtime.assert_not_called()

    def test_inference_dispatch_propagates_explicit_optin_without_running_a_model(self):
        from bigbrain_scout import neural
        with patch.object(neural, "run_model", return_value={"gpu_used": True}) as infer:
            code, output, error = self.call("infer", "--audit", "a", "--checkpoint", "c", "--output", "o", "--allow-gpu")
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["gpu_used"])
            infer.assert_called_once_with("a", "c", "o", allow_gpu=True)


if __name__ == "__main__":
    unittest.main()
