"""Offline guards and mocked inference only. No checkpoint download or model run."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bigbrain_scout import neural
from bigbrain_scout.core import ScoutError

try:
    import numpy as np
except ImportError:
    np = None


class ModelPlanTests(unittest.TestCase):
    def test_plan_is_dependency_free_and_held(self):
        code = "from bigbrain_scout.neural import model_plan; import json,sys; p=model_plan(); assert p['on_hold']; assert not any(x in sys.modules for x in ('numpy','scipy','torch','cellpose')); print(json.dumps(p,allow_nan=False))"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertFalse(plan["inference_executed"])
        self.assertFalse(plan["training_executed"])
        self.assertFalse(plan["accuracy_established"])
        self.assertFalse(plan["gpu_runtime_tested"])
        self.assertEqual(plan["checkpoint"]["bytes"], 609391431)
        self.assertEqual(plan["checkpoint"]["sha256"], "98f1b62df3db68ad21a54725caacc3b718f7d46c1c1a40eafef9932674a3e534")
        self.assertIsNone(plan["checkpoint"]["training_section_ids"])
        self.assertEqual(plan["protocol"]["diameter_pixels"], [30, 15, 7.5])

    def test_plan_mutation_cannot_change_future_requests(self):
        plan = neural.model_plan()
        plan["checkpoint"]["sha256"] = "bad"
        plan["protocol"]["eval_parameters"]["batch_size"] = 1000
        self.assertEqual(neural.model_plan()["checkpoint"]["sha256"], neural.CHECKPOINT_SHA256)
        self.assertEqual(neural.model_plan()["protocol"]["eval_parameters"]["batch_size"], 4)

    def test_optin_is_exact_boolean_and_precedes_all_io(self):
        for value in (False, None, 1, "true", [], {}):
            with self.subTest(value=value), patch.object(neural, "_audit_source") as source, \
                    patch.object(neural, "_checkpoint") as checkpoint, patch.object(neural, "_runtime") as runtime:
                with self.assertRaisesRegex(ScoutError, "on hold"):
                    neural.run_model("missing", "missing", "missing", allow_gpu=value)
                source.assert_not_called()
                checkpoint.assert_not_called()
                runtime.assert_not_called()

    def test_refusal_does_not_import_optional_packages(self):
        code = "from bigbrain_scout.neural import run_model; from bigbrain_scout.core import ScoutError; import sys\ntry: run_model('absent','absent','absent')\nexcept ScoutError: pass\nassert not any(x in sys.modules for x in ('numpy','scipy','torch','cellpose'))"
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    def test_wrong_version_refuses_before_framework_import(self):
        with patch.object(neural.importlib.metadata, "version", return_value="4.1.0"), \
                patch.object(neural.importlib, "import_module") as imports:
            with self.assertRaisesRegex(ScoutError, "cellpose==4.1.1"):
                neural._runtime()
            imports.assert_not_called()

    def test_cuda_absence_does_not_import_model(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        calls = []

        def modules(name):
            calls.append(name)
            return torch if name == "torch" else object()

        with patch.object(neural.importlib.metadata, "version", return_value="4.1.1"), \
                patch.object(neural.importlib, "import_module", side_effect=modules):
            with self.assertRaisesRegex(ScoutError, "no CPU/MPS fallback"):
                neural._runtime()
        self.assertEqual(calls, ["numpy", "torch"])

    def test_missing_dependency_is_actionable(self):
        with patch.object(neural.importlib.metadata, "version", side_effect=neural.importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(ScoutError, "no install or download"):
                neural._runtime()

    def test_broken_gpu_library_import_is_actionable(self):
        with patch.object(neural.importlib.metadata, "version", return_value="4.1.1"), \
                patch.object(neural.importlib, "import_module", side_effect=OSError("missing CUDA library")):
            with self.assertRaisesRegex(ScoutError, "initialization failed"):
                neural._runtime()

    def test_no_download_hook_always_refuses(self):
        with self.assertRaisesRegex(ScoutError, "download is disabled"):
            neural._no_download()

    def test_invalid_measurements_are_rejected(self):
        for value in (float("nan"), float("inf"), -1, True, "1"):
            with self.assertRaises(ScoutError):
                neural._measurement(value, "runtime")
        for value in (1.0, -1, True, float("nan")):
            with self.assertRaises(ScoutError):
                neural._measurement(value, "memory", integer=True)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "checkpoint"
        self.data = b"NONBIOLOGICAL checkpoint-shaped test bytes; not a model"
        self.path.write_bytes(self.data)
        self.size = patch.object(neural, "CHECKPOINT_BYTES", len(self.data))
        self.digest = patch.object(neural, "CHECKPOINT_SHA256", hashlib.sha256(self.data).hexdigest())
        self.size.start()
        self.digest.start()
        self.addCleanup(self.size.stop)
        self.addCleanup(self.digest.stop)

    def test_valid_small_mock_checkpoint_streamed(self):
        path, record = neural._checkpoint(self.path)
        self.assertEqual(path, self.path)
        self.assertEqual(record.st_size, len(self.data))

    def test_missing_checkpoint_is_not_downloaded(self):
        with self.assertRaisesRegex(ScoutError, "no download"):
            neural._checkpoint(self.root / "missing")

    def test_size_mismatch(self):
        self.path.write_bytes(self.data + b"x")
        with self.assertRaisesRegex(ScoutError, "exact recorded size"):
            neural._checkpoint(self.path)

    def test_hash_mismatch(self):
        self.path.write_bytes(b"x" * len(self.data))
        with self.assertRaisesRegex(ScoutError, "SHA-256"):
            neural._checkpoint(self.path)

    def test_symlink_and_directory_refused(self):
        alias = self.root / "alias"
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(ScoutError, "symlinks"):
            neural._checkpoint(alias)
        with self.assertRaisesRegex(ScoutError, "regular local file"):
            neural._checkpoint(self.root)


class Device:
    type = "cuda"

    def __str__(self):
        return "cuda:0"


@unittest.skipIf(np is None, "NumPy unavailable; model is never installed by tests")
class MockInferenceTests(CheckpointTests):
    def setUp(self):
        super().setUp()
        self.audit_root = self.root / "audit"
        self.audit_root.mkdir()
        self.output = self.root / "model-output"
        self.plane = np.arange(64, dtype=np.uint8).reshape(8, 8)
        self.report = {"source": {"id": "section-3905", "crop_sha256": hashlib.sha256(self.plane.tobytes()).hexdigest()}}
        self.audit_bytes = json.dumps(self.report).encode()
        self.inputs = []
        self.model = SimpleNamespace(pretrained_model=str(self.path), device=Device(), net=SimpleNamespace(eval=Mock()))

        def evaluate(values, **options):
            self.inputs.append((values.copy(), options))
            mask = np.zeros(values.shape, dtype=np.int32)
            mask[0, 0] = 17
            return mask, [], np.zeros(1)

        self.model.eval = Mock(side_effect=evaluate)
        self.factory = Mock(return_value=self.model)
        self.models = SimpleNamespace(CellposeModel=self.factory, cache_CPSAM_model_path=Mock(side_effect=AssertionError("must not download")))
        self.torch = SimpleNamespace(
            device=lambda name: Device(), inference_mode=nullcontext, __version__="mock-no-model-execution",
            version=SimpleNamespace(cuda="mock"),
            cuda=SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
                                 max_memory_allocated=Mock(return_value=1024), max_memory_reserved=Mock(return_value=2048)),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)),
        )
        self.source_patch = patch.object(neural, "_audit_source", return_value=(self.audit_root, self.report, self.audit_bytes, self.plane))
        self.runtime_patch = patch.object(neural, "_runtime", return_value=(np, self.torch, self.models))
        self.source_patch.start()
        self.runtime_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.addCleanup(self.runtime_patch.stop)

    def run_mock(self):
        return neural.run_model(self.audit_root, self.path, self.output, allow_gpu=True)

    def test_mock_positive_preserves_fov_protocol_and_uint32_artifacts(self):
        report = self.run_mock()
        self.assertEqual([level["shape"] for level in report["levels"]], [[8, 8], [4, 4], [2, 2]])
        self.assertEqual([level["candidate_count"] for level in report["levels"]], [1, 1, 1])
        self.assertEqual([options["diameter"] for _, options in self.inputs], [30, 15, 7.5])
        for factor, (values, options) in zip(neural.RESOLUTIONS, self.inputs):
            expected = self.plane.reshape(8 // factor, factor, 8 // factor, factor).mean(axis=(1, 3)).astype(np.float32)
            np.testing.assert_array_equal(values, expected)
            self.assertEqual(values.dtype, np.float32)
            self.assertTrue(options["normalize"])
            self.assertFalse(options["invert"])
            self.assertEqual(options["flow_threshold"], 0.6)
            record = report["artifacts"][f"instances_{factor}um"]
            raw = (self.output / record["path"]).read_bytes()
            self.assertEqual(len(raw), values.size * 4)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
            self.assertEqual(np.frombuffer(raw, dtype="<u4")[0], 17)
        self.assertFalse(self.factory.call_args.kwargs["use_bfloat16"])
        self.assertFalse(report["accuracy_established"])
        self.assertFalse(report["training_executed"])
        self.assertIsNone(report["checkpoint"]["training_section_ids"])
        self.assertEqual(json.loads((self.output / "report.json").read_bytes()), report)
        self.assertEqual({entry.name for entry in self.output.iterdir()}, {"report.json", *neural.ARTIFACT_PATHS.values()})
        self.models.cache_CPSAM_model_path.assert_not_called()
        self.assertTrue(self.torch.backends.cuda.matmul.allow_tf32)

    def test_existing_output_refused_before_verification(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ScoutError, "refusing overwrite"):
            self.run_mock()
        neural._audit_source.assert_not_called()
        neural._runtime.assert_not_called()

    def test_invalid_audit_refuses_before_checkpoint_or_runtime(self):
        neural._audit_source.side_effect = ScoutError("invalid verified audit")
        with patch.object(neural, "_checkpoint") as checkpoint:
            with self.assertRaisesRegex(ScoutError, "invalid verified audit"):
                self.run_mock()
            checkpoint.assert_not_called()
        neural._runtime.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_bad_checkpoint_refuses_before_runtime(self):
        self.path.write_bytes(b"x" * len(self.data))
        with self.assertRaisesRegex(ScoutError, "SHA-256"):
            self.run_mock()
        neural._runtime.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_output_inside_audit_refused(self):
        self.output = self.audit_root / "nested"
        with self.assertRaisesRegex(ScoutError, "inside the verified audit"):
            self.run_mock()
        neural._runtime.assert_not_called()

    def test_model_silent_fallback_and_cpu_are_rejected(self):
        for field, value in (("pretrained_model", "cpsam"), ("device", SimpleNamespace(type="cpu"))):
            with self.subTest(field=field):
                old = getattr(self.model, field)
                setattr(self.model, field, value)
                with self.assertRaisesRegex(ScoutError, "differs from the trusted request"):
                    self.run_mock()
                setattr(self.model, field, old)
        self.model.eval.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_constructor_download_fallback_is_blocked(self):
        self.factory.side_effect = lambda **options: self.models.cache_CPSAM_model_path()
        with self.assertRaisesRegex(ScoutError, "download is disabled"):
            self.run_mock()
        self.assertFalse(self.output.exists())

    def test_bad_masks_refuse_before_output(self):
        for mask in (np.zeros((7, 8), dtype=np.uint32), np.zeros((8, 8), dtype=bool),
                     np.full((8, 8), -1, dtype=np.int32), np.full((8, 8), 2**32, dtype=np.uint64),
                     np.full((8, 8), np.nan), np.zeros((8, 8), dtype=np.float32)):
            with self.subTest(dtype=mask.dtype, shape=mask.shape):
                self.model.eval.side_effect = None
                self.model.eval.return_value = mask, [], []
                with self.assertRaisesRegex(ScoutError, "mask must match"):
                    self.run_mock()
                self.assertFalse(self.output.exists())

    def test_bad_return_and_runtime_errors_are_actionable(self):
        self.model.eval.side_effect = None
        self.model.eval.return_value = [np.zeros((8, 8), dtype=np.uint32)]
        with self.assertRaisesRegex(ScoutError, "return fields"):
            self.run_mock()
        self.model.eval.side_effect = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(ScoutError, "no CPU fallback"):
            self.run_mock()
        self.assertFalse(self.output.exists())

    def test_nonfinite_memory_refuses_before_output(self):
        self.torch.cuda.max_memory_allocated.return_value = float("nan")
        with self.assertRaisesRegex(ScoutError, "measurement"):
            self.run_mock()
        self.assertFalse(self.output.exists())

    def test_nonfinite_runtime_refuses_before_output(self):
        with patch.object(neural.time, "perf_counter", side_effect=[0.0, float("nan")]):
            with self.assertRaisesRegex(ScoutError, "measurement"):
                self.run_mock()
        self.assertFalse(self.output.exists())


@unittest.skipIf(np is None, "NumPy unavailable")
class AuditReadTests(unittest.TestCase):
    def test_source_read_is_bounded_and_checksum_checked(self):
        from bigbrain_scout import audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = bytes(range(64))
            report = {"source": {"width": 8, "height": 8, "crop_sha256": hashlib.sha256(raw).hexdigest()}}
            (root / "report.json").write_text(json.dumps(report))
            (root / "source.raw").write_bytes(raw)
            with patch.object(audit, "verify_audit", return_value=report):
                _, loaded, _, plane = neural._audit_source(root)
                self.assertEqual(loaded, report)
                self.assertEqual(plane.tobytes(), raw)
                (root / "source.raw").write_bytes(b"x" * 64)
                with self.assertRaisesRegex(ScoutError, "source changed"):
                    neural._audit_source(root)

    def test_report_change_rejected(self):
        from bigbrain_scout import audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "report.json").write_text('{}')
            with patch.object(audit, "verify_audit", return_value={"source": {}}):
                with self.assertRaisesRegex(ScoutError, "report changed"):
                    neural._audit_source(root)
