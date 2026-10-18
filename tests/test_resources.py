"""Installed-style defaults have no checkout or current-directory dependency."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bigbrain_scout import resources
from bigbrain_scout.core import ScoutError

ROOT = Path(__file__).resolve().parent.parent


class ResourceTests(unittest.TestCase):
    def test_bundled_examples_match_checkout_examples(self):
        for name, relative in (("safe", "examples/safe-request.json"),
                               ("unsafe", "examples/unsafe-request.json"),
                               ("fixture", "fixtures/synthetic-section.json")):
            self.assertEqual(resources.load(name), json.loads((ROOT / relative).read_text()))

    def test_bundled_defaults_are_independent(self):
        first = resources.load("safe")
        first["roi_mm"][0][0] = 999
        self.assertEqual(resources.load("safe")["roi_mm"][0][0], 0.002)

    def test_arbitrary_resource_paths_refused(self):
        for value in ("../private", "/etc/passwd", None, []):
            with self.subTest(value=value), self.assertRaises(ScoutError):
                resources.resource(value)

    def test_bundled_cli_example_success_and_refusal(self):
        safe = subprocess.run([sys.executable, "-m", "bigbrain_scout", "plan", "--example", "safe"],
                              cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(safe.returncode, 0, safe.stderr)
        self.assertEqual(json.loads(safe.stdout)["shape"], [2, 3, 1])
        unsafe = subprocess.run([sys.executable, "-m", "bigbrain_scout", "plan", "--example", "unsafe"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(unsafe.returncode, 2)
        self.assertIn("Decoded ROI", unsafe.stderr)

    def test_demo_defaults_are_bundled(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capsule"
            result = subprocess.run([sys.executable, "-m", "bigbrain_scout", "demo", "--output", str(output)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["fixture_type"], "synthetic")
            self.assertTrue((output / "fixture.json").is_file())


if __name__ == "__main__":
    unittest.main()
