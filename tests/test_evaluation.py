"""Synthetic-only label-evaluation checks; no real reference annotations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bigbrain_scout import evaluation
from bigbrain_scout.core import ScoutError

try:
    import numpy as np
    import scipy
except ImportError:
    np = scipy = None


@unittest.skipIf(np is None or scipy is None, "NumPy/SciPy evaluation extras unavailable")
class InstanceEvaluationTests(unittest.TestCase):
    def evaluate(self, predicted, reference, **options):
        return evaluation.evaluate_instances(np.array(predicted, dtype=np.int32),
                                             np.array(reference, dtype=np.int32), **options)

    def test_perfect_instances_and_permuted_ids(self):
        result = self.evaluate([[0, 1, 1], [2, 2, 0]], [[0, 97, 97], [3, 3, 0]])
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(result["f1"], 1)
        self.assertEqual(result["absolute_count_error"], 0)
        self.assertEqual({pair["iou"] for pair in result["matches"]}, {1})
        permuted = self.evaluate([[0, 53, 53], [7, 7, 0]], [[0, 2, 2], [400, 400, 0]])
        for field in ("true_positives", "precision", "recall", "f1", "absolute_count_error"):
            self.assertEqual(result[field], permuted[field])

    def test_maximum_cardinality_not_greedy_iou(self):
        # IoU matrix [[0.5, 1/3], [1/3, 0]]. Greedy highest-IoU matching
        # would lose a true positive; the maximum qualifying matching has two.
        result = self.evaluate([[1, 1, 1, 1], [1, 1, 2, 2]],
                               [[10, 10, 10, 10], [11, 11, 10, 10]], iou_threshold=0.3)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual({(pair["prediction_id"], pair["reference_id"]) for pair in result["matches"]},
                         {(1, 11), (2, 10)})
        stricter = self.evaluate([[1, 1, 1, 1], [1, 1, 2, 2]],
                                 [[10, 10, 10, 10], [11, 11, 10, 10]], iou_threshold=0.5)
        self.assertEqual(stricter["true_positives"], 1)

    def test_threshold_is_inclusive_and_declared(self):
        result = self.evaluate([[1, 1]], [[3, 0]], iou_threshold=0.5)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["iou_threshold"], 0.5)
        self.assertEqual(result["matches"][0]["iou"], 0.5)
        self.assertEqual(self.evaluate([[1, 1]], [[3, 0]], iou_threshold=0.5001)["true_positives"], 0)

    def test_no_overlap_and_background_are_not_matches(self):
        result = self.evaluate([[1, 0]], [[0, 7]], iou_threshold=0.01)
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (0, 1, 1))
        self.assertEqual(result["f1"], 0)

    def test_split_and_merge_counts(self):
        split = self.evaluate([[1, 2]], [[3, 3]])
        self.assertEqual((split["true_positives"], split["false_positives"], split["false_negatives"]), (1, 1, 0))
        self.assertEqual(split["absolute_count_error"], 1)
        self.assertAlmostEqual(split["precision"], 0.5)
        self.assertAlmostEqual(split["recall"], 1)
        self.assertAlmostEqual(split["f1"], 2 / 3)
        merge = self.evaluate([[3, 3]], [[1, 2]])
        self.assertEqual((merge["true_positives"], merge["false_positives"], merge["false_negatives"]), (1, 0, 1))
        self.assertAlmostEqual(merge["precision"], 1)
        self.assertAlmostEqual(merge["recall"], 0.5)

    def test_zero_instance_semantics_are_explicit(self):
        for predicted, reference, expected in [
            ([[0]], [[0]], (1, 1, 1)),
            ([[0]], [[3]], (1, 0, 0)),
            ([[7]], [[0]], (0, 1, 0)),
        ]:
            with self.subTest(predicted=predicted, reference=reference):
                result = self.evaluate(predicted, reference)
                self.assertEqual(tuple(result[key] for key in ("precision", "recall", "f1")), expected)
                self.assertIn("both empty F1", result["empty_instance_semantics"])

    def test_sparse_large_ids_do_not_allocate_by_id(self):
        result = self.evaluate([[evaluation.MAX_LABEL_ID, 0]], [[17, 0]])
        self.assertEqual(result["true_positives"], 1)

    def test_noncontiguous_arrays(self):
        plane = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
        self.assertEqual(evaluation.evaluate_instances(plane[:, ::-1], plane[:, ::-1])["f1"], 1)

    def test_invalid_array_values(self):
        valid = np.zeros((1, 1), dtype=np.int32)
        for array in (np.array([[True]]), np.array([[1.0]]), np.array([[float("nan")]]),
                      np.array([[-1]]), np.array([[2**40]], dtype=np.uint64),
                      np.array([[1]], dtype=object), np.array([["1"]])):
            with self.subTest(dtype=array.dtype):
                with self.assertRaises(ScoutError):
                    evaluation.evaluate_instances(array, valid)

    def test_ndarray_subclass_and_arbitrary_conversion_are_refused(self):
        class Untrusted:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Conversion must not execute")

        with self.assertRaisesRegex(ScoutError, "ndarray"):
            evaluation.evaluate_instances(Untrusted(), np.zeros((1, 1), dtype=np.int32))

    def test_shapes_empty_and_dimensions(self):
        for array in (np.array([], dtype=np.int32), np.zeros((0, 2), dtype=np.int32),
                      np.zeros((1, 1, 1), dtype=np.int32)):
            with self.assertRaisesRegex(ScoutError, "2D"):
                evaluation.evaluate_instances(array, array)
        with self.assertRaisesRegex(ScoutError, "shapes"):
            self.evaluate([[1]], [[1, 1]])

    def test_plane_byte_and_object_limits(self):
        for array in (np.zeros((1, evaluation.MAX_PIXELS + 1), dtype=np.uint8),
                      np.zeros((1, evaluation.MAX_PLANE_BYTES // 8 + 1), dtype=np.int64)):
            with self.assertRaisesRegex(ScoutError, "1 MiB"):
                evaluation.evaluate_instances(array, array)
        too_many = np.arange(evaluation.MAX_OBJECTS + 2, dtype=np.int32)[None, :]
        with self.assertRaisesRegex(ScoutError, "instance limit"):
            evaluation.evaluate_instances(too_many, too_many)

    def test_exact_limits_accepted(self):
        plane = np.zeros((512, 512), dtype=np.uint32)
        self.assertEqual(evaluation.evaluate_instances(plane, plane)["f1"], 1)
        objects = np.arange(1, evaluation.MAX_OBJECTS + 1, dtype=np.int32)[None, :]
        self.assertEqual(evaluation.evaluate_instances(objects, objects)["true_positives"], evaluation.MAX_OBJECTS)

    def test_invalid_thresholds(self):
        for threshold in (True, 0, -0.1, 1.01, float("nan"), float("inf"), "0.5", None, 10**1000):
            with self.subTest(threshold=repr(threshold)[:50]):
                with self.assertRaises(ScoutError):
                    self.evaluate([[1]], [[1]], iou_threshold=threshold)


@unittest.skipIf(np is None or scipy is None, "NumPy/SciPy evaluation extras unavailable")
class LabelFileEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.prediction = self.root / "prediction.json"
        self.reference = self.root / "reference.json"
        self.document = {
            "schema_version": 1,
            "section_id": "synthetic-held-out-section",
            "coordinate_space": "NONBIOLOGICAL synthetic test coordinates",
            "pixel_spacing_um": [1, 1],
            "origin_um": [0, 0],
            "axis_directions": ["+Z", "+X"],
            "labels": [[1, 1, 0], [1, 0, 2], [0, 2, 2]],
            "labels_sha256": "",
            "provenance": {"type": "predicted", "independent_expert_reviewed": False,
                           "training_section_ids": ["synthetic-training-section"]},
        }
        self.expected = deepcopy(self.document)
        self.expected["provenance"] = {"type": "independent_expert_reviewed", "independent_expert_reviewed": True,
                                       "evaluation_split": "held_out_section",
                                       "description": "Synthetic fixture only; not genuine expert annotations."}
        self.save(self.prediction, self.document)
        self.save(self.reference, self.expected)

    def save(self, path, document, refresh=True):
        if refresh:
            labels = json.dumps(document["labels"], separators=(",", ":")).encode("ascii")
            document["labels_sha256"] = hashlib.sha256(labels).hexdigest()
        path.write_text(json.dumps(document), encoding="utf-8")

    def evaluate(self):
        return evaluation.evaluate_label_files(self.prediction, self.reference)

    def test_pair_reports_metrics_geometry_and_assertions_not_certification(self):
        before = set(self.root.iterdir())
        result = self.evaluate()
        self.assertEqual(result["f1"], 1)
        self.assertEqual(result["geometry"]["section_id"], self.document["section_id"])
        self.assertTrue(result["provenance_assertion_only"])
        self.assertIn("no measured segmentation accuracy", " ".join(result["limitations"]))
        self.assertIn("cannot certify", " ".join(result["limitations"]))
        self.assertEqual(result["file_sha256"]["prediction"], hashlib.sha256(self.prediction.read_bytes()).hexdigest())
        self.assertEqual(set(self.root.iterdir()), before)
        json.dumps(result, allow_nan=False)

    def test_tampered_labels_without_new_checksum_refused(self):
        self.document["labels"][0][0] = 99
        self.save(self.prediction, self.document, refresh=False)
        with self.assertRaisesRegex(ScoutError, "checksum"):
            self.evaluate()

    def test_bad_hash_format_refused(self):
        for digest in ("", "f" * 63, "G" * 64, 0, ["f" * 64]):
            self.document["labels_sha256"] = digest
            self.save(self.prediction, self.document, refresh=False)
            with self.assertRaisesRegex(ScoutError, "checksum"):
                self.evaluate()

    def test_geometry_mismatch_refused(self):
        for field, value in (("section_id", "another-section"), ("coordinate_space", "another-space"),
                             ("pixel_spacing_um", [2, 1]), ("origin_um", [1, 0]),
                             ("axis_directions", ["-Z", "+X"])):
            with self.subTest(field=field):
                mutated = deepcopy(self.expected)
                mutated[field] = value
                self.save(self.reference, mutated)
                with self.assertRaisesRegex(ScoutError, "must match exactly"):
                    self.evaluate()

    def test_declared_training_section_cannot_be_evaluation_section(self):
        self.document["provenance"]["training_section_ids"] = [self.expected["section_id"]]
        self.save(self.prediction, self.document)
        with self.assertRaisesRegex(ScoutError, "held-out evaluation refused"):
            self.evaluate()

    def test_untrained_baseline_empty_training_list_is_allowed(self):
        self.document["provenance"]["training_section_ids"] = []
        self.save(self.prediction, self.document)
        self.assertEqual(self.evaluate()["f1"], 1)

    def test_predicted_contours_not_reference_ground_truth(self):
        self.expected["provenance"] = deepcopy(self.document["provenance"])
        self.save(self.reference, self.expected)
        with self.assertRaisesRegex(ScoutError, "provenance"):
            self.evaluate()

    def test_review_and_split_require_explicit_values(self):
        for field, value in (("type", "cortical_layer_annotation"), ("independent_expert_reviewed", False),
                             ("independent_expert_reviewed", 1), ("evaluation_split", "random_tile")):
            mutated = deepcopy(self.expected)
            mutated["provenance"][field] = value
            self.save(self.reference, mutated)
            with self.assertRaisesRegex(ScoutError, "independent expert review"):
                self.evaluate()

    def test_training_section_schema(self):
        for sections in (None, "one-section", ["same", "same"], ["../section"], [False], ["a"] * 1001):
            self.document["provenance"]["training_section_ids"] = sections
            self.save(self.prediction, self.document)
            with self.assertRaises(ScoutError):
                self.evaluate()

    def test_schema_unknown_fields_and_versions_refused(self):
        for field, value in (("schema_version", "1"), ("schema_version", 1.0), ("schema_version", True),
                             ("run", "arbitrary.py")):
            mutated = deepcopy(self.document)
            mutated[field] = value
            self.save(self.prediction, mutated)
            with self.assertRaises(ScoutError):
                self.evaluate()
        mutated = deepcopy(self.document)
        del mutated["origin_um"]
        self.save(self.prediction, mutated)
        with self.assertRaisesRegex(ScoutError, "schema"):
            self.evaluate()

    def test_strict_nonfinite_duplicate_and_nonobject_json(self):
        for content in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e9999}', b'{"a":1,"a":2}',
                        b'[]', b'"text"', b'\xff'):
            self.prediction.write_bytes(content)
            with self.assertRaises(ScoutError):
                self.evaluate()

    def test_bad_label_rows(self):
        for labels in ([], [[]], [[0], [0, 1]], [0], [[True]], [[1.0]], [[-1]], [[2**40]], [["1"]]):
            self.document["labels"] = labels
            self.save(self.prediction, self.document)
            with self.assertRaises(ScoutError):
                self.evaluate()

    def test_file_and_plane_size_limits(self):
        self.prediction.write_bytes(b" " * (evaluation.MAX_FILE_BYTES + 1))
        with self.assertRaisesRegex(ScoutError, "4 MiB"):
            self.evaluate()
        self.document["labels"] = [[0] * (evaluation.MAX_PIXELS + 1)]
        self.save(self.prediction, self.document)
        with self.assertRaisesRegex(ScoutError, "1 MiB"):
            self.evaluate()

    def test_symlink_file_and_parent_refused(self):
        link = self.root / "linked.json"
        link.symlink_to(self.prediction)
        with self.assertRaisesRegex(ScoutError, "symlinks"):
            evaluation.evaluate_label_files(link, self.reference)
        parent = self.root / "linked-parent"
        parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ScoutError, "symlinks"):
            evaluation.evaluate_label_files(parent / "prediction.json", self.reference)

    def test_parent_traversal_missing_directory_and_non_json_paths(self):
        for path in (self.root / ".." / self.root.name / "prediction.json", self.root / "missing.json", self.root,
                     self.root / "labels.npy"):
            with self.assertRaises(ScoutError):
                evaluation.evaluate_label_files(path, self.reference)

    def test_geometry_validation(self):
        for field, value in (("pixel_spacing_um", [0, 1]), ("pixel_spacing_um", [True, 1]),
                             ("origin_um", [10**1000, 0]), ("origin_um", [0]),
                             ("axis_directions", ["+X", "-X"]), ("axis_directions", ["row", "column"]),
                             ("section_id", "../escape"), ("coordinate_space", "")):
            mutated = deepcopy(self.document)
            mutated[field] = value
            self.save(self.prediction, mutated)
            with self.assertRaises(ScoutError):
                self.evaluate()


class DependencyFreeImportTests(unittest.TestCase):
    def test_module_import_does_not_import_optional_numeric_packages(self):
        code = ("import sys; import bigbrain_scout.evaluation; "
                "assert 'numpy' not in sys.modules; assert 'scipy' not in sys.modules")
        completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
