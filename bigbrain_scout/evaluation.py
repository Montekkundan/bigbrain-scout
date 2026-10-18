"""Bounded instance-overlap evaluation, not a source of reference annotations.

The current real Scout crop has no independent instance labels. This module only
evaluates labels supplied by a caller. JSON provenance, review and section-split
declarations are assertions, not proof of expertise, independence or validity.
NumPy and SciPy are imported only when evaluation is requested.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re

from .core import ScoutError

MAX_PLANE_BYTES = 1024**2
MAX_PIXELS = MAX_PLANE_BYTES // 4
MAX_OBJECTS = 1000
MAX_LABEL_ID = 2**31 - 1
MAX_FILE_BYTES = 4 * 1024**2
LABEL_FIELDS = {
    "schema_version", "section_id", "coordinate_space", "pixel_spacing_um",
    "origin_um", "axis_directions", "labels", "labels_sha256", "provenance",
}
LIMITATIONS = [
    "The current unlabeled real crop has no measured segmentation accuracy.",
    "Caller-supplied provenance cannot certify independent expert review or a true held-out split.",
    "Layer annotations and predicted contours are not independent instance reference labels.",
    "One supplied section pair does not establish generalization to other sections or brains.",
    "Hashes check local label consistency, not annotation authenticity or scientific validity.",
]


def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ScoutError("IoU threshold must be a finite number in (0, 1]")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ScoutError("IoU threshold must be a finite number in (0, 1]") from exc
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ScoutError("IoU threshold must be a finite number in (0, 1]")
    return value


def _array(value, name):
    import numpy as np

    # ndarray inputs cannot trigger caller-provided conversion methods or ragged
    # list allocation. The file adapter validates and converts JSON lists first.
    if type(value) is not np.ndarray:
        raise ScoutError(f"{name} must be a NumPy ndarray")
    if value.ndim != 2 or any(size <= 0 for size in value.shape):
        raise ScoutError(f"{name} must be a nonempty 2D label plane")
    if value.dtype.kind not in {"u", "i"}:
        raise ScoutError(f"{name} must contain nonnegative integer instance IDs, not floats or booleans")
    if value.size > MAX_PIXELS or value.nbytes > MAX_PLANE_BYTES:
        raise ScoutError(f"{name} exceeds the 1 MiB label-plane limit")
    if value.min() < 0 or value.max() > MAX_LABEL_ID:
        raise ScoutError(f"{name} instance IDs must be in [0, {MAX_LABEL_ID}]")
    ids = np.unique(value)
    if np.count_nonzero(ids) > MAX_OBJECTS:
        raise ScoutError(f"{name} exceeds the {MAX_OBJECTS}-instance limit")
    return value


def evaluate_instances(predicted_labels, reference_labels, *, iou_threshold=0.5):
    """Match foreground IDs one-to-one at a declared IoU threshold.

    ID 0 is background. Matching maximizes the number of qualifying pairs, not
    overlap sum or greedy order. Empty denominators have vacuous value 1: two
    empty planes score precision/recall/F1 1; just one empty side scores F1 0.
    This low-level function has no information about geometry or label origins.
    """
    import numpy as np

    threshold = _threshold(iou_threshold)
    predicted = _array(predicted_labels, "Prediction")
    reference = _array(reference_labels, "Reference")
    if predicted.shape != reference.shape:
        raise ScoutError("Prediction and reference label-plane shapes must match")
    predicted_ids, predicted_inverse = np.unique(predicted, return_inverse=True)
    reference_ids, reference_inverse = np.unique(reference, return_inverse=True)
    predicted_foreground = predicted_ids != 0
    reference_foreground = reference_ids != 0
    predicted_count = int(predicted_foreground.sum())
    reference_count = int(reference_foreground.sum())
    matches = []
    if predicted_count and reference_count:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching

        intersections = np.bincount(
            predicted_inverse.ravel() * len(reference_ids) + reference_inverse.ravel(),
            minlength=len(predicted_ids) * len(reference_ids),
        ).reshape(len(predicted_ids), len(reference_ids))
        predicted_areas = intersections.sum(axis=1)[predicted_foreground]
        reference_areas = intersections.sum(axis=0)[reference_foreground]
        overlap = intersections[predicted_foreground][:, reference_foreground]
        union = predicted_areas[:, None] + reference_areas[None, :] - overlap
        iou = overlap / union
        # The sparse graph represents only qualifying edges. A maximum matching
        # cannot sacrifice a true positive for a larger individual overlap.
        assignment = maximum_bipartite_matching(csr_matrix(iou >= threshold), perm_type="column")
        prediction_objects = predicted_ids[predicted_foreground]
        reference_objects = reference_ids[reference_foreground]
        for prediction_index, reference_index in enumerate(assignment):
            if reference_index >= 0:
                matches.append({
                    "prediction_id": int(prediction_objects[prediction_index]),
                    "reference_id": int(reference_objects[reference_index]),
                    "iou": float(iou[prediction_index, reference_index]),
                })
    true_positives = len(matches)
    precision = true_positives / predicted_count if predicted_count else 1.0
    recall = true_positives / reference_count if reference_count else 1.0
    denominator = predicted_count + reference_count
    f1 = 2 * true_positives / denominator if denominator else 1.0
    return {
        "schema_version": 1,
        "evaluation_kind": "instance_overlap_against_supplied_reference",
        "iou_threshold": threshold,
        "matching": "one_to_one_maximum_cardinality_at_or_above_threshold",
        "shape": list(predicted.shape),
        "predicted_count": predicted_count,
        "reference_count": reference_count,
        "true_positives": true_positives,
        "false_positives": predicted_count - true_positives,
        "false_negatives": reference_count - true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "absolute_count_error": abs(predicted_count - reference_count),
        "matches": matches,
        "empty_instance_semantics": "Empty precision/recall denominators equal 1; both empty F1 equals 1.",
        "limitations": list(LIMITATIONS),
    }


def _read(path):
    path = Path(path)
    if ".." in path.parts:
        raise ScoutError("Label paths must not contain parent traversal")
    path = path.absolute()
    if any(entry.is_symlink() for entry in (path, *path.parents)):
        raise ScoutError("Label paths must not contain symlinks")
    if path.suffix.lower() != ".json" or not path.is_file():
        raise ScoutError("Labels must be existing regular JSON files")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ScoutError("Label JSON exceeds the 4 MiB file limit")
    # Limit the read itself even if the file grows between stat and open.
    with path.open("rb") as handle:
        content = handle.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ScoutError("Label JSON exceeds the 4 MiB file limit")

    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Nonfinite number")
        return value

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(content, parse_float=finite, object_pairs_hook=unique,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ScoutError("Labels must contain strict, finite, unambiguous JSON") from exc
    if not isinstance(value, dict) or set(value) != LABEL_FIELDS:
        raise ScoutError("Label JSON fields must match the version-1 schema")
    return value, hashlib.sha256(content).hexdigest()


def _text(value, name, maximum=160):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise ScoutError(f"{name} must be a bounded, nonempty text value")
    return value


def _section(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value) is None:
        raise ScoutError("Section ID must be bounded plain text without paths")
    return value


def _vector(value, name, positive=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ScoutError(f"{name} must contain exactly two finite numbers")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ScoutError(f"{name} must contain exactly two finite numbers")
        if abs(item) > 1_000_000 or not math.isfinite(item) or (positive and item <= 0):
            raise ScoutError(f"{name} contains an out-of-range value")
    return [float(item) for item in value]


def _document(value, role):
    import numpy as np

    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ScoutError("Unsupported label schema version")
    geometry = {
        "section_id": _section(value["section_id"]),
        "coordinate_space": _text(value["coordinate_space"], "Coordinate space"),
        "pixel_spacing_um": _vector(value["pixel_spacing_um"], "Pixel spacing", positive=True),
        "origin_um": _vector(value["origin_um"], "Origin"),
    }
    axes = value["axis_directions"]
    allowed_axes = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    if (not isinstance(axes, list) or len(axes) != 2
            or any(not isinstance(axis, str) or axis not in allowed_axes for axis in axes)
            or axes[0][-1] == axes[1][-1]):
        raise ScoutError("Axis directions must identify two distinct signed physical axes in row/column order")
    geometry["axis_directions"] = axes[:]
    rows = value["labels"]
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], list) or not rows[0]:
        raise ScoutError("Labels must contain a nonempty rectangular 2D array")
    width = len(rows[0])
    if len(rows) * width > MAX_PIXELS:
        raise ScoutError("Labels exceed the 1 MiB label-plane limit")
    for row in rows:
        if not isinstance(row, list) or len(row) != width:
            raise ScoutError("Labels must contain a nonempty rectangular 2D array")
        if any(type(item) is not int or not 0 <= item <= MAX_LABEL_ID for item in row):
            raise ScoutError("Labels must contain bounded nonnegative integer IDs, not floats or booleans")
    canonical = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode("ascii")
    digest = value["labels_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None or hashlib.sha256(canonical).hexdigest() != digest:
        raise ScoutError("Label checksum mismatch")
    provenance = value["provenance"]
    required = {"type", "independent_expert_reviewed"}
    role_field = "training_section_ids" if role == "prediction" else "evaluation_split"
    if (not isinstance(provenance, dict) or not (required | {role_field}) <= set(provenance)
            or not set(provenance) <= (required | {role_field, "description"})):
        raise ScoutError("Label provenance must match its prediction/reference role")
    if "description" in provenance:
        _text(provenance["description"], "Provenance description", maximum=1000)
    if role == "prediction":
        if provenance["type"] != "predicted" or provenance["independent_expert_reviewed"] is not False:
            raise ScoutError("Prediction provenance must declare predicted labels")
        sections = provenance["training_section_ids"]
        if not isinstance(sections, list) or len(sections) > 1000:
            raise ScoutError("Declared training section IDs must be a bounded list")
        for section in sections:
            _section(section)
        if len(set(sections)) != len(sections):
            raise ScoutError("Declared training section IDs must not repeat")
    elif (provenance["type"] != "independent_expert_reviewed"
          or provenance["independent_expert_reviewed"] is not True
          or provenance["evaluation_split"] != "held_out_section"):
        raise ScoutError("Reference labels must explicitly assert independent expert review and a held-out section")
    return _array(np.asarray(rows, dtype=np.uint32), role.title()), geometry, provenance


def evaluate_label_files(prediction, reference, *, iou_threshold=0.5):
    """Read two bounded label JSON files and evaluate only compatible geometry.

    Label checksums are SHA-256 of ``json.dumps(rows, separators=(',', ':'))``
    encoded as ASCII. Prediction training sections must be completely declared,
    even for a nontrained baseline (an empty list). An empty list must not be used
    for a checkpoint with unknown training sections: it means genuinely untrained,
    not unknown. References must assert independent expert
    review and a held-out section. These declarations cannot be authenticated by
    this local tool. No files are written and no real reference labels are made.
    """
    prediction_document, prediction_hash = _read(prediction)
    reference_document, reference_hash = _read(reference)
    predicted, prediction_geometry, prediction_provenance = _document(prediction_document, "prediction")
    expected, reference_geometry, reference_provenance = _document(reference_document, "reference")
    if prediction_geometry != reference_geometry:
        raise ScoutError("Prediction and reference section, spacing, origin and axes must match exactly")
    if reference_geometry["section_id"] in prediction_provenance["training_section_ids"]:
        raise ScoutError("Evaluation section appears in the declared training sections; held-out evaluation refused")
    result = evaluate_instances(predicted, expected, iou_threshold=iou_threshold)
    result.update(
        geometry=prediction_geometry,
        file_sha256={"prediction": prediction_hash, "reference": reference_hash},
        labels_sha256={"prediction": prediction_document["labels_sha256"], "reference": reference_document["labels_sha256"]},
        provenance={"prediction": prediction_provenance.copy(), "reference": reference_provenance.copy()},
        provenance_assertion_only=True,
    )
    return result
