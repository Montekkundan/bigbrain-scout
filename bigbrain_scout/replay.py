"""Trusted local capsule replay and decoded-value comparison.

Capsule-provided Python and Markdown are never executed. Verification establishes
local consistency, not scientific validity, authenticity, or redistribution rights.
Geometry uses 1e-5 mm absolute tolerance and zero relative tolerance; uint8 data
values must match exactly. File hashes and captured metadata are reported apart.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile

from . import core, resources

MAX_BYTES = 8 * 1024**2
MAX_MANIFEST_BYTES = 256 * 1024
MAX_METADATA_BYTES = 2 * 1024**2
MAX_TOTAL_BYTES = 4 * MAX_BYTES
GEOMETRY_ATOL_MM = 1e-5
SYNTHETIC_ARTIFACTS = {"data": "data.raw", "preview": "preview.pgm",
                       "fixture": "fixture.json", "reproduce": "reproduce.py"}
SOURCE_IDENTITY_FIELDS = ("id", "url", "coordinate_space", "units")
LIMITATIONS = [
    "Local consistency checks do not establish scientific validity or source authenticity.",
    "Exact decoded-value equality does not prove an independent network re-download.",
    "Network byte counts are unknown; decoded/chunk estimates are not a peak-memory guarantee.",
    "File/header, environment and metadata differences are not automatically image-value differences.",
]


class ReplayMismatch(core.ScoutError):
    """A verified replay differs; its output remains available for inspection."""

    def __init__(self, output, comparison):
        super().__init__("Replay differs from the verified source capsule; output was preserved")
        self.output = str(output)
        self.comparison = deepcopy(comparison)


def _path(value):
    path = Path(value).absolute()
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise core.ScoutError("Capsule, output and cache paths must not contain symlinks")
    return path.resolve()


def _read(path, maximum):
    _path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise core.ScoutError("Capsule file is missing or exceeds its size limit")
    content = path.read_bytes()
    if len(content) > maximum:
        raise core.ScoutError("Capsule file grew beyond its size limit")
    return content


def _json(content):
    def finite(text):
        result = float(text)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON number")
        return result

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=unique, parse_float=finite,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise core.ScoutError("Capsule must contain strict, finite, unambiguous JSON") from exc
    if not isinstance(value, dict):
        raise core.ScoutError("Capsule JSON must be an object")
    return value


def _live():
    return importlib.import_module("bigbrain_scout.live")


def _live_source(identifier):
    adapter = _live()
    if not isinstance(identifier, str) or identifier not in adapter.source_ids():
        raise core.ScoutError("Unknown live source; capsule URLs are not accepted")
    return adapter, adapter.catalog(identifier)


def _limits(manifest):
    core._required(manifest, core.MANIFEST_REQUIRED, "manifest")
    if set(manifest) != core.MANIFEST_REQUIRED:
        raise core.ScoutError("Unknown manifest fields in the v1 replay schema")
    core._required(manifest["source"], set(SOURCE_IDENTITY_FIELDS), "source")
    core._required(manifest["request"], {"bbox_mm", "resolution_key", "max_decoded_bytes", "max_estimated_working_bytes"}, "request")
    core._bbox(manifest["request"]["bbox_mm"], "request bbox")
    for row in manifest["request"]["bbox_mm"]:
        if any(abs(value) > 1000 for value in row):
            raise core.ScoutError("Replay bounds must remain within +/-1000 mm")
    for key in ("max_decoded_bytes", "max_estimated_working_bytes"):
        if core._budget(manifest["request"][key], key) > MAX_BYTES:
            raise core.ScoutError("Replay budgets must be at most 8 MiB each")
    core._required(manifest["returned"], {"shape", "dtype"}, "returned")
    shape = core._vector(manifest["returned"]["shape"], "returned shape", positive=True, integer=True)
    if manifest["returned"]["dtype"] != "uint8" or math.prod(shape) > MAX_BYTES:
        raise core.ScoutError("Replay supports bounded uint8 decoded arrays only")


def _synthetic_request(manifest, fixture):
    core._required(fixture, {"fixture_type", "source", "shape", "values"}, "fixture")
    if fixture["fixture_type"] != "synthetic":
        raise core.ScoutError("Expected a preserved NONBIOLOGICAL fixture")
    trusted = resources.load("fixture")["source"]
    if fixture["source"] != trusted:
        raise core.ScoutError("Unknown synthetic source metadata")
    expected_source = {key: trusted[key] for key in manifest["source"] if key in trusted}
    if manifest["source"] != expected_source:
        raise core.ScoutError("Synthetic source identity differs from the preserved fixture")
    request = manifest["request"]
    return {"source": deepcopy(fixture["source"]), "roi_mm": deepcopy(request["bbox_mm"]),
            "scale": request["resolution_key"], "max_decoded_bytes": request["max_decoded_bytes"],
            "max_estimated_working_bytes": request["max_estimated_working_bytes"]}


def _inspect(capsule):
    """Return bounded, verified snapshots; no capsule code is evaluated."""
    root = _path(capsule)
    if not root.is_dir():
        raise core.ScoutError("Capsule directory does not exist")
    manifest_bytes = _read(root / "manifest.json", MAX_MANIFEST_BYTES)
    manifest = _json(manifest_bytes)
    try:
        _limits(manifest)
        schema, fixture_type = manifest["schema_version"], manifest["fixture_type"]
        if schema == "1.0" and fixture_type == "synthetic":
            names, adapter = SYNTHETIC_ARTIFACTS, None
        elif schema == "live-1.0" and fixture_type == "real_remote_data":
            adapter, descriptor = _live_source(manifest["source"]["id"])
            if manifest["source"]["url"] != descriptor["source_url"]:
                raise core.ScoutError("Live source URL differs from its trusted catalog entry")
            names = adapter.artifact_names(manifest["source"]["id"])
        else:
            raise core.ScoutError("Unknown source/schema combination; replay refused")
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(names):
            raise core.ScoutError("Capsule artifacts must match the trusted filename contract")
        contents, total = {}, 0
        for role, name in names.items():
            record = artifacts[role]
            core._required(record, {"path", "bytes", "sha256"}, "artifact")
            if set(record) != {"path", "bytes", "sha256"} or record["path"] != name:
                raise core.ScoutError("Unexpected artifact fields or filename")
            maximum = MAX_BYTES + 4096 if role in {"data", "preview"} else MAX_METADATA_BYTES
            size = core._budget(record["bytes"], "artifact size")
            if size > maximum:
                raise core.ScoutError("Recorded artifact exceeds its replay size limit")
            content = _read(root / name, maximum)
            if len(content) != size or hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise core.ScoutError("Capsule artifact checksum or size mismatch")
            total += len(content)
            if total > MAX_TOTAL_BYTES:
                raise core.ScoutError("Capsule exceeds its total replay size limit")
            contents[role] = content
        if adapter is not None:
            _json(contents["metadata"])
        verified = core.verify_capsule(root) if adapter is None else adapter.verify(root)
        if verified != manifest:
            raise core.ScoutError("Capsule manifest changed during verification")
        if adapter is None:
            fixture = _json(contents["fixture"])
            request = _synthetic_request(manifest, fixture)
            planned = core.plan(request)
            if fixture["shape"] != planned["source_shape"] or manifest["returned"]["indices"] != planned["indices"]:
                raise core.ScoutError("Preserved fixture/request grid differs from recorded geometry")
            if (manifest["returned"]["shape"] != planned["shape"]
                    or manifest["returned"]["bbox_mm"] != planned["returned_bbox_mm"]
                    or manifest["returned"]["clipped_bbox_mm"] != planned["clipped_bbox_mm"]):
                raise core.ScoutError("Synthetic returned geometry differs from the trusted plan")
            data = core._extract(fixture, planned["indices"])
            if contents["data"] != data:
                raise core.ScoutError("Synthetic data differs from the preserved fixture request")
            shape = planned["shape"]
            preview = f"P5\n# NONBIOLOGICAL synthetic fixture, first z plane\n{shape[0]} {shape[1]}\n255\n".encode() + data[:shape[0] * shape[1]]
            if contents["preview"] != preview:
                raise core.ScoutError("Synthetic preview differs from the preserved fixture data")
        return root, deepcopy(manifest), contents, manifest_bytes
    except core.ScoutError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError) as exc:
        raise core.ScoutError("Capsule does not satisfy the bounded replay contract") from exc


def verify_capsule(capsule):
    """Dispatch strict local verification for known synthetic/live capsule schemas."""
    return _inspect(capsule)[1]


def _close(first, second):
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(_close(a, b) for a, b in zip(first, second))
    if isinstance(first, bool) or isinstance(second, bool):
        return first is second
    return isinstance(first, (int, float)) and isinstance(second, (int, float)) and math.isclose(first, second, rel_tol=0, abs_tol=GEOMETRY_ATOL_MM)


def _decoded(manifest, contents):
    if manifest["fixture_type"] == "synthetic":
        return contents["data"], hashlib.sha256(contents["data"]).hexdigest()
    try:
        import numpy as np
        import nibabel as nib
        image = nib.Nifti1Image.from_bytes(contents["data"])
        array = np.asanyarray(image.dataobj)
    except (ImportError, ValueError, OSError) as exc:
        raise core.ScoutError("Cannot decode the verified NIfTI snapshot") from exc
    if array.dtype != np.uint8 or list(array.shape) != manifest["returned"]["shape"] or array.nbytes > MAX_BYTES:
        raise core.ScoutError("Decoded NIfTI snapshot violates its verified uint8 shape/budget")
    return array, hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _snapshots(manifest, contents):
    if manifest["fixture_type"] == "synthetic":
        source = _json(contents["fixture"])["source"]
        return {"source_metadata_sha256": hashlib.sha256(core._json_bytes(source)).hexdigest()}
    return manifest["provenance"]["metadata_snapshots"]


def compare_capsules(first, second):
    """Compare verified geometry and uint8 values, independently of file headers."""
    _, a, files_a, manifest_a = _inspect(first)
    _, b, files_b, manifest_b = _inspect(second)
    same_kind = a["fixture_type"] == b["fixture_type"]
    source_match = all(a["source"].get(key) == b["source"].get(key) for key in SOURCE_IDENTITY_FIELDS)
    request_match = a["request"] == b["request"]
    fields = ("affine", "bbox_mm", "resolution_mm", "clipped_bbox_mm")
    geometry_match = (source_match and a["returned"]["shape"] == b["returned"]["shape"]
                      and a["returned"]["dtype"] == b["returned"]["dtype"]
                      and all(_close(a["returned"][key], b["returned"][key]) for key in fields))
    values_a, decoded_a = _decoded(a, files_a)
    values_b, decoded_b = _decoded(b, files_b)
    data_equal = False
    if same_kind and a["returned"]["shape"] == b["returned"]["shape"]:
        if a["fixture_type"] == "synthetic":
            data_equal = values_a == values_b and decoded_a == decoded_b
        else:
            import numpy as np
            data_equal = bool(np.array_equal(values_a, values_b))
    differences = {}
    for role in sorted(set(a["artifacts"]) | set(b["artifacts"])):
        left, right = a["artifacts"].get(role), b["artifacts"].get(role)
        if left != right:
            differences[role] = {"first": deepcopy(left), "second": deepcopy(right)}
    return {
        "matched": source_match and request_match and geometry_match and data_equal,
        "source_identity_match": source_match, "source_details_match": a["source"] == b["source"],
        "request_match": request_match, "geometry_match": geometry_match,
        "geometry_absolute_tolerance_mm": GEOMETRY_ATOL_MM, "geometry_relative_tolerance": 0,
        "data_comparable": same_kind, "data_bytes_equal": data_equal,
        "decoded_data_sha256": {"first": decoded_a, "second": decoded_b},
        "metadata_snapshots_match": _snapshots(a, files_a) == _snapshots(b, files_b),
        "environment_match": a["provenance"]["environment"] == b["provenance"]["environment"],
        "artifact_hashes_match": not differences, "file_hash_differences": differences,
        "manifest_file_hash_match": hashlib.sha256(manifest_a).digest() == hashlib.sha256(manifest_b).digest(),
        "match_definition": "Known source identity, exact request, geometry within tolerance, and exact decoded uint8 values. Metadata/environment/file hashes are separate signals.",
        "network_bytes": None, "independent_download_proven": False, "limitations": deepcopy(LIMITATIONS),
    }


def reproduce(capsule, output, *, allow_live=False, cache_dir=None):
    """Replay through trusted package code to a new directory, never capsule code.

    Live replay needs a process that has not imported siibra. Its singleton cache
    location cannot safely be reset here. A newly created default cache directory
    is preserved locally and reported, but is not proof of network independence.
    """
    _, manifest, contents, _ = _inspect(capsule)
    destination = _path(output)
    if destination.exists():
        raise core.ScoutError("Replay output already exists; refusing overwrite")
    cache_report = None
    if manifest["fixture_type"] == "synthetic":
        if cache_dir is not None:
            raise core.ScoutError("Offline replay does not use a live cache directory")
        fixture = _json(contents["fixture"])
        request = _synthetic_request(manifest, fixture)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="scout-offline-replay-") as temporary:
            request_file, fixture_file = Path(temporary) / "request.json", Path(temporary) / "fixture.json"
            request_file.write_bytes(core._json_bytes(request))
            fixture_file.write_bytes(core._json_bytes(fixture))
            core.create_demo(request_file, fixture_file, destination)
    else:
        if allow_live is not True:
            raise core.ScoutError("Live replay requires explicit allow_live=True")
        if "siibra" in sys.modules:
            raise core.ScoutError("Live replay requires a fresh process: siibra already imported, so a fresh singleton cache cannot be guaranteed")
        adapter, _ = _live_source(manifest["source"]["id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir is None:
            cache = Path(tempfile.mkdtemp(prefix="scout-live-replay-cache-", dir=destination.parent))
            fresh = True
        else:
            cache = _path(cache_dir)
            fresh = not cache.exists()
            if cache == destination or cache.is_relative_to(destination) or destination.is_relative_to(cache):
                raise core.ScoutError("Replay output and live cache directories must not overlap")
            if cache.exists() and not cache.is_dir():
                raise core.ScoutError("Live cache must be a directory")
            cache.mkdir(parents=True, exist_ok=True)
            if not cache.is_dir():
                raise core.ScoutError("Live cache must be a directory")
        cache_report = {"path": str(cache), "fresh_directory": fresh, "explicit_cache": cache_dir is not None,
                        "independent_download_proven": False, "note": "Fresh cache path is not proof that every byte was independently downloaded"}
        fields = manifest["request"]
        request = {"source_id": manifest["source"]["id"], "roi_mm": deepcopy(fields["bbox_mm"]),
                   "scale": fields["resolution_key"], "max_decoded_bytes": fields["max_decoded_bytes"],
                   "max_estimated_working_bytes": fields["max_estimated_working_bytes"]}
        context, planned = adapter.prepare(request, cache)
        adapter.fetch(context, planned, destination)
    verified = verify_capsule(destination)
    comparison = compare_capsules(capsule, destination)
    if not comparison["matched"]:
        raise ReplayMismatch(destination, comparison)
    return {"output": str(destination), "manifest": verified, "comparison": comparison,
            "cache": cache_report, "network_bytes": None, "independent_download_proven": False,
            "limitations": deepcopy(LIMITATIONS)}
