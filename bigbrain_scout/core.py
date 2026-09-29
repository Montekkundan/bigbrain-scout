"""Metadata-only planning and a tiny, explicitly nonbiological offline capsule.

Byte estimates describe decoded ROI and a conservative chunk assembly model.
They are neither network-transfer estimates nor a peak-memory guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path, PurePosixPath

from . import __version__

DTYPE_BYTES = {"uint8": 1, "uint16": 2, "float32": 4, "float64": 8}
MANIFEST_REQUIRED = {"schema_version", "fixture_type", "source", "request", "returned", "provenance", "artifacts"}
SOURCE_REQUIRED = {"id", "url", "license", "citation", "fixture_type", "coordinate_space", "units", "bbox_mm", "scales"}


class ScoutError(ValueError):
    """A request or capsule violates the prototype's explicit safety contract."""


def load_json(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ScoutError(f"Cannot read JSON {path}: {exc}") from exc


def _number(value, name, *, positive=False):
    try:
        finite = math.isfinite(value) if isinstance(value, (int, float)) else False
    except OverflowError:
        finite = False
    if isinstance(value, bool) or not finite:
        raise ScoutError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ScoutError(f"{name} must be positive")
    return value


def _vector(value, name, *, positive=False, integer=False):
    if not isinstance(value, list) or len(value) != 3:
        raise ScoutError(f"{name} must have three coordinates")
    for item in value:
        _number(item, name, positive=positive)
        if integer and not isinstance(item, int):
            raise ScoutError(f"{name} must contain integers")
    return value


def _bbox(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ScoutError(f"{name} must contain lower and upper coordinates")
    lower, upper = (_vector(value[0], name), _vector(value[1], name))
    if any(a >= b for a, b in zip(lower, upper)):
        raise ScoutError(f"{name} must have strictly positive extents")
    return lower, upper


def _budget(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScoutError(f"{name} must be a positive integer byte count")
    return value


def _required(mapping, keys, name):
    if not isinstance(mapping, dict) or not keys.issubset(mapping):
        raise ScoutError(f"{name} is missing required fields: {', '.join(sorted(keys))}")


def plan(request):
    """Plan an axis-aligned ROI without fetching bytes or choosing a fallback scale."""
    _required(request, {"source", "roi_mm", "scale", "max_decoded_bytes", "max_estimated_working_bytes"}, "request")
    source = request["source"]
    _required(source, SOURCE_REQUIRED, "source")
    for key in ("id", "url", "license", "citation", "fixture_type", "coordinate_space"):
        if not isinstance(source[key], str) or not source[key].strip():
            raise ScoutError(f"source.{key} must be a nonempty string")
    if source["units"] != "mm":
        raise ScoutError("Planner bounds and resolution require explicit millimetre units")
    source_lo, source_hi = _bbox(source["bbox_mm"], "source.bbox_mm")
    requested_lo, requested_hi = _bbox(request["roi_mm"], "roi_mm")
    if not isinstance(source["scales"], list) or not isinstance(request["scale"], str):
        raise ScoutError("scales must be a list and scale must be an exact string key")
    scales = [item for item in source["scales"] if isinstance(item, dict) and item.get("key") == request["scale"]]
    if len(scales) != 1:
        raise ScoutError(f"Exact scale {request['scale']!r} unavailable or ambiguous; no automatic fallback")
    scale = scales[0]
    _required(scale, {"resolution_mm", "dtype", "chunk_shape"}, "scale")
    resolution = _vector(scale["resolution_mm"], "resolution_mm", positive=True)
    chunks = _vector(scale["chunk_shape"], "chunk_shape", positive=True, integer=True)
    if not isinstance(scale["dtype"], str) or scale["dtype"] not in DTYPE_BYTES:
        raise ScoutError("Unsupported dtype; supply an explicit supported decoded representation")
    source_shape = []
    for lo, hi, step in zip(source_lo, source_hi, resolution):
        count = (hi - lo) / step
        _number(count, "source grid extent", positive=True)
        if not math.isclose(count, round(count), rel_tol=1e-10, abs_tol=1e-10) or count < 1:
            raise ScoutError("Source bounds must describe an integral voxel grid at this scale")
        source_shape.append(round(count))
    lo = [max(a, b) for a, b in zip(requested_lo, source_lo)]
    hi = [min(a, b) for a, b in zip(requested_hi, source_hi)]
    if any(a >= b for a, b in zip(lo, hi)):
        raise ScoutError("ROI does not intersect the source bounds")
    starts = [max(0, math.floor((a - b) / r + 1e-10)) for a, b, r in zip(lo, source_lo, resolution)]
    stops = [min(n, math.ceil((a - b) / r - 1e-10)) for a, b, r, n in zip(hi, source_lo, resolution, source_shape)]
    shape = [b - a for a, b in zip(starts, stops)]
    if any(n <= 0 for n in shape):
        raise ScoutError("ROI is too small for numerical grid precision")
    item_bytes = DTYPE_BYTES[scale["dtype"]]
    decoded = math.prod(shape) * item_bytes
    touched = [(b - 1) // c - a // c + 1 for a, b, c in zip(starts, stops, chunks)]
    one_chunk = math.prod(chunks) * item_bytes
    mosaic = math.prod(touched) * one_chunk
    working = decoded + mosaic + one_chunk
    if decoded > _budget(request["max_decoded_bytes"], "max_decoded_bytes"):
        raise ScoutError(f"Decoded ROI estimate {decoded} exceeds the request budget")
    if working > _budget(request["max_estimated_working_bytes"], "max_estimated_working_bytes"):
        raise ScoutError(f"Chunk assembly estimate {working} exceeds the request budget")
    returned_bbox = [[o + i * r for o, i, r in zip(source_lo, indices, resolution)] for indices in (starts, stops)]
    return {
        "source_id": source["id"], "scale_key": request["scale"], "dtype": scale["dtype"],
        "resolution_mm": resolution, "source_shape": source_shape, "shape": shape,
        "requested_bbox_mm": request["roi_mm"], "clipped_bbox_mm": [lo, hi],
        "returned_bbox_mm": returned_bbox, "indices": {"start": starts, "stop": stops},
        "clipped": [lo, hi] != request["roi_mm"],
        "estimates": {"decoded_roi_bytes": decoded, "chunk_mosaic_bytes": mosaic,
                      "one_chunk_bytes": one_chunk, "estimated_working_bytes": working,
                      "model": "decoded ROI + full touched-chunk mosaic + one full chunk",
                      "network_bytes": None, "peak_memory_guarantee": False},
    }


def _extract(fixture, indices):
    shape = _vector(fixture.get("shape"), "fixture.shape", positive=True, integer=True)
    values = fixture.get("values")
    if not isinstance(values, list) or len(values) != math.prod(shape):
        raise ScoutError("Fixture values do not match its recorded shape")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in values):
        raise ScoutError("Synthetic fixture must contain uint8 values")
    nx, ny, _ = shape
    start, stop = indices["start"], indices["stop"]
    return bytes(values[(z * ny + y) * nx + x]
                 for z in range(start[2], stop[2])
                 for y in range(start[1], stop[1])
                 for x in range(start[0], stop[0]))


REPRODUCE = '''"""Rebuild this NONBIOLOGICAL fixture ROI offline; no live-data download."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, help="New raw file; existing files are refused")
args = parser.parse_args()
root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
fixture = json.loads((root / "fixture.json").read_text())
nx, ny, nz = fixture["shape"]
start = manifest["returned"]["indices"]["start"]
stop = manifest["returned"]["indices"]["stop"]
data = bytes(fixture["values"][(z * ny + y) * nx + x]
             for z in range(start[2], stop[2])
             for y in range(start[1], stop[1])
             for x in range(start[0], stop[0]))
if hashlib.sha256(data).hexdigest() != manifest["artifacts"]["data"]["sha256"]:
    raise SystemExit("Reproduced bytes do not match the recorded checksum")
with Path(args.output).open("xb") as handle:
    handle.write(data)
print("Reproduced NONBIOLOGICAL fixture bytes; no network access performed")
'''


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def create_demo(request_path, fixture_path, output):
    """Create a new synthetic capsule, refusing any existing output directory."""
    request, fixture = load_json(request_path), load_json(fixture_path)
    result = plan(request)
    if request["source"]["fixture_type"] != "synthetic" or fixture.get("fixture_type") != "synthetic":
        raise ScoutError("demo accepts only the explicitly synthetic, NONBIOLOGICAL fixture")
    if fixture.get("source") != request["source"] or fixture.get("shape") != result["source_shape"]:
        raise ScoutError("Fixture source metadata does not match the request")
    if result["dtype"] != "uint8":
        raise ScoutError("Offline demo supports only its recorded uint8 fixture")
    data = _extract(fixture, result["indices"])
    shape = result["shape"]
    origin, resolution = result["returned_bbox_mm"][0], result["resolution_mm"]
    affine = [[resolution[i] if i == j else 0 for j in range(3)] + [origin[i]] for i in range(3)]
    affine.append([0, 0, 0, 1])
    preview = f"P5\n# NONBIOLOGICAL synthetic fixture, first z plane\n{shape[0]} {shape[1]}\n255\n".encode() + data[:shape[0] * shape[1]]
    contents = {"data": ("data.raw", data), "preview": ("preview.pgm", preview),
                "fixture": ("fixture.json", _json_bytes(fixture)), "reproduce": ("reproduce.py", REPRODUCE.encode())}
    manifest = {
        "schema_version": "1.0", "fixture_type": "synthetic",
        "source": {k: request["source"][k] for k in ("id", "url", "license", "citation", "coordinate_space", "units")},
        "request": {"bbox_mm": result["requested_bbox_mm"], "resolution_key": result["scale_key"],
                    "resolution_mm": resolution, "max_decoded_bytes": request["max_decoded_bytes"],
                    "max_estimated_working_bytes": request["max_estimated_working_bytes"]},
        "returned": {"shape": shape, "affine": affine, "dtype": "uint8", "resolution_mm": resolution,
                     "bbox_mm": result["returned_bbox_mm"], "clipped_bbox_mm": result["clipped_bbox_mm"],
                     "indices": result["indices"], "storage_order": "x fastest, then y, then z"},
        "provenance": {"generator": "bigbrain-scout", "version": __version__, "offline_only": True,
                       "warning": "NONBIOLOGICAL fixture; not a BigBrain scientific result or live-fetch validation",
                       "environment": {"python": platform.python_version(), "platform": platform.platform()},
                       "estimates": result["estimates"]},
        "artifacts": {key: {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                      for key, (name, content) in contents.items()},
    }
    validate_manifest(manifest)
    destination = Path(output)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ScoutError(f"Output already exists; refusing overwrite: {destination}") from exc
    for name, content in contents.values():
        (destination / name).write_bytes(content)
    (destination / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def validate_manifest(manifest):
    """Check the required schema contract, not the complete JSON Schema vocabulary."""
    _required(manifest, MANIFEST_REQUIRED, "manifest")
    if manifest["schema_version"] != "1.0" or manifest["fixture_type"] != "synthetic":
        raise ScoutError("This prototype verifies only schema 1.0 synthetic capsules")
    _required(manifest["source"], {"id", "url", "license", "citation", "coordinate_space", "units"}, "manifest.source")
    _required(manifest["request"], {"bbox_mm", "resolution_key", "resolution_mm", "max_decoded_bytes", "max_estimated_working_bytes"}, "manifest.request")
    _required(manifest["returned"], {"shape", "affine", "dtype", "resolution_mm", "bbox_mm", "clipped_bbox_mm", "indices", "storage_order"}, "manifest.returned")
    _required(manifest["provenance"], {"generator", "version", "offline_only", "warning", "environment", "estimates"}, "manifest.provenance")
    _required(manifest["provenance"]["environment"], {"python", "platform"}, "manifest.environment")
    _required(manifest["artifacts"], {"data", "preview", "fixture", "reproduce"}, "manifest.artifacts")
    returned = manifest["returned"]
    for key in ("id", "url", "license", "citation", "coordinate_space"):
        if not isinstance(manifest["source"][key], str) or not manifest["source"][key].strip():
            raise ScoutError(f"manifest.source.{key} must be a nonempty string")
    if manifest["source"]["units"] != "mm":
        raise ScoutError("Capsule geometry must record explicit millimetre units")
    if not isinstance(manifest["request"]["resolution_key"], str):
        raise ScoutError("Requested resolution key must be a string")
    if manifest["provenance"]["offline_only"] is not True or manifest["provenance"]["generator"] != "bigbrain-scout":
        raise ScoutError("Synthetic capsule must record its offline-only generator")
    if returned["storage_order"] != "x fastest, then y, then z":
        raise ScoutError("Unsupported recorded storage order")
    _vector(returned["shape"], "returned.shape", positive=True, integer=True)
    resolution = _vector(returned["resolution_mm"], "returned.resolution_mm", positive=True)
    _vector(manifest["request"]["resolution_mm"], "request.resolution_mm", positive=True)
    _required(returned["indices"], {"start", "stop"}, "returned.indices")
    for key in ("start", "stop"):
        values = _vector(returned["indices"][key], f"indices.{key}", integer=True)
        if any(value < 0 for value in values):
            raise ScoutError("Recorded indices must be nonnegative")
    if [b - a for a, b in zip(returned["indices"]["start"], returned["indices"]["stop"])] != returned["shape"]:
        raise ScoutError("Recorded indices do not match returned shape")
    lo, hi = _bbox(returned["bbox_mm"], "returned.bbox_mm")
    _bbox(returned["clipped_bbox_mm"], "returned.clipped_bbox_mm")
    _bbox(manifest["request"]["bbox_mm"], "request.bbox_mm")
    _budget(manifest["request"]["max_decoded_bytes"], "max_decoded_bytes")
    _budget(manifest["request"]["max_estimated_working_bytes"], "max_estimated_working_bytes")
    if resolution != manifest["request"]["resolution_mm"] or returned["dtype"] != "uint8":
        raise ScoutError("Returned resolution/dtype does not meet this prototype's exact contract")
    affine = returned["affine"]
    if not isinstance(affine, list) or len(affine) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in affine):
        raise ScoutError("Affine must be a 4 by 4 matrix")
    for row in affine:
        for value in row:
            _number(value, "affine")
    expected = [[resolution[i] if i == j else 0 for j in range(3)] + [lo[i]] for i in range(3)] + [[0, 0, 0, 1]]
    if affine != expected or any(not math.isclose(b - a, n * r) for a, b, n, r in zip(lo, hi, returned["shape"], resolution)):
        raise ScoutError("Returned shape, affine and bbox are inconsistent")
    for artifact in manifest["artifacts"].values():
        _required(artifact, {"path", "bytes", "sha256"}, "artifact")
        if not isinstance(artifact["path"], str) or not isinstance(artifact["sha256"], str):
            raise ScoutError("Artifact paths/checksums must be strings")
        _budget(artifact["bytes"], "artifact.bytes")
        if len(artifact["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in artifact["sha256"]):
            raise ScoutError("Artifact checksum must be a lowercase SHA-256 digest")


def verify_capsule(directory):
    """Verify recorded geometry and every artifact checksum without fetching data."""
    root = Path(directory).resolve()
    manifest = load_json(root / "manifest.json")
    validate_manifest(manifest)
    for artifact in manifest["artifacts"].values():
        relative = PurePosixPath(artifact["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in artifact["path"]:
            raise ScoutError("Unsafe artifact path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ScoutError("Artifact is missing or escapes the capsule directory")
        content = path.read_bytes()
        if len(content) != artifact["bytes"] or hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise ScoutError(f"Artifact checksum/size mismatch: {relative}")
    expected_bytes = math.prod(manifest["returned"]["shape"])
    if manifest["artifacts"]["data"]["bytes"] != expected_bytes:
        raise ScoutError("Raw byte count does not match recorded uint8 shape")
    return manifest
