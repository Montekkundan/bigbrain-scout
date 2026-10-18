"""Bounded CPU-only dark-component resolution audit, not cell ground truth.

The same untrained heuristic operates on native 1um and derived 2/4um float64
block means. No weights, labels, accuracy estimates, network calls or GPU work
are used. Verified source bytes and fixed settings make the local outputs
auditable. Recomputing a preserved crop does not prove biological validity,
source authenticity, redistribution rights or an independent acquisition.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import struct
import time
from types import MappingProxyType
import zlib

from . import __version__, core, replay

MAX_DECODED = 1024**2
MAX_REPORT = 2 * 1024**2
MAX_COMPONENTS = 16384
MAX_CANDIDATES = 4096
RESOLUTIONS = (1, 2, 4)
DEFAULT_SETTINGS = MappingProxyType({"smoothing_sigma_um": 1.5, "dark_threshold_uint8": 90.0,
                                    "min_area_um2": 16.0, "max_area_um2": 600.0})
ORIENTATION = "PNG/raw columns increase BigBrain X; rows increase BigBrain Z (downwards); fixed section Y. Not a radiological display."
ARTIFACT_PATHS = {"source": "source.raw", **{role: f"candidate-{size}um.{suffix}"
                   for size in RESOLUTIONS for role, suffix in ((f"mask_{size}um", "raw"), (f"png_{size}um", "png"))}}
LIMITATIONS = [
    "Candidates are connected dark components, not annotated or validated biological cells.",
    "No ground truth, accuracy, recall, precision, segmentation quality or biological count is established.",
    "One fixed registered crop from one donor is not a representative or anatomically selected sample.",
    "2/4um levels are block means derived from the same native crop, not independent acquisitions.",
    "Fixed physical parameters still produce different discrete kernels, pixels and component topology across scales.",
    "Merged, split, faint and artifact components can change counts; border-touching components are excluded.",
    "Runtime is one local wall-clock measurement, not a benchmark, CPU-time measurement or memory guarantee.",
    "Offline recomputation checks local consistency, not source authenticity, independent download or redistribution rights.",
    "Real source data and masks remain local; Scout's MIT code license does not relicense BigBrain data.",
]


def _runtime():
    try:
        import numpy as np
        import scipy.ndimage as ndi
    except ImportError as exc:
        raise core.ScoutError("CPU audit requires the existing NumPy and SciPy environment; no install performed") from exc
    return np, ndi


def _settings(value=None):
    result = dict(DEFAULT_SETTINGS)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise core.ScoutError("Unknown audit settings")
        result.update(value)
    bounds = {"smoothing_sigma_um": (0.1, 10), "dark_threshold_uint8": (0, 255),
              "min_area_um2": (1, 4096), "max_area_um2": (1, 16384)}
    for key, (lower, upper) in bounds.items():
        value = core._number(result[key], key)
        if not lower <= value <= upper:
            raise core.ScoutError(f"{key} is outside the bounded audit range")
        result[key] = float(value)
    if result["min_area_um2"] > result["max_area_um2"]:
        raise core.ScoutError("Minimum candidate area exceeds maximum area")
    return result


def _method(settings):
    return {"id": "fixed-dark-components-v1", "execution": "CPU NumPy/SciPy only",
            "scale_derivation": "Disjoint square block means using float64; no per-level intensity normalization or uint8 rounding",
            "smoothing": "scipy.ndimage.gaussian_filter; float64 input/output; sigma_um / resolution_um",
            "gaussian_mode": "reflect", "gaussian_truncate": 4.0,
            "threshold_rule": "smoothed_intensity <= dark_threshold_uint8",
            "connectivity": 8, "area_rule": "Inclusive min/max area_um2; pixel_count * resolution_um squared",
            "border_rule": "Exclude every component containing any outermost row or column pixel",
            "settings": deepcopy(settings), "mask_encoding": "uint8 C order; 255 accepted candidate, 0 background",
            "parameter_selection": "Untrained fixed heuristic, not fitted to labels or tuned separately by scale"}


def _block_mean(plane, factor):
    np, _ = _runtime()
    if (not isinstance(plane, np.ndarray) or plane.ndim != 2 or plane.dtype != np.uint8
            or type(factor) is not int or factor not in RESOLUTIONS
            or min(plane.shape) < 1 or any(size % factor for size in plane.shape) or plane.size > MAX_DECODED):
        raise core.ScoutError("Expected a bounded divisible uint8 source plane and factor 1/2/4")
    height, width = plane.shape
    return plane.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3), dtype=np.float64)


def _candidates(values, resolution_um, settings):
    """Return summary plus exact binary mask. Tests use NONBIOLOGICAL arrays."""
    np, ndi = _runtime()
    settings = _settings(settings)
    if (type(resolution_um) is not int or resolution_um not in RESOLUTIONS
            or not isinstance(values, np.ndarray) or values.ndim != 2 or values.dtype != np.float64
            or values.size > MAX_DECODED or min(values.shape) < 2
            or not np.isfinite(values).all() or values.min() < 0 or values.max() > 255):
        raise core.ScoutError("Expected bounded finite float64 intensities at an exact audit resolution")
    smoothed = ndi.gaussian_filter(values, sigma=settings["smoothing_sigma_um"] / resolution_um,
                                  output=np.float64, mode="reflect", truncate=4.0)
    labels, count = ndi.label(smoothed <= settings["dark_threshold_uint8"], structure=np.ones((3, 3), dtype=np.uint8))
    if count > MAX_COMPONENTS:
        raise core.ScoutError("Too many threshold components for the bounded audit")
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    border = set(int(item) for item in np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))) - {0}
    areas = counts.astype(float) * resolution_um**2
    accepted = [identifier for identifier in range(1, count + 1)
                if identifier not in border and settings["min_area_um2"] <= areas[identifier] <= settings["max_area_um2"]]
    if len(accepted) > MAX_CANDIDATES:
        raise core.ScoutError("Too many candidates for a bounded public report")
    keep = np.zeros(count + 1, dtype=np.uint8)
    keep[accepted] = 255
    mask = keep[labels]
    candidates = []
    objects = ndi.find_objects(labels, max_label=count)
    for identifier in accepted:
        row_slice, col_slice = objects[identifier - 1]
        local = labels[row_slice, col_slice] == identifier
        rows, columns = np.nonzero(local)
        candidates.append({"id": identifier, "area_pixels": int(counts[identifier]),
                           "area_um2": float(areas[identifier]),
                           "centroid_row_col": [float(rows.mean() + row_slice.start), float(columns.mean() + col_slice.start)],
                           "bbox_row_col_exclusive": [[row_slice.start, col_slice.start], [row_slice.stop, col_slice.stop]],
                           "mean_smoothed_intensity": float(smoothed[row_slice, col_slice][local].mean())})
    accepted_areas = [item["area_um2"] for item in candidates]
    result = {"resolution_um": resolution_um, "factor": resolution_um,
              "width": int(values.shape[1]), "height": int(values.shape[0]),
              "smoothing_sigma_pixels": settings["smoothing_sigma_um"] / resolution_um,
              "candidate_count": len(candidates), "raw_component_count": int(count),
              "excluded_border_components": len(border),
              "rejected_area_components": int(count) - len(border) - len(candidates),
              "total_candidate_area_um2": sum(accepted_areas),
              "mean_candidate_area_um2": sum(accepted_areas) / len(accepted_areas) if accepted_areas else None,
              "min_candidate_area_um2": min(accepted_areas) if accepted_areas else None,
              "max_candidate_area_um2": max(accepted_areas) if accepted_areas else None,
              "candidates": candidates, "mask_raw_role": f"mask_{resolution_um}um", "mask_png_role": f"png_{resolution_um}um"}
    return result, np.ascontiguousarray(mask)


def _png(mask):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    height, width = mask.shape
    raw = b"".join(b"\0" + row.tobytes() for row in mask)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"tEXt", b"Description\0Classical dark-component candidate mask; not biological cell labels. BigBrain source terms apply.")
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _identity(manifest):
    from .live import defaults, catalog
    source, request, returned = manifest["source"], manifest["request"], manifest["returned"]
    if (manifest["schema_version"] != "live-1.0" or manifest["fixture_type"] != "real_remote_data"
            or source["id"] != "section-3905" or source["url"] != catalog("section-3905")["source_url"]
            or request["resolution_key"] != "1um" or request["bbox_mm"] != defaults()["roi_mm"]):
        raise core.ScoutError("Audit requires the trusted section-3905 preset at exact native 1um")
    shape = core._vector(returned["shape"], "source shape", positive=True, integer=True)
    if (returned["dtype"] != "uint8" or shape[1] != 1 or any(size > 2048 for size in shape)
            or min(shape[0], shape[2]) < 8 or math.prod(shape) > MAX_DECODED
            or returned.get("decoded_bytes") != math.prod(shape)):
        raise core.ScoutError("Audit source must be a bounded uint8 single section, at most 1 MiB decoded")
    np, _ = _runtime()
    affine = np.asarray(returned["affine"], dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[:3, :3], np.diag([0.001, 0.020, 0.001]), rtol=0, atol=1e-8)
            or np.abs(affine[:3, 3]).max() > 1000 or not np.array_equal(affine[3], [0, 0, 0, 1])):
        raise core.ScoutError("Unreviewed source spacing, affine or orientation")


def _source(capsule):
    """Precheck the 1 MiB limit, then verify before decoding a saved snapshot."""
    root = replay._path(capsule)
    manifest = replay._json(replay._read(root / "manifest.json", replay.MAX_MANIFEST_BYTES))
    try:
        replay._limits(manifest)
        _identity(manifest)
        if manifest["artifacts"]["data"]["bytes"] > MAX_DECODED + 4096:
            raise core.ScoutError("Audit NIfTI exceeds its bounded file size")
        _, verified, contents, _ = replay._inspect(root)
        if verified != manifest:
            raise core.ScoutError("Audit source changed during verification")
        array, decoded_hash = replay._decoded(verified, contents)
    except core.ScoutError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise core.ScoutError("Invalid audit source capsule") from exc
    np, _ = _runtime()
    if array.nbytes > MAX_DECODED or array.dtype != np.uint8:
        raise core.ScoutError("Decoded audit source violates its bounded dtype/size")
    whole = array[:, 0, :].T
    height, width = (size - size % 4 for size in whole.shape)
    plane = np.ascontiguousarray(whole[:height, :width])
    original = verified["source"]
    record = {key: original[key] for key in ("id", "url", "citation", "license", "coordinate_space", "units")}
    record.update(native_resolution_um=1, through_section_um=20, source_shape=list(array.shape),
                  source_affine_mm=deepcopy(verified["returned"]["affine"]), request_bbox_mm=deepcopy(verified["request"]["bbox_mm"]),
                  width=width, height=height, fov_um=[width, height],
                  nifti_sha256=verified["artifacts"]["data"]["sha256"], decoded_sha256=decoded_hash,
                  crop_sha256=_sha(plane.tobytes()), orientation=ORIENTATION,
                  crop_index_start_xyz=[0, 0, 0], crop_index_stop_xyz_exclusive=[width, 1, height])
    return record, plane


def _calculate(plane, settings, measured=False):
    files = {"source": plane.tobytes()}
    levels = []
    for resolution in RESOLUTIONS:
        started = time.perf_counter()
        level, mask = _candidates(_block_mean(plane, resolution), resolution, settings)
        elapsed = time.perf_counter() - started
        level["runtime_seconds"] = elapsed if measured else 0.0
        levels.append(level)
        files[f"mask_{resolution}um"] = mask.tobytes()
        files[f"png_{resolution}um"] = _png(mask)
    if sum(level["candidate_count"] for level in levels) > MAX_CANDIDATES:
        raise core.ScoutError("Total candidates exceed the bounded report size")
    count = levels[0]["candidate_count"]
    for level in levels:
        drift = level["candidate_count"] - count
        level["count_drift_from_native"] = drift
        level["count_drift_percent_from_native"] = drift * 100 / count if count else None
    return levels, files


def _without_runtime(levels):
    return [{key: value for key, value in level.items() if key != "runtime_seconds"} for level in levels]


def run_audit(capsule, output, *, settings=None):
    """Write a new local audit directory and return its freshly verified report."""
    destination = replay._path(output)
    if destination.exists():
        raise core.ScoutError("Audit output exists; refusing overwrite")
    settings = _settings(settings)
    source, plane = _source(capsule)
    levels, files = _calculate(plane, settings, measured=True)
    environment = {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "nibabel")}}
    report = {"schema_version": "audit-1.0", "kind": "classical_dark_candidate_resolution_audit",
              "generator": "bigbrain-scout", "generator_version": __version__, "source": source,
              "settings": settings, "method": _method(settings), "environment": environment,
              "levels": levels, "runtime_seconds_total": sum(level["runtime_seconds"] for level in levels),
              "artifacts": {role: {"path": ARTIFACT_PATHS[role], "bytes": len(data), "sha256": _sha(data)} for role, data in files.items()},
              "limitations": list(LIMITATIONS), "accuracy_established": False, "gpu_used": False,
              "source_capsule_verified_before_decode": True}
    encoded = core._json_bytes(report)
    if len(encoded) > MAX_REPORT:
        raise core.ScoutError("Audit report exceeds its size limit")
    try:
        destination.mkdir(parents=True, exist_ok=False)
        for role, data in files.items():
            with (destination / ARTIFACT_PATHS[role]).open("xb") as handle:
                handle.write(data)
        with (destination / "report.json").open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise core.ScoutError("Audit output appeared; refusing overwrite") from exc
    except OSError as exc:
        raise core.ScoutError("Cannot write local audit output") from exc
    return verify_audit(destination)


def verify_audit(output):
    """Verify known paths, hashes and recomputed masks/metrics, never execute files.

    The original capsule is not needed. A preserved raw crop and report can
    establish local consistency only, not that a claimed source hash is genuine.
    """
    root = replay._path(output)
    report = replay._json(replay._read(root / "report.json", MAX_REPORT))
    required = {"schema_version", "kind", "generator", "generator_version", "source", "settings", "method", "environment", "levels", "runtime_seconds_total", "artifacts", "limitations", "accuracy_established", "gpu_used", "source_capsule_verified_before_decode"}
    try:
        if (set(report) != required or report["schema_version"] != "audit-1.0"
                or report["kind"] != "classical_dark_candidate_resolution_audit"
                or report["generator"] != "bigbrain-scout" or report["generator_version"] != __version__
                or report["accuracy_established"] is not False or report["gpu_used"] is not False
                or report["source_capsule_verified_before_decode"] is not True or report["limitations"] != LIMITATIONS):
            raise core.ScoutError("Unknown audit schema or claims")
        settings = _settings(report["settings"])
        if (set(report["settings"]) != set(DEFAULT_SETTINGS)
                or core._json_bytes(settings) != core._json_bytes(report["settings"])
                or core._json_bytes(report["method"]) != core._json_bytes(_method(settings))):
            raise core.ScoutError("Audit method/settings mismatch")
        source = report["source"]
        source_fields = {"id", "url", "citation", "license", "coordinate_space", "units", "native_resolution_um", "through_section_um", "source_shape", "source_affine_mm", "request_bbox_mm", "width", "height", "fov_um", "nifti_sha256", "decoded_sha256", "crop_sha256", "orientation", "crop_index_start_xyz", "crop_index_stop_xyz_exclusive"}
        if not isinstance(source, dict) or set(source) != source_fields:
            raise core.ScoutError("Unknown audit source fields")
        from .live import defaults, catalog, LICENSE_STATUS
        trusted = catalog("section-3905")
        if (source["id"] != trusted["id"] or source["url"] != trusted["source_url"]
                or source["citation"] != trusted["configuration_url"] or source["license"] != LICENSE_STATUS
                or source["coordinate_space"] != "BigBrain histological reference space" or source["units"] != "mm"
                or type(source["native_resolution_um"]) is not int or source["native_resolution_um"] != 1
                or type(source["through_section_um"]) is not int or source["through_section_um"] != 20
                or source["request_bbox_mm"] != defaults()["roi_mm"] or source["orientation"] != ORIENTATION):
            raise core.ScoutError("Audit source identity differs from the trusted preset")
        width, height = source["width"], source["height"]
        if (type(width) is not int or type(height) is not int or not 8 <= width <= 2048 or not 8 <= height <= 2048
                or width % 4 or height % 4 or width * height > MAX_DECODED or source["fov_um"] != [width, height]
                or source["crop_index_start_xyz"] != [0, 0, 0] or source["crop_index_stop_xyz_exclusive"] != [width, 1, height]):
            raise core.ScoutError("Invalid audit crop dimensions")
        shape = core._vector(source["source_shape"], "source shape", positive=True, integer=True)
        if shape[1] != 1 or math.prod(shape) > MAX_DECODED or not 0 <= shape[0] - width < 4 or not 0 <= shape[2] - height < 4:
            raise core.ScoutError("Audit crop does not match its original source shape")
        _identity({"schema_version": "live-1.0", "fixture_type": "real_remote_data", "source": source,
                   "request": {"resolution_key": "1um", "bbox_mm": source["request_bbox_mm"]},
                   "returned": {"shape": shape, "dtype": "uint8", "decoded_bytes": math.prod(shape), "affine": source["source_affine_mm"]}})
        for key in ("nifti_sha256", "decoded_sha256", "crop_sha256"):
            digest = source[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise core.ScoutError("Invalid audit source hash")
        environment = report["environment"]
        if not isinstance(environment, dict) or set(environment) != {"python", "numpy", "scipy", "nibabel"} or any(not isinstance(value, str) or not value or len(value) > 32 for value in environment.values()):
            raise core.ScoutError("Invalid audit environment fields")
        for name in ("numpy", "scipy"):
            if environment[name] != importlib.metadata.version(name):
                raise core.ScoutError("Use the recorded NumPy/SciPy versions for exact audit recomputation")
        artifacts, contents = report["artifacts"], {}
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_PATHS):
            raise core.ScoutError("Unknown audit artifacts")
        if {entry.name for entry in root.iterdir()} != {"report.json", *ARTIFACT_PATHS.values()}:
            raise core.ScoutError("Audit contains unexpected files or directories")
        for role, name in ARTIFACT_PATHS.items():
            record = artifacts[role]
            if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"} or record["path"] != name:
                raise core.ScoutError("Unsafe audit artifact path")
            data = replay._read(root / name, MAX_DECODED + 8192)
            if type(record["bytes"]) is not int or len(data) != record["bytes"] or _sha(data) != record["sha256"]:
                raise core.ScoutError("Audit artifact checksum or size mismatch")
            contents[role] = data
        if len(contents["source"]) != width * height or _sha(contents["source"]) != source["crop_sha256"]:
            raise core.ScoutError("Audit source crop checksum/shape mismatch")
        np, _ = _runtime()
        plane = np.frombuffer(contents["source"], dtype=np.uint8).reshape(height, width)
        levels, recalculated = _calculate(plane, settings)
        if (recalculated != contents
                or core._json_bytes(_without_runtime(report["levels"])) != core._json_bytes(_without_runtime(levels))):
            raise core.ScoutError("Audit masks or candidate metrics differ from trusted recomputation")
        for level in report["levels"]:
            runtime = core._number(level["runtime_seconds"], "runtime_seconds")
            if not 0 <= runtime <= 3600:
                raise core.ScoutError("Invalid audit runtime measurement")
        total = core._number(report["runtime_seconds_total"], "runtime_seconds_total")
        if total != sum(level["runtime_seconds"] for level in report["levels"]):
            raise core.ScoutError("Audit runtime total differs from measurements")
        return report
    except core.ScoutError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, OSError) as exc:
        raise core.ScoutError("Audit violates the strict bounded verification contract") from exc
