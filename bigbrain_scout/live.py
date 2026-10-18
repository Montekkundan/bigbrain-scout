"""Bounded registered-section workflow using the pinned siibra metadata adapter.

Preparation reads metadata only. Estimates cover decoded arrays and chunk
assembly, not network transfer bytes or a guaranteed process-memory ceiling.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from itertools import product
from pathlib import Path, PurePosixPath

from . import __version__
from .core import ScoutError, _bbox, _budget, _json_bytes, _required, _vector
from .sources import artifact_names, catalog, source_ids

LIMIT = 8 * 1024**2
PIN = "1.0.1a22"
SOURCE_ID = "section-3905"
SOURCE_URL = "https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3905/precomputed"
CONFIG_URL = "https://github.com/FZJ-INM1-BDA/siibra-configurations/blob/master/features/images/sections/cellbody/73c1fa55-d099-4854-8cda-c9a403c6080a_bigbrain_1um_3905.json"
SCALES = {f"{size}um": size / 1000 for size in (1, 2, 4, 8, 16, 32, 64, 128)}
REQUEST_KEYS = {"source_id", "roi_mm", "scale", "max_decoded_bytes", "max_estimated_working_bytes"}
ARTIFACT_NAMES = {"data": "section-3905.nii", "preview": "preview.pgm",
                  "metadata": "metadata.json", "reproduce": "README.md"}
ESTIMATE_MODEL = "decoded ROI + full touched-chunk mosaic + one full chunk"
LICENSE_STATUS = "Source data terms not verified for redistribution; keep data local"


def defaults(source_id=SOURCE_ID):
    """Return a fresh small geometry request for one whitelisted section."""
    if not isinstance(source_id, str):
        raise ScoutError("Live source must be a whitelisted registered section")
    source = catalog(source_id)
    return {"source_id": source["id"], "roi_mm": source["default_roi_mm"],
            "scale": source["default_scale"], "max_decoded_bytes": LIMIT,
            "max_estimated_working_bytes": LIMIT}


def _request(value):
    _required(value, REQUEST_KEYS, "live request")
    if set(value) != REQUEST_KEYS:
        raise ScoutError("Live request has unknown fields")
    if not isinstance(value["source_id"], str):
        raise ScoutError("Live source must be a whitelisted registered section")
    catalog(value["source_id"])
    if not isinstance(value["scale"], str) or value["scale"] not in SCALES:
        raise ScoutError("Exact live scale unavailable; no automatic fallback")
    lower, upper = _bbox(value["roi_mm"], "roi_mm")
    if any(not math.isfinite(b - a) for a, b in zip(lower, upper)):
        raise ScoutError("ROI extents must remain finite")
    for key in ("max_decoded_bytes", "max_estimated_working_bytes"):
        if _budget(value[key], key) > LIMIT:
            raise ScoutError(f"{key} must be at most {LIMIT} bytes")
    return copy.deepcopy(value)


def _runtime(cache_dir):
    """Keep the prerelease's internal metadata helper in this adapter."""
    os.environ["SIIBRA_MAX_FETCH_SIZE_BYTES"] = str(LIMIT)
    os.environ["SIIBRA_CACHEDIR"] = str(Path(cache_dir).resolve())
    try:
        if importlib.metadata.version("siibra") != PIN:
            raise ScoutError(f"Live workflow requires siibra=={PIN}")
        import numpy as np
        import siibra
        from siibra.volumes.providers.neuroglancer import NeuroglancerVolume
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ScoutError("Install the existing project live extra to use remote data") from exc
    return np, siibra, NeuroglancerVolume


def _array(np, value, name, shape):
    try:
        if np.asarray(value).dtype.kind in "bUSO":
            raise ScoutError(f"{name} must contain numeric values, not booleans or strings")
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ScoutError(f"Invalid {name}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ScoutError(f"{name} must have finite shape {shape}")
    return result


def _affine(np, value):
    result = _array(np, value, "affine", (4, 4))
    if not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ScoutError("Affine must use homogeneous millimetre coordinates")
    if abs(float(np.linalg.det(result[:3, :3]))) < 1e-20:
        raise ScoutError("Affine is singular")
    return result


def _transform_bbox(np, bbox, affine):
    corners = np.asarray([affine @ np.asarray([*corner, 1.0])
                          for corner in product(*zip(bbox[0], bbox[1]))])[:, :3]
    if not np.isfinite(corners).all():
        raise ScoutError("Transformed ROI coordinates must remain finite")
    return np.asarray([corners.min(axis=0), corners.max(axis=0)])


def _image_bbox(np, affine, shape):
    return _transform_bbox(np, [[0, 0, 0], list(shape)], affine).tolist()


def _metadata(np, value, request):
    """Validate the captured scale and independently derive its affine."""
    _required(value, {"source_url", "source_info", "source_transform_nm", "source_bbox_mm",
                      "selected_scale_key", "selected_scale_affine_mm", "environment"}, "metadata")
    source = catalog(request["source_id"])
    if value["source_url"] != source["source_url"]:
        raise ScoutError("Live metadata source is not whitelisted")
    if value.get("configuration_url", source["configuration_url"]) != source["configuration_url"]:
        raise ScoutError("Live metadata configuration reference differs from its source")
    if "feature_name" in value and (not isinstance(value["feature_name"], str)
                                    or not value["feature_name"].startswith(f"#{source['section_number']}:")):
        raise ScoutError("Captured section identity differs from its requested source")
    info = value["source_info"]
    _required(info, {"data_type", "num_channels", "scales"}, "source_info")
    if info["data_type"] != "uint8" or info["num_channels"] != 1:
        raise ScoutError("Live workflow supports only single-channel uint8 sections")
    if not isinstance(info["scales"], list) or not info["scales"]:
        raise ScoutError("Source must publish scale metadata")
    matches = []
    smallest = None
    seen = set()
    for scale in info["scales"]:
        _required(scale, {"key", "resolution", "size", "chunk_sizes", "voxel_offset", "encoding"}, "source scale")
        if not isinstance(scale["key"], str) or not scale["key"] or scale["key"] in seen:
            raise ScoutError("Source scale keys must be unique strings")
        seen.add(scale["key"])
        resolution = _array(np, scale["resolution"], "scale resolution", (3,))
        if np.any(resolution <= 0) or resolution[1] != 20000 or resolution[0] != resolution[2]:
            raise ScoutError("Unexpected section resolution metadata")
        _vector(scale["size"], "scale size", positive=True, integer=True)
        _array(np, scale["voxel_offset"], "voxel offset", (3,))
        if np.any(np.asarray(scale["voxel_offset"]) != 0):
            raise ScoutError("This pinned adapter requires zero voxel offsets")
        if not isinstance(scale["chunk_sizes"], list) or len(scale["chunk_sizes"]) != 1:
            raise ScoutError("Expected one chunk shape per source scale")
        _vector(scale["chunk_sizes"][0], "chunk size", positive=True, integer=True)
        if scale["size"][1] != 1 or scale["chunk_sizes"][0][1] != 1 or scale["encoding"] != "raw":
            raise ScoutError("Unexpected section storage metadata")
        if smallest is None or resolution[0] < smallest["resolution"][0]:
            smallest = scale
        if scale["key"] == value["selected_scale_key"]:
            matches.append(scale)
    if len(matches) != 1:
        raise ScoutError("Selected source scale unavailable or ambiguous")
    selected = matches[0]
    expected_nm = [SCALES[request["scale"]] * 1e6, 20000, SCALES[request["scale"]] * 1e6]
    if not np.allclose(selected["resolution"], expected_nm, rtol=0, atol=1e-7):
        raise ScoutError("Exact requested scale unavailable; no automatic fallback")
    transform = _affine(np, value["source_transform_nm"])
    # This reproduces the pinned provider's half-voxel convention. Keep the
    # original metadata too, so future versions can review that convention.
    affine = transform @ np.diag([*selected["resolution"], 1.0])
    affine[:3, 3] += smallest["resolution"][0] * 0.5
    affine[:3] /= 1e6
    if not np.allclose(_affine(np, value["selected_scale_affine_mm"]), affine, rtol=0, atol=1e-10):
        raise ScoutError("Selected affine does not match captured source metadata")
    _bbox(value["source_bbox_mm"], "source bbox")
    return selected, affine


def _plan(np, request, metadata):
    selected, affine = _metadata(np, metadata, request)
    source_bbox = np.asarray(metadata["source_bbox_mm"], dtype=float)
    scale_bbox = np.asarray(_image_bbox(np, affine, selected["size"]))
    lower = np.maximum(request["roi_mm"][0], np.maximum(source_bbox[0], scale_bbox[0]))
    upper = np.minimum(request["roi_mm"][1], np.minimum(source_bbox[1], scale_bbox[1]))
    if np.any(upper <= lower):
        raise ScoutError("ROI does not intersect the selected section scale")
    clipped = [lower.tolist(), upper.tolist()]
    voxels = _transform_bbox(np, clipped, np.linalg.inv(affine))
    voxel_min, voxel_max = voxels
    offset = np.asarray(selected["voxel_offset"])
    size = np.asarray(selected["size"])
    if np.any(voxel_min < offset - 1e-6) or np.any(voxel_max > offset + size + 1e-6):
        raise ScoutError("Transformed ROI falls outside source voxel bounds")
    shape = np.ceil(voxel_max).astype(int) - np.floor(voxel_min).astype(int)
    chunks = np.asarray(selected["chunk_sizes"][0])
    if np.any(shape <= 0) or np.any(shape > 32767):
        raise ScoutError("Invalid or excessive NIfTI-1 dimensions")
    chunk_min = np.floor((voxel_min - offset) / chunks).astype(int)
    chunk_max = np.ceil((voxel_max - offset) / chunks).astype(int)
    mosaic = math.prod(int(n) for n in ((chunk_max - chunk_min) * chunks))
    decoded = math.prod(int(n) for n in shape)
    one_chunk = math.prod(int(n) for n in chunks)
    working = decoded + mosaic + one_chunk
    if decoded > request["max_decoded_bytes"]:
        raise ScoutError(f"Decoded ROI estimate {decoded} exceeds the request budget")
    if working > request["max_estimated_working_bytes"]:
        raise ScoutError(f"Chunk assembly estimate {working} exceeds the request budget")
    shift = np.eye(4)
    shift[:3, 3] = voxel_min
    reversal = np.eye(4)[[2, 1, 0, 3], :]
    returned_affine = affine @ shift @ reversal
    array_shape = shape[::-1].tolist()
    return {"source_id": request["source_id"], "scale_key": request["scale"],
            "source_scale_key": selected["key"], "dtype": "uint8",
            "resolution_mm": np.linalg.norm(returned_affine[:3, :3], axis=0).tolist(),
            "resolution_mm_provider_xyz": (np.asarray(selected["resolution"]) / 1e6).tolist(),
            "shape": array_shape, "shape_provider_xyz": shape.tolist(),
            "requested_bbox_mm": request["roi_mm"], "clipped_bbox_mm": clipped,
            "returned_bbox_mm": _image_bbox(np, returned_affine, array_shape),
            "returned_affine_mm": returned_affine.tolist(),
            "clipped": clipped != request["roi_mm"],
            "estimates": {"decoded_roi_bytes": decoded, "chunk_mosaic_bytes": mosaic,
                          "one_chunk_bytes": one_chunk, "estimated_working_bytes": working,
                          "model": ESTIMATE_MODEL, "network_bytes": None,
                          "peak_memory_guarantee": False}}


def prepare(request, cache_dir):
    """Validate locally, then inspect source metadata without fetching image chunks."""
    request = _request(request)
    source = catalog(request["source_id"])
    np, siibra, volume_class = _runtime(cache_dir)
    try:
        roi = siibra.locations.BoundingBox(*request["roi_mm"], space="bigbrain")
        sections = siibra.features.get(roi, "CellbodyStainedSection")
        section = next((item for item in sections
                        if item.name.startswith(f"#{source['section_number']}:")), None)
        if section is None:
            raise ScoutError(f"Section #{source['section_number']} not found for this ROI")
        if section.providers.get("neuroglancer/precomputed") != source["source_url"]:
            raise ScoutError("Configured section source changed; review it before fetching")
        bounds = section.get_boundingbox(clip=False)
        volume = volume_class(source["source_url"])
        wanted = [SCALES[request["scale"]], 0.020, SCALES[request["scale"]]]
        scales = [scale for scale in volume.scales
                  if np.allclose(scale.res_mm, wanted, rtol=0, atol=1e-9)]
        if len(scales) != 1:
            raise ScoutError("Exact requested scale unavailable or ambiguous; no automatic fallback")
        scale = scales[0]
        metadata = {"source_url": source["source_url"], "feature_name": section.name,
                    "feature_id": section.id, "configuration_url": source["configuration_url"],
                    "source_bbox_mm": [list(bounds.minpoint), list(bounds.maxpoint)],
                    "source_info": copy.deepcopy(volume._info),
                    "source_transform_nm": volume.transform_nm.tolist(),
                    "selected_scale_key": scale.key, "selected_scale_affine_mm": scale.affine.tolist(),
                    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                    "siibra": PIN, "numpy": importlib.metadata.version("numpy"),
                                    "nibabel": importlib.metadata.version("nibabel")}}
        result = _plan(np, request, metadata)
        clipped = siibra.locations.BoundingBox(*result["clipped_bbox_mm"], space="bigbrain")
    except ScoutError:
        raise
    except Exception as exc:
        raise ScoutError(f"Cannot prepare section metadata: {exc}") from exc
    context = {"request": request, "plan": copy.deepcopy(result), "metadata": metadata,
               "section": section, "roi": clipped}
    return context, result


def _returned(np, image, plan, request, *, tolerance=1e-7):
    affine = _affine(np, image.affine)
    shape = list(image.shape)
    _vector(shape, "returned shape", positive=True, integer=True)
    dtype = np.dtype(image.dataobj.dtype)
    decoded = math.prod(shape) * dtype.itemsize
    resolution = np.linalg.norm(affine[:3, :3], axis=0)
    bbox = _image_bbox(np, affine, shape)
    if shape != plan["shape"] or dtype != np.dtype("uint8"):
        raise ScoutError("Returned shape or dtype differs from the exact prepared request")
    if decoded != plan["estimates"]["decoded_roi_bytes"] or decoded > request["max_decoded_bytes"]:
        raise ScoutError("Returned decoded data exceeds or differs from its prepared budget")
    if not np.allclose(resolution, plan["resolution_mm"], rtol=0, atol=tolerance):
        raise ScoutError("Returned resolution changed; refusing a capsule")
    if not np.allclose(affine, plan["returned_affine_mm"], rtol=0, atol=tolerance):
        raise ScoutError("Returned affine differs from the prepared geometry")
    if not np.allclose(bbox, plan["returned_bbox_mm"], rtol=0, atol=tolerance):
        raise ScoutError("Returned spatial bounds differ from the prepared geometry")
    return {"shape": shape, "affine": affine.tolist(), "bbox_mm": bbox,
            "resolution_mm": resolution.tolist(), "dtype": str(dtype), "decoded_bytes": decoded,
            "clipped_bbox_mm": plan["clipped_bbox_mm"],
            "storage_order": "NIfTI array axes; interpret with the recorded affine"}


def _snapshots(metadata):
    return {key + "_sha256": hashlib.sha256(_json_bytes(metadata[key])).hexdigest()
            for key in ("source_info", "source_transform_nm")}


def fetch(context, plan, output):
    """Fetch the prepared small request and export a new, locally kept capsule."""
    destination = Path(output)
    if destination.exists():
        raise ScoutError("Output already exists; refusing overwrite")
    request = _request(context["request"])
    source = catalog(request["source_id"])
    names = artifact_names(request["source_id"])
    try:
        import numpy as np
        import nibabel as nib
    except ImportError as exc:
        raise ScoutError("Install the existing live extra to fetch NIfTI data") from exc
    recalculated = _plan(np, request, context["metadata"])
    if plan != recalculated or context["plan"] != recalculated:
        raise ScoutError("Prepared plan changed; prepare the request again")
    if context["section"].providers.get("neuroglancer/precomputed") != source["source_url"]:
        raise ScoutError("Section source changed after preparation")
    try:
        image = context["section"].fetch(format="neuroglancer/precomputed", voi=context["roi"],
                                         resolution_mm=SCALES[request["scale"]],
                                         max_bytes=request["max_decoded_bytes"])
    except Exception as exc:
        raise ScoutError(f"Section fetch failed: {exc}") from exc
    if image is None:
        raise ScoutError("siibra returned no image")
    _returned(np, image, plan, request)
    array = np.asanyarray(image.dataobj)
    # Only remove the known through-section axis, preserving 1 by N previews.
    plane = array[:, 0, :]
    if plane.ndim != 2 or plane.dtype != np.uint8:
        raise ScoutError("Expected a two-dimensional uint8 section preview")
    metadata = copy.deepcopy(context["metadata"])
    metadata.update(request=request, plan=plan)
    instructions = ("# Reproduce the recorded section request\n\n"
                    "Keep these source data local until the source dataset terms are reviewed.\n\n"
                    "Install this repository with its existing live extra and replay this verified capsule. "
                    "Choose a fresh output directory. Re-running may observe changed remote content.\n\n"
                    "```sh\npython -m bigbrain_scout reproduce --capsule /path/to/this-capsule "
                    f"--output artifacts/reproduced-{request['source_id']} --allow-live\n```\n\n"
                    "Compare the returned geometry, data arrays and checksums. This capsule does not prove "
                    "an independent reproduction or scientific validity. Verification does not execute these instructions.\n")
    try:
        destination.mkdir(parents=True, exist_ok=False)
        nib.save(image, destination / names["data"])
        # Record the geometry represented by the saved NIfTI header, whose
        # sform storage rounds floats. Compare it with the original prediction.
        saved = nib.load(destination / names["data"])
        returned = _returned(np, saved, plan, request, tolerance=1e-5)
        (destination / names["preview"]).write_bytes(
            f"P5\n{plane.shape[1]} {plane.shape[0]}\n255\n".encode() + plane.tobytes())
        (destination / names["metadata"]).write_bytes(_json_bytes(metadata))
        (destination / names["reproduce"]).write_text(instructions, encoding="utf-8")
        artifacts = {}
        for key, name in names.items():
            content = (destination / name).read_bytes()
            artifacts[key] = {"path": name, "bytes": len(content),
                              "sha256": hashlib.sha256(content).hexdigest()}
        manifest = {"schema_version": "live-1.0", "fixture_type": "real_remote_data",
                    "source": {"id": source["id"], "url": source["source_url"], "license": LICENSE_STATUS,
                               "citation": source["configuration_url"], "coordinate_space": "BigBrain histological reference space",
                               "units": "mm"},
                    "request": {"bbox_mm": request["roi_mm"], "resolution_key": request["scale"],
                                "resolution_mm": plan["resolution_mm_provider_xyz"],
                                "resolution_axes": "provider xyz",
                                "max_decoded_bytes": request["max_decoded_bytes"],
                                "max_estimated_working_bytes": request["max_estimated_working_bytes"]},
                    "returned": returned,
                    "provenance": {"generator": "bigbrain-scout", "version": __version__, "offline_only": False,
                                   "environment": metadata["environment"], "estimates": plan["estimates"],
                                   "metadata_snapshots": _snapshots(metadata),
                                   "warning": "Checksums and geometry checks do not establish scientific validity or source-data redistribution rights"},
                    "artifacts": artifacts}
        (destination / "manifest.json").write_bytes(_json_bytes(manifest))
        return verify(destination)
    except FileExistsError as exc:
        raise ScoutError("Output appeared during fetch; refusing overwrite") from exc
    except OSError as exc:
        raise ScoutError(f"Cannot save live capsule: {exc}") from exc


def _read_json(path, maximum):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ScoutError("Capsule JSON is missing, linked or exceeds its size limit")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, OSError) as exc:
        raise ScoutError("Cannot read strict capsule JSON") from exc


def verify(directory):
    """Check schema, local paths, hashes and NIfTI geometry without network access."""
    root = Path(directory).resolve()
    manifest = _read_json(root / "manifest.json", 256 * 1024)
    _required(manifest, {"schema_version", "fixture_type", "source", "request", "returned", "provenance", "artifacts"}, "manifest")
    if manifest["schema_version"] != "live-1.0" or manifest["fixture_type"] != "real_remote_data":
        raise ScoutError("Expected live-1.0 real_remote_data capsule")
    source = manifest["source"]
    _required(source, {"id", "url", "license", "citation", "coordinate_space", "units"}, "manifest source")
    if not isinstance(source["id"], str):
        raise ScoutError("Capsule source must be a whitelisted registered section")
    trusted = catalog(source["id"])
    names = artifact_names(source["id"])
    if source["url"] != trusted["source_url"] or source["units"] != "mm":
        raise ScoutError("Capsule source must be the whitelisted section in millimetres")
    if source["coordinate_space"] != "BigBrain histological reference space":
        raise ScoutError("Unexpected live source coordinate space")
    if source["license"] != LICENSE_STATUS or source["citation"] != trusted["configuration_url"]:
        raise ScoutError("Capsule must preserve unverified source terms and configuration reference")
    request_fields = manifest["request"]
    _required(request_fields, {"bbox_mm", "resolution_key", "resolution_mm", "max_decoded_bytes", "max_estimated_working_bytes"}, "manifest request")
    request = _request({"source_id": source["id"], "roi_mm": request_fields["bbox_mm"],
                        "scale": request_fields["resolution_key"],
                        "max_decoded_bytes": request_fields["max_decoded_bytes"],
                        "max_estimated_working_bytes": request_fields["max_estimated_working_bytes"]})
    _vector(request_fields["resolution_mm"], "request resolution", positive=True)
    if request_fields.get("resolution_axes") != "provider xyz":
        raise ScoutError("Requested resolution axes must be explicit")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(names):
        raise ScoutError("Live capsule artifacts must contain data, preview, metadata and reproduce")
    for key, artifact in artifacts.items():
        _required(artifact, {"path", "bytes", "sha256"}, "artifact")
        name = artifact["path"]
        if not isinstance(name, str) or name != names[key]:
            raise ScoutError("Unexpected or unsafe live artifact path")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ScoutError("Unsafe artifact path")
        path = root / name
        size = _budget(artifact["bytes"], "artifact bytes")
        maximum = LIMIT + 4096 if key in ("data", "preview") else 2 * 1024**2
        digest = artifact["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ScoutError("Invalid artifact SHA-256")
        if size > maximum or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ScoutError("Artifact is missing, linked, oversized or outside its capsule")
        if path.stat().st_size != size:
            raise ScoutError("Artifact size mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ScoutError("Artifact checksum mismatch")
    metadata = _read_json(root / names["metadata"], 2 * 1024**2)
    _required(metadata, {"request", "plan", "environment"}, "metadata snapshot")
    _required(metadata["environment"], {"python", "platform", "siibra", "numpy", "nibabel"}, "metadata environment")
    if metadata["request"] != request:
        raise ScoutError("Request differs from the captured metadata request")
    provenance = manifest["provenance"]
    _required(provenance, {"generator", "version", "offline_only", "environment", "estimates", "metadata_snapshots"}, "provenance")
    if provenance["generator"] != "bigbrain-scout" or provenance["offline_only"] is not False:
        raise ScoutError("Invalid live provenance")
    if provenance["environment"] != metadata["environment"] or metadata["environment"].get("siibra") != PIN:
        raise ScoutError("Environment differs from pinned metadata provenance")
    try:
        import numpy as np
        import nibabel as nib
    except ImportError as exc:
        raise ScoutError("Install the existing live extra to verify NIfTI geometry") from exc
    plan = _plan(np, request, metadata)
    if metadata["plan"] != plan or provenance["estimates"] != plan["estimates"]:
        raise ScoutError("Captured plan or estimates do not match source metadata")
    if provenance["metadata_snapshots"] != _snapshots(metadata):
        raise ScoutError("Source metadata snapshot checksum mismatch")
    if not np.allclose(request_fields["resolution_mm"], plan["resolution_mm_provider_xyz"], rtol=0, atol=1e-10):
        raise ScoutError("Recorded request resolution differs from selected scale")
    returned = manifest["returned"]
    _required(returned, {"shape", "affine", "bbox_mm", "resolution_mm", "dtype", "decoded_bytes", "clipped_bbox_mm", "storage_order"}, "returned")
    _vector(returned["shape"], "returned shape", positive=True, integer=True)
    _vector(returned["resolution_mm"], "returned resolution", positive=True)
    _affine(np, returned["affine"])
    _bbox(returned["bbox_mm"], "returned bbox")
    _bbox(returned["clipped_bbox_mm"], "returned clipped bbox")
    _budget(returned["decoded_bytes"], "returned decoded bytes")
    try:
        image = nib.load(root / names["data"])
        actual = _returned(np, image, plan, request, tolerance=1e-5)
    except ScoutError:
        raise
    except Exception as exc:
        raise ScoutError(f"Cannot read NIfTI data: {exc}") from exc
    if returned != actual:
        raise ScoutError("Recorded returned fields differ from the saved NIfTI geometry")
    # Read only the already bounded array, then ensure the preview is exactly
    # the section projection saved by this workflow.
    plane = np.asanyarray(image.dataobj)[:, 0, :]
    if plane.ndim != 2 or plane.dtype != np.uint8:
        raise ScoutError("Expected bounded two-dimensional uint8 data")
    preview = f"P5\n{plane.shape[1]} {plane.shape[0]}\n255\n".encode() + plane.tobytes()
    if (root / names["preview"]).read_bytes() != preview:
        raise ScoutError("Preview differs from the verified section array")
    return manifest
