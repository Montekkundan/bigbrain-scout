"""Pinned, narrow section #3905 example; metadata only unless --fetch is set.

NeuroglancerVolume is an internal siibra API: keep its use isolated here.
The estimated chunk allocation is NOT a hard process-memory ceiling.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from itertools import product

LIMIT = 8 * 1024**2
PIN = "1.0.1a22"
SOURCE = "https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3905/precomputed"
CONFIG = "https://github.com/FZJ-INM1-BDA/siibra-configurations/blob/master/features/images/sections/cellbody/73c1fa55-d099-4854-8cda-c9a403c6080a_bigbrain_1um_3905.json"
LOWER = [7.6, 8.0, 7.6]
UPPER = [8.0, 10.0, 8.0]
RESOLUTION = [0.004, 0.020, 0.004]


def prepare():
    # This setting must precede siibra import. Explicit fetch max_bytes is also set.
    os.environ["SIIBRA_MAX_FETCH_SIZE_BYTES"] = str(LIMIT)
    os.environ.setdefault("SIIBRA_CACHEDIR", str(Path("artifacts/siibra-cache").resolve()))
    if importlib.metadata.version("siibra") != PIN:
        raise RuntimeError(f"This internal-API example requires siibra=={PIN}")
    import numpy as np
    import siibra
    from siibra.volumes.providers.neuroglancer import NeuroglancerVolume

    roi = siibra.locations.BoundingBox(LOWER, UPPER, space="bigbrain")
    sections = siibra.features.get(roi, "CellbodyStainedSection")
    section = next((s for s in sections if "#3905:" in s.name), None)
    if section is None:
        raise RuntimeError("Section #3905 not found for the fixed tiny ROI")
    source = section.providers["neuroglancer/precomputed"]
    if source != SOURCE:
        raise RuntimeError("Source changed; review metadata before fetching")
    bbox = section.get_boundingbox(clip=False)
    lower = np.maximum(LOWER, tuple(bbox.minpoint))
    upper = np.minimum(UPPER, tuple(bbox.maxpoint))
    if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
        raise RuntimeError("No finite positive overlap with section metadata bounds")
    clipped = siibra.locations.BoundingBox(lower.tolist(), upper.tolist(), space="bigbrain")
    volume = NeuroglancerVolume(source)
    scale = next((s for s in volume.scales if np.allclose(s.res_mm, RESOLUTION,
                  rtol=0, atol=1e-9)), None)
    if scale is None:
        raise RuntimeError("Requested exact anisotropic scale unavailable; no fallback")
    voxel_bbox = clipped.transform(np.linalg.inv(scale.affine), space=None)
    voxel_min = np.asarray(tuple(voxel_bbox.minpoint))
    voxel_max = np.asarray(tuple(voxel_bbox.maxpoint))
    chunks = np.asarray(scale.chunk_sizes, dtype=int)
    offset = np.asarray(scale.voxel_offset)
    shape = np.ceil(voxel_max).astype(int) - np.floor(voxel_min).astype(int)
    if np.any(shape <= 0) or np.any(shape > 32767) or np.any(chunks <= 0):
        raise RuntimeError("Invalid or excessive voxel/chunk dimensions")
    if np.any(voxel_min < offset - 1e-6) or np.any(voxel_max > offset + scale.size + 1e-6):
        raise RuntimeError("Transformed ROI falls outside the source voxel bounds")
    chunk_min = np.floor((voxel_min - offset) / chunks).astype(int)
    chunk_max = np.ceil((voxel_max - offset) / chunks).astype(int)
    mosaic_shape = (chunk_max - chunk_min) * chunks
    itemsize = volume.dtype.itemsize
    roi_bytes = int(np.prod(shape)) * itemsize
    mosaic_bytes = int(np.prod(mosaic_shape)) * itemsize
    one_chunk_bytes = int(np.prod(chunks)) * itemsize
    if max(roi_bytes, mosaic_bytes + one_chunk_bytes) > LIMIT:
        raise RuntimeError("ROI or rounded chunk allocation exceeds the example budget")
    info = volume._info  # pinned internal metadata; keep the inspected snapshot
    if info.get("num_channels") != 1:
        raise RuntimeError("This starter only supports single-channel sections")
    metadata = {
        "source_url": source,
        "feature_name": section.name,
        "feature_id": section.id,
        "configuration_url": CONFIG,
        "configuration_selection": os.environ.get("SIIBRA_USE_CONFIGURATION", "siibra-1.0.1-alpha.22"),
        "space": "BigBrain histological reference space",
        "units": "mm",
        "requested_bbox_mm": [LOWER, UPPER],
        "effective_bbox_mm": [lower.tolist(), upper.tolist()],
        "requested_voxel_sizes_mm_provider_xyz": RESOLUTION,
        "available_scales": [{"key": s.key, "voxel_sizes_mm": s.res_mm.tolist()}
                             for s in volume.scales],
        "selected_scale_key": scale.key,
        "estimated_shape_provider_xyz": shape.tolist(),
        "estimated_decoded_roi_bytes": roi_bytes,
        "estimated_chunk_mosaic_plus_one_chunk_bytes": mosaic_bytes + one_chunk_bytes,
        "budget_bytes": LIMIT,
        "estimate_warning": "Not network bytes or a guaranteed peak-memory ceiling",
        "source_info": info,
        "source_transform_nm": volume.transform_nm.tolist(),
        "environment": {"python": platform.python_version(),
                        "siibra": PIN,
                        "numpy": importlib.metadata.version("numpy"),
                        "nibabel": importlib.metadata.version("nibabel")},
    }
    return section, clipped, metadata


def run(fetch=False, output=None):
    if fetch and Path(output).exists():
        raise FileExistsError("Output already exists; choose a fresh capsule directory")
    section, roi, metadata = prepare()
    print(json.dumps({k: v for k, v in metadata.items()
                      if k not in ("source_info", "source_transform_nm")}, indent=2))
    if not fetch:
        print("METADATA ONLY: no image fetch requested.")
        return
    import nibabel as nib
    import numpy as np

    output = Path(output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a fresh capsule directory")
    image = section.fetch(format="neuroglancer/precomputed", voi=roi,
                          resolution_mm=0.004, max_bytes=LIMIT)
    if image is None:
        raise RuntimeError("siibra returned no image")
    array = np.asanyarray(image.dataobj)
    actual_sizes = np.linalg.norm(image.affine[:3, :3], axis=0)
    # siibra currently returns provider axes in reversed order. Do not relabel
    # them xyz; preserve the affine and compare the size multiset explicitly.
    if not np.allclose(sorted(actual_sizes), sorted(RESOLUTION), rtol=0, atol=1e-7):
        raise RuntimeError(f"Unexpected returned voxel sizes {actual_sizes}; refusing capsule")
    if array.nbytes > LIMIT or array.size == 0:
        raise RuntimeError("Unexpected returned data size")
    plane = np.squeeze(array)
    if plane.ndim != 2 or plane.dtype != np.uint8:
        raise RuntimeError("Expected a two-dimensional uint8 section preview")
    corners = np.array([image.affine @ np.array([*corner, 1])
                        for corner in product(*[(0, n) for n in image.shape])])[:, :3]
    actual_bbox = [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()]
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "section-3905.nii"
    nib.save(image, data_path)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    preview = output / "preview.pgm"
    preview.write_bytes(f"P5\n{plane.shape[1]} {plane.shape[0]}\n255\n".encode() + plane.tobytes())
    manifest = {
        "manifest_version": "live-example-0.1",
        "fixture_type": "real_remote_data",
        "request": metadata,
        "returned": {"shape_array_axis_order": list(image.shape),
                     "affine_mm": image.affine.tolist(),
                     "voxel_sizes_mm_array_axis_order": actual_sizes.tolist(),
                     "bbox_mm_voxel_edge_convention": actual_bbox,
                     "dtype": str(array.dtype), "decoded_bytes": array.nbytes},
        "files": {"data": data_path.name, "sha256": digest, "preview": preview.name,
                  "preview_sha256": hashlib.sha256(preview.read_bytes()).hexdigest()},
        "license": {"status": "Source data terms not verified for redistribution",
                    "action": "Keep data local; review source dataset terms before sharing"},
        "citation": {"dataset_configuration": CONFIG,
                     "project_information": "https://bigbrainproject.org/",
                     "note": "Confirm the exact dataset-version citation with a domain contributor"},
        "reproduction": "Run the same fixed request into a new output directory; compare returned metadata and data arrays. Remote content may change. This run does not prove independent reproduction.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"FETCHED {list(image.shape)}, {array.nbytes} decoded bytes, sha256 {digest}")
    print(f"Saved local capsule: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Explicitly fetch the fixed tiny ROI")
    parser.add_argument("--output", default="artifacts/section-3905")
    args = parser.parse_args()
    try:
        run(args.fetch, args.output)
    except (ValueError, RuntimeError, FileExistsError, ImportError,
            importlib.metadata.PackageNotFoundError) as exc:
        print(f"Scout stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
