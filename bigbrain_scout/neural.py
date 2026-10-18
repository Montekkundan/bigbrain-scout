"""Frozen candidate-model audit. Execution is held until explicit CUDA opt-in.

This adapter does not train, install packages or download checkpoints. Outputs
are candidate instances, not validated biological cells or cell-type labels.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import platform
import stat
import time
from unittest.mock import patch

from . import __version__, core, replay

CELLPOSE_VERSION = "4.1.1"
CELLPOSE_REVISION = "a9f8bfcde43033247309e3982747df9fe9f09315"
FRAMEWORK_REVISION = "8d3b968f6cbcd1898da7443500fbef6477f1a1ab"
MODEL_REVISION = "46d13ceec0233e5c32b7d7cec0a0bd65edd5bcd2"
MODEL_REPOSITORY = "AImageLab-Zip/CALHippo-Framework-Models"
MODEL_FILE = "segmentation/cellpose/finetune_v4_astrocytes_big_brain"
CHECKPOINT_BYTES = 609391431
CHECKPOINT_SHA256 = "98f1b62df3db68ad21a54725caacc3b718f7d46c1c1a40eafef9932674a3e534"
MAX_SOURCE_BYTES = 1024**2
RESOLUTIONS = (1, 2, 4)
DIAMETER_UM = 30.0
ARTIFACT_PATHS = {f"instances_{level}um": f"instances-{level}um.raw" for level in RESOLUTIONS}
LIMITATIONS = [
    "Candidate instances are not validated biological cells, neurons, glia or astrocyte identities.",
    "No accuracy, biological count or independent held-out validation is established.",
    "Checkpoint training-section identities are incompletely documented; section-level leakage is unresolved.",
    "One fixed crop and derived block means do not establish generalization or independent acquisitions.",
    "Physical diameter is fixed but Cellpose's pixel-based minimum area and dynamics defaults vary physically by scale.",
    "This is a Scout application protocol, not reproduction of the full CALHippo published benchmark.",
    "CUDA allocator measurements exclude host/process memory and are not a peak-memory guarantee.",
    "Local audit verification establishes consistency, not source authenticity or redistribution rights.",
    "BigBrain source terms and the checkpoint's CC BY-NC-SA 4.0 terms apply independently of Scout's MIT code.",
]


def model_plan():
    """Return dependency-free metadata, without inspecting files or hardware."""
    return {
        "schema_version": "model-plan-1.0",
        "model_id": "calhippo-cellpose-bigbrain-frozen",
        "on_hold": True, "inference_executed": False, "training_executed": False,
        "accuracy_established": False, "gpu_runtime_tested": False,
        "checkpoint": {
            "repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "file": MODEL_FILE,
            "bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256,
            "source_url": f"https://huggingface.co/{MODEL_REPOSITORY}/blob/{MODEL_REVISION}/{MODEL_FILE}",
            "model_card_url": f"https://huggingface.co/{MODEL_REPOSITORY}/blob/{MODEL_REVISION}/README.md",
            "license": "CC-BY-NC-SA-4.0", "access": "Public ungated repository when checked; no weights downloaded",
            "training_section_ids": None, "training_section_provenance_complete": False,
        },
        "runtime": {
            "cellpose_version": CELLPOSE_VERSION, "official_source_revision": CELLPOSE_REVISION,
            "official_source_url": f"https://github.com/MouseLand/cellpose/tree/{CELLPOSE_REVISION}",
            "version_check": "Exact installed distribution version; source revision is a compatibility reference, not runtime attestation",
            "device": "cuda:0 only; no CPU/MPS fallback", "cpu_possible_upstream": True,
            "architecture": "SAM ViT-L; patch size 8; tile size 256; three flow/cell-probability outputs",
            "precision": "float32 model/input; no autocast; CUDA matmul/cuDNN TF32 disabled",
            "download_policy": "Local SHA-256/size-verified checkpoint only; default download hook blocked",
        },
        "protocol": {
            "source": "Verified audit-1.0 section-3905 native 1um source.raw; at most 1 MiB",
            "resolutions_um": list(RESOLUTIONS), "diameter_um": DIAMETER_UM,
            "diameter_pixels": [DIAMETER_UM / resolution for resolution in RESOLUTIONS],
            "input": "Native uint8 grayscale preserved; derived disjoint block means float64 converted to float32; range 0..255",
            "fov": "Unchanged source crop at all levels; no intensity inversion or per-level uint8 rounding",
            "channels": "Pinned Cellpose converts 2D grayscale to [gray, zero, zero], not replicated RGB",
            "normalization": "Cellpose normalize=True: per-channel float32 1st/99th percentiles; constant channels zero; no clipping after normalization",
            "published_protocol_url": f"https://github.com/AImageLab-zip/CALHippo-Framework/blob/{FRAMEWORK_REVISION}/experiments/segmentation/allmodels/allmodels-RCA1.yaml",
            "deviations": [
                "One frozen model, not CALHippo's multimodel fusion/classification pipeline",
                "Scout holds physical diameter at 30um: 30/15/7.5px instead of published 20/30/40px sweeps",
                "Grayscale input rather than the published three-channel slide crop",
                "Explicit float32 without CUDA autocast, rather than upstream default bfloat16 with wrapper AMP",
            ],
            "eval_parameters": {"batch_size": 4, "flow_threshold": 0.6, "cellprob_threshold": 0.0,
                                "normalize": True, "augment": False, "invert": False,
                                "do_3D": False, "min_size": 15, "max_size_fraction": 0.4,
                                "bsize": 256, "tile_overlap": 0.1, "resample": True},
            "instance_encoding": "Little-endian uint32, C-order [row, column]; 0 background",
        },
        "limitations": list(LIMITATIONS),
    }


def _file_identity(record):
    # Reading can legitimately update atime. Only identity/content-change
    # metadata belongs in this guard.
    return (record.st_dev, record.st_ino, record.st_size, record.st_mtime_ns, record.st_ctime_ns)


def _checkpoint(path):
    """Stream a bounded local checkpoint before importing model libraries."""
    target = replay._path(path)
    try:
        before = target.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != CHECKPOINT_BYTES:
            raise core.ScoutError("Trusted checkpoint must be a regular local file with the exact recorded size")
        digest = hashlib.sha256()
        count = 0
        with target.open("rb") as handle:
            opened = handle.fileno()
            import os
            identity = os.fstat(opened)
            if (identity.st_dev, identity.st_ino, identity.st_size) != (before.st_dev, before.st_ino, CHECKPOINT_BYTES):
                raise core.ScoutError("Checkpoint changed before hashing")
            while block := handle.read(1024**2):
                count += len(block)
                if count > CHECKPOINT_BYTES:
                    raise core.ScoutError("Checkpoint grew beyond its exact size")
                digest.update(block)
            after = os.fstat(opened)
        if (count != CHECKPOINT_BYTES or digest.hexdigest() != CHECKPOINT_SHA256
                or (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                or _file_identity(target.stat()) != _file_identity(before)):
            raise core.ScoutError("Checkpoint SHA-256/size mismatch or file changed during hashing")
        return target, before
    except core.ScoutError:
        raise
    except OSError as exc:
        raise core.ScoutError("Cannot read the trusted local checkpoint; no download performed") from exc


def _audit_source(directory):
    from . import audit
    root = replay._path(directory)
    report = audit.verify_audit(root)
    report_bytes = replay._read(root / "report.json", audit.MAX_REPORT)
    if replay._json(report_bytes) != report:
        raise core.ScoutError("Audit report changed after verification")
    source = report["source"]
    raw = replay._read(root / "source.raw", MAX_SOURCE_BYTES)
    if (len(raw) != source["width"] * source["height"]
            or hashlib.sha256(raw).hexdigest() != source["crop_sha256"]):
        raise core.ScoutError("Verified audit source changed or exceeds the 1 MiB crop bound")
    np = importlib.import_module("numpy")
    plane = np.frombuffer(raw, dtype=np.uint8).reshape(source["height"], source["width"])
    return root, report, report_bytes, plane


def _no_download(*args, **kwargs):
    raise core.ScoutError("Default model download is disabled; only the trusted local checkpoint is allowed")


def _runtime():
    try:
        if importlib.metadata.version("cellpose") != CELLPOSE_VERSION:
            raise core.ScoutError(f"Frozen inference requires cellpose=={CELLPOSE_VERSION}")
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise core.ScoutError("Explicit inference requires CUDA; no CPU/MPS fallback")
        models = importlib.import_module("cellpose.models")
        if not callable(getattr(models, "cache_CPSAM_model_path", None)):
            raise core.ScoutError("Installed Cellpose lacks the pinned default-download guard hook")
        return np, torch, models
    except core.ScoutError:
        raise
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise core.ScoutError("Frozen GPU dependencies are unavailable; no install or download performed") from exc
    except Exception as exc:
        raise core.ScoutError(f"Frozen GPU initialization failed; no install or fallback: {exc}") from exc


def _mask(value, shape, np):
    if (type(value) is not np.ndarray or value.ndim != 2 or value.shape != shape
            or value.dtype.kind not in {"u", "i"} or value.min() < 0
            or value.max() > 2**32 - 1 or value.size > MAX_SOURCE_BYTES):
        raise core.ScoutError("Model mask must match the bounded input shape with nonnegative uint32-range integer IDs")
    return np.ascontiguousarray(value, dtype="<u4")


def _measurement(value, name, *, integer=False):
    if integer:
        if type(value) is not int or value < 0:
            raise core.ScoutError(f"Invalid {name} measurement")
    elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise core.ScoutError(f"Invalid {name} measurement")
    return value


def run_model(audit_dir, checkpoint, output, *, allow_gpu=False):
    """Infer candidates only after explicit opt-in and local provenance checks.

    Model execution is untested on actual CUDA hardware in this starter. A
    false or nonboolean opt-in refuses before optional imports or file access.
    """
    if allow_gpu is not True:
        raise core.ScoutError("Model inference is on hold; explicit allow_gpu=True is required")
    destination = replay._path(output)
    if destination.exists():
        raise core.ScoutError("Model output exists; refusing overwrite")
    root, audit_report, report_bytes, plane = _audit_source(audit_dir)
    if root in destination.parents:
        raise core.ScoutError("Model output must not be inside the verified audit")
    model_path, checkpoint_stat = _checkpoint(checkpoint)
    np, torch, models = _runtime()
    # Upstream can silently substitute/download cpsam when a local path vanishes.
    # Disable that hook and verify the constructor retained the trusted path.
    try:
        device = torch.device("cuda:0")
        if _file_identity(model_path.stat()) != _file_identity(checkpoint_stat):
            raise core.ScoutError("Checkpoint changed before model loading")
        with patch.object(models, "cache_CPSAM_model_path", _no_download), \
                patch.object(torch.backends.cuda.matmul, "allow_tf32", False), \
                patch.object(torch.backends.cudnn, "allow_tf32", False):
            model = models.CellposeModel(gpu=True, pretrained_model=str(model_path),
                                        device=device, use_bfloat16=False)
            if (_file_identity(model_path.stat()) != _file_identity(checkpoint_stat) or str(model.pretrained_model) != str(model_path)
                    or getattr(model.device, "type", None) != "cuda"):
                raise core.ScoutError("Model checkpoint or CUDA device differs from the trusted request")
            if hasattr(model.net, "eval"):
                model.net.eval()
            levels, files = [], {}
            height, width = plane.shape
            for resolution in RESOLUTIONS:
                values = plane.reshape(height // resolution, resolution, width // resolution, resolution).mean(axis=(1, 3), dtype=np.float64)
                values = np.ascontiguousarray(values, dtype=np.float32)
                if not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
                    raise core.ScoutError("Derived model input is not finite native-range grayscale")
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    result = model.eval(values, diameter=DIAMETER_UM / resolution,
                                        **model_plan()["protocol"]["eval_parameters"])
                torch.cuda.synchronize(device)
                elapsed = _measurement(time.perf_counter() - started, "runtime_seconds")
                if not isinstance(result, (tuple, list)) or len(result) != 3:
                    raise core.ScoutError("Unexpected pinned Cellpose inference return fields")
                mask = _mask(result[0], values.shape, np)
                ids = np.unique(mask)
                role = f"instances_{resolution}um"
                files[role] = mask.tobytes(order="C")
                levels.append({
                    "resolution_um": resolution, "shape": list(mask.shape),
                    "diameter_pixels": DIAMETER_UM / resolution,
                    "candidate_count": int(np.count_nonzero(ids)),
                    "runtime_seconds": elapsed,
                    "cuda_peak_allocated_bytes": _measurement(torch.cuda.max_memory_allocated(device), "CUDA allocated bytes", integer=True),
                    "cuda_peak_reserved_bytes": _measurement(torch.cuda.max_memory_reserved(device), "CUDA reserved bytes", integer=True),
                    "mask_role": role, "dtype": "uint32", "byte_order": "little", "storage_order": "C",
                })
                del result
    except core.ScoutError:
        raise
    except Exception as exc:
        raise core.ScoutError(f"Frozen CUDA candidate inference failed; no CPU fallback: {exc}") from exc
    plan = model_plan()
    report = {
        "schema_version": "neural-audit-1.0", "kind": "frozen_candidate_resolution_audit",
        "generator": "bigbrain-scout", "generator_version": __version__,
        "on_hold": False, "inference_executed": True, "training_executed": False,
        "accuracy_established": False, "independent_heldout_validation_established": False,
        "gpu_used": True, "source": deepcopy(audit_report["source"]),
        "audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "source_raw_sha256": hashlib.sha256(plane.tobytes()).hexdigest(),
        "checkpoint": plan["checkpoint"], "protocol": plan["protocol"], "runtime_contract": plan["runtime"],
        "levels": levels, "runtime_seconds_total": sum(level["runtime_seconds"] for level in levels),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__, "cellpose": CELLPOSE_VERSION,
                        "cuda_runtime": torch.version.cuda, "device": str(device)},
        "artifacts": {role: {"path": ARTIFACT_PATHS[role], "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()} for role, data in files.items()},
        "limitations": list(LIMITATIONS),
    }
    encoded = core._json_bytes(report)
    # Strict finite JSON and all mask checks happen before creating output.
    if len(encoded) > 256 * 1024:
        raise core.ScoutError("Frozen model report exceeds its size bound")
    try:
        destination.mkdir(parents=True, exist_ok=False)
        for role, data in files.items():
            with (destination / ARTIFACT_PATHS[role]).open("xb") as handle:
                handle.write(data)
        with (destination / "report.json").open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise core.ScoutError("Model output appeared; refusing overwrite") from exc
    except OSError as exc:
        raise core.ScoutError("Cannot write local model output") from exc
    return report
