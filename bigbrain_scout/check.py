"""Bounded teammate checks, with remote acquisition only after explicit opt-in."""

from importlib.resources import as_file
import platform

from . import __version__, core, replay, resources


def compare_reference(report):
    """Compare verified outputs to an aggregate regression record, not cell truth.

    Callers must pass the report returned by run_audit/verify_audit, not unchecked
    JSON. Runtime, Python patch version and NIfTI header hashes are not compared.
    """
    reference = resources.load("audit-reference")
    observed = {
        "source": {key: report["source"][key] for key in reference["source"]},
        "settings": report["settings"],
        "environment": {key: report["environment"][key] for key in reference["environment"]},
        "levels": [{key: level[key] for key in ("resolution_um", "candidate_count")}
                   for level in report["levels"]],
        "mask_sha256": {key: report["artifacts"][key]["sha256"] for key in reference["mask_sha256"]},
    }
    differences = [key for key, value in observed.items() if value != reference[key]]
    return {"matched": not differences, "differences": differences, "observed": observed,
            "reference_id": reference["id"], "reference_is_ground_truth": False}


def run_check(output, *, allow_live=False, capsule=None):
    """Create a fresh evidence directory; never install packages or run a model.

    The default needs only the Python standard library. --allow-live explicitly
    collects the fixed native preset. A supplied local capsule stays offline.
    Partial evidence is retained on failure; existing output is never reused.
    """
    if type(allow_live) is not bool or (allow_live and capsule is not None):
        raise core.ScoutError("Choose an existing capsule or explicit live opt-in, not both")
    root = replay._path(output)
    if root.exists():
        raise core.ScoutError("Check output already exists; choose a new directory")
    if allow_live or capsule is not None:
        # Refuse missing scientific packages before creating partial outputs.
        from .audit import _runtime
        _runtime()
        try:
            import nibabel  # noqa: F401
        except ImportError as exc:
            raise core.ScoutError("Install Scout's audit extra for a local audit, or live extra to fetch") from exc
    try:
        core.plan(resources.load("unsafe"))
    except core.ScoutError as exc:
        if "Decoded ROI" not in str(exc):
            raise core.ScoutError("Bundled unsafe request failed for an unexpected reason") from exc
    else:
        raise core.ScoutError("Unsafe request was not refused")
    root.mkdir(parents=True, exist_ok=False)
    with as_file(resources.resource("safe")) as request, as_file(resources.resource("fixture")) as fixture:
        manifest = core.create_demo(request, fixture, root / "synthetic")
    replayed = replay.reproduce(root / "synthetic", root / "synthetic-replayed")
    expected = "69c71336771f104a66f9aa5fe5aa098feae18ad12d7ca599e2a83336a77780c7"
    synthetic_matched = (replayed["comparison"]["matched"]
                         and manifest["artifacts"]["data"]["sha256"] == expected)
    result = {
        "schema_version": "check-1.0", "scout_version": __version__,
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine()},
        "passed": synthetic_matched,
        "synthetic": {"matched": synthetic_matched, "fixture_type": "synthetic_nonbiological",
                      "unsafe_request_refused": True, "decoded_sha256": manifest["artifacts"]["data"]["sha256"]},
        "real_audit": {"executed": False}, "gpu_used": False, "accuracy_claimed": False,
        "limitations": [
            "A passing check establishes software consistency, not biological cell accuracy.",
            "The real regression record is one fixed crop, not a representative brain sample.",
            "Remote access and metadata may change; mismatches must be inspected, not relabeled as success.",
            "This command cannot certify a second computer or independently audited network transfer.",
            "Keep real tissue, masks and caches local until source redistribution terms are reviewed.",
        ],
    }
    if allow_live or capsule is not None:
        from . import audit, live
        if allow_live:
            request = {**live.defaults(), "scale": "1um"}
            context, plan = live.prepare(request, root / "cache")
            capsule = root / "native"
            live.fetch(context, plan, capsule)
        report = audit.run_audit(capsule, root / "audit")
        comparison = compare_reference(report)
        result["real_audit"] = {
            "executed": True, "acquisition": "explicit_live_fetch" if allow_live else "existing_local_capsule",
            "reference_matched": comparison["matched"], "comparison": comparison,
            "independent_download_proven": False,
        }
        result["passed"] = synthetic_matched and comparison["matched"]
    with (root / "check.json").open("xb") as stream:
        stream.write(core._json_bytes(result))
    return result
