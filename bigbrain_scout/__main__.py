"""CLI for metadata planning, provenance capsules and a local browser workspace."""

import argparse
from contextlib import ExitStack
from importlib.resources import as_file
import json
from pathlib import Path

from . import resources
from .core import ScoutError, create_demo, load_json, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    planning = subcommands.add_parser("plan", help="Metadata only; no image collection or scale fallback")
    planning_input = planning.add_mutually_exclusive_group(required=True)
    planning_input.add_argument("--request", help="Local request JSON file")
    planning_input.add_argument("--example", choices=("safe", "unsafe"), help="Bundled nonbiological example")
    planning_input.add_argument("--source", help="Fixed live source preset")
    planning.add_argument("--allow-live", action="store_true", help="Explicitly allow live metadata access")
    planning.add_argument("--cache-dir", default="artifacts/siibra-cache")
    subcommands.add_parser("sources", help="List fixed live presets without network access")
    demo = subcommands.add_parser("demo", help="Create a new NONBIOLOGICAL offline capsule")
    demo.add_argument("--request", help="Local request JSON, otherwise the bundled safe example")
    demo.add_argument("--fixture", help="Local fixture JSON, otherwise the bundled synthetic fixture")
    demo.add_argument("--output", required=True)
    verification = subcommands.add_parser("verify", help="Check geometry, paths and artifact checksums")
    verification.add_argument("--capsule", required=True)
    collection = subcommands.add_parser("fetch", help="Explicitly fetch a bounded fixed live request")
    collection_input = collection.add_mutually_exclusive_group(required=True)
    collection_input.add_argument("--source", help="Fixed live source preset")
    collection_input.add_argument("--request", help="Local canonical live request JSON")
    collection.add_argument("--output", required=True, help="New capsule directory; never overwritten")
    collection.add_argument("--allow-live", action="store_true", help="Required opt-in for image collection")
    collection.add_argument("--cache-dir", default="artifacts/siibra-cache")
    replay = subcommands.add_parser("reproduce", help="Replay a verified capsule using trusted package code")
    replay.add_argument("--capsule", required=True)
    replay.add_argument("--output", required=True, help="New capsule directory; never overwritten")
    replay.add_argument("--allow-live", action="store_true", help="Explicitly allow a new live collection")
    replay.add_argument("--cache-dir", help="Optional cache directory; default is a fresh temporary cache")
    comparison = subcommands.add_parser("compare", help="Compare verified geometry and decoded data bytes")
    comparison.add_argument("--first", required=True)
    comparison.add_argument("--second", required=True)
    checking = subcommands.add_parser("check", help="Teammate reproduction check; offline unless explicitly opted into a live fetch")
    checking.add_argument("--output", required=True, help="New evidence directory; never overwritten")
    checking_input = checking.add_mutually_exclusive_group()
    checking_input.add_argument("--allow-live", action="store_true", help="Fetch the fixed native crop and run its CPU audit")
    checking_input.add_argument("--capsule", help="Audit an existing native capsule offline instead of fetching")
    audit = subcommands.add_parser("audit", help="CPU-only classical candidate stability audit; not cell accuracy")
    audit.add_argument("--capsule", required=True, help="Verified native 1um section-3905 capsule")
    audit.add_argument("--output", required=True, help="New audit directory; never overwritten")
    audit_verification = subcommands.add_parser("verify-audit", help="Offline recomputation of saved candidate outputs")
    audit_verification.add_argument("--audit", required=True)
    evaluation = subcommands.add_parser("evaluate", help="Instance matching against explicitly independent reference labels")
    evaluation.add_argument("--prediction", required=True)
    evaluation.add_argument("--reference", required=True)
    evaluation.add_argument("--iou-threshold", type=float, default=0.5)
    subcommands.add_parser("model-plan", help="Inspect the pinned learned-model protocol without running it")
    inference = subcommands.add_parser("infer", help="Hackathon-day CUDA-only frozen-model run, held by default")
    inference.add_argument("--audit", required=True)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--allow-gpu", action="store_true", help="Explicit opt-in to the held model execution")
    workspace = subcommands.add_parser("serve", help="Run a loopback-only browser workspace")
    workspace.add_argument("--output", default="artifacts/workspace", help="Local capsule directory")
    workspace.add_argument("--port", type=int, default=8765)
    workspace.add_argument("--allow-live", action="store_true", help="Explicitly enable fixed section presets with pinned optional dependencies")
    args = parser.parse_args()
    try:
        if args.command == "serve":
            if not 0 <= args.port <= 65535:
                raise ScoutError("Port must be between 0 and 65535")
            from .server import serve
            serve(args.output, args.port, args.allow_live)
            return
        elif args.command == "sources":
            from .live import catalog
            result = {"sources": catalog(), "image_fetch_performed": False}
        elif args.command in {"plan", "fetch"}:
            if getattr(args, "example", None):
                result = plan(resources.load(args.example))
            else:
                if args.source:
                    from .live import defaults
                    request = defaults(args.source)
                else:
                    request = load_json(args.request)
                if args.command == "plan" and isinstance(request, dict) and "source_id" not in request:
                    result = plan(request)
                else:
                    if not args.allow_live:
                        raise ScoutError("Live metadata/fetch requires explicit --allow-live")
                    from .live import prepare, fetch
                    if args.command == "fetch" and (Path(args.output).exists() or Path(args.output).is_symlink()):
                        raise ScoutError("Output already exists; refusing overwrite")
                    context, result = prepare(request, args.cache_dir)
                    if args.command == "fetch":
                        result = fetch(context, result, args.output)
        elif args.command == "demo":
            with ExitStack() as stack:
                request = args.request or stack.enter_context(as_file(resources.resource("safe")))
                fixture = args.fixture or stack.enter_context(as_file(resources.resource("fixture")))
                result = create_demo(request, fixture, args.output)
        elif args.command == "verify":
            from .replay import verify_capsule
            manifest = verify_capsule(args.capsule)
            result = {"verified": True, "fixture_type": manifest["fixture_type"],
                      "warning": manifest["provenance"].get("warning", "Inspect provenance; geometry checks are not scientific validation"),
                      "shape": manifest["returned"]["shape"]}
        elif args.command == "reproduce":
            from .replay import reproduce
            result = reproduce(args.capsule, args.output, allow_live=args.allow_live, cache_dir=args.cache_dir)
        elif args.command == "audit":
            from .audit import run_audit
            result = run_audit(args.capsule, args.output)
        elif args.command == "check":
            from .check import run_check
            result = run_check(args.output, allow_live=args.allow_live, capsule=args.capsule)
            if not result["passed"]:
                print(json.dumps(result, indent=2, allow_nan=False))
                parser.exit(3, "Scout check differs from its recorded regression reference; outputs were preserved\n")
        elif args.command == "verify-audit":
            from .audit import verify_audit
            result = verify_audit(args.audit)
        elif args.command == "evaluate":
            from .evaluation import evaluate_label_files
            result = evaluate_label_files(args.prediction, args.reference, iou_threshold=args.iou_threshold)
        elif args.command == "model-plan":
            from .neural import model_plan
            result = model_plan()
        elif args.command == "infer":
            from .neural import run_model
            result = run_model(args.audit, args.checkpoint, args.output, allow_gpu=args.allow_gpu)
        else:
            from .replay import compare_capsules
            result = compare_capsules(args.first, args.second)
            if not result["matched"]:
                print(json.dumps(result, indent=2, allow_nan=False))
                parser.exit(3, "Scout comparison: decoded data, geometry, source or request differs\n")
    except (ScoutError, OSError, ImportError, RuntimeError) as exc:
        if hasattr(exc, "comparison"):
            print(json.dumps({"output": str(exc.output), "comparison": exc.comparison}, indent=2, allow_nan=False))
            parser.exit(3, f"Scout reproduction differs: {exc}\n")
        parser.exit(2, f"Scout refused: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
