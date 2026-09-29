"""CLI for offline metadata planning and NONBIOLOGICAL capsule preparation."""

import argparse
import json
from pathlib import Path

from .core import ScoutError, create_demo, load_json, plan, verify_capsule


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    planning = subcommands.add_parser("plan", help="Metadata only; no network or scale fallback")
    planning.add_argument("--request", required=True)
    demo = subcommands.add_parser("demo", help="Create a new NONBIOLOGICAL offline capsule")
    demo.add_argument("--request", default=str(root / "examples" / "safe-request.json"))
    demo.add_argument("--fixture", default=str(root / "fixtures" / "synthetic-section.json"))
    demo.add_argument("--output", required=True)
    verification = subcommands.add_parser("verify", help="Check geometry, paths and artifact checksums")
    verification.add_argument("--capsule", required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = plan(load_json(args.request))
        elif args.command == "demo":
            result = create_demo(args.request, args.fixture, args.output)
        else:
            manifest = verify_capsule(args.capsule)
            result = {"verified": True, "fixture_type": manifest["fixture_type"],
                      "warning": manifest["provenance"]["warning"], "shape": manifest["returned"]["shape"]}
    except (ScoutError, OSError) as exc:
        parser.exit(2, f"Scout refused: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
