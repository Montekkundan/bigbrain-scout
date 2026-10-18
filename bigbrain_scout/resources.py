"""Packaged examples and aggregate reference signatures, never real image data."""

from copy import deepcopy
from importlib.resources import files
import json

from .core import ScoutError

NAMES = {"safe": "safe-request.json", "unsafe": "unsafe-request.json",
         "fixture": "synthetic-section.json", "audit-reference": "audit-reference.json"}


def resource(name):
    """Return only a known packaged resource, never a client-supplied path."""
    if not isinstance(name, str) or name not in NAMES:
        raise ScoutError("Unknown bundled example")
    return files("bigbrain_scout").joinpath("data", NAMES[name])


def load(name):
    """Read a fresh default copy without depending on the working directory."""
    try:
        value = json.loads(resource(name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScoutError("Cannot read bundled example") from exc
    return deepcopy(value)
