"""Local plan-to-capsule service with fixed source presets and one-use tokens.

This service never accepts source URLs or arbitrary artifact paths from clients.
The optional live adapter is loaded only when explicitly enabled. Downloads and
ZIP creation stay inside the local workspace; this module publishes nothing.
"""

from __future__ import annotations

import base64
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import re
import secrets
import tempfile
import threading
import time
import uuid
import zipfile

from . import core, resources
from .sources import source_ids

MAX_BUDGET_BYTES = 8_388_608
MAX_PENDING_PLANS = 32
MAX_CAPSULES = 64
PLAN_TTL_SECONDS = 30 * 60
REQUEST_FIELDS = {"source_id", "roi_mm", "scale", "max_decoded_bytes", "max_estimated_working_bytes"}
ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
MAX_MANIFEST_BYTES = 1_048_576
MAX_ARTIFACT_BYTES = MAX_BUDGET_BYTES + MAX_MANIFEST_BYTES
MAX_ARCHIVE_BYTES = 4 * MAX_BUDGET_BYTES
ARTIFACT_PATHS = {
    "data": {"data.raw", *(f"{source}.nii" for source in source_ids())},
    "preview": {"preview.pgm"},
    "fixture": {"fixture.json"},
    "reproduce": {"reproduce.py", "README.md"},
    "request": {"request.json"},
    "metadata": {"metadata.json"},
}


class WorkflowError(core.ScoutError):
    """User-facing failure with a stable code for a local HTTP layer."""

    def __init__(self, message, code="invalid_request"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Pending:
    """Private deep-copy snapshots, never returned to API callers."""

    request: dict
    context: object
    plan: dict
    expires_at: float


def _no_symlinks(path):
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise WorkflowError("Workspace paths must not contain symlinks", "unsafe_path")


def _canonical_request(request):
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        raise WorkflowError("Request must contain only source_id, roi_mm, scale, and both byte budgets")
    if not isinstance(request["source_id"], str) or request["source_id"] not in {"synthetic", *source_ids()}:
        raise WorkflowError("Unknown fixed source preset")
    if not isinstance(request["scale"], str) or not 0 < len(request["scale"]) <= 64:
        raise WorkflowError("Scale must be a nonempty exact metadata key")
    roi = request["roi_mm"]
    if not isinstance(roi, list) or len(roi) != 2 or any(not isinstance(row, list) or len(row) != 3 for row in roi):
        raise WorkflowError("ROI must contain two three-coordinate bounds in millimetres")
    for row in roi:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or abs(value) > 1000:
                raise WorkflowError("ROI coordinates must be finite numbers within +/-1000 mm")
            if not math.isfinite(value):
                raise WorkflowError("ROI coordinates must be finite numbers within +/-1000 mm")
    if any(a >= b for a, b in zip(roi[0], roi[1])):
        raise WorkflowError("ROI bounds must have strictly positive extents")
    for key in ("max_decoded_bytes", "max_estimated_working_bytes"):
        value = request[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_BUDGET_BYTES:
            raise WorkflowError(f"{key} must be an integer between 1 and {MAX_BUDGET_BYTES}")
    return deepcopy(request)


def read_pgm(content):
    """Parse a bounded, 8-bit P5 image without treating pixels as header text."""
    if not isinstance(content, bytes) or len(content) > MAX_ARTIFACT_BYTES:
        raise WorkflowError("Preview is missing or too large", "verification_failed")
    position = 0

    def token():
        nonlocal position
        while position < len(content):
            if content[position] in b" \t\r\n\v\f":
                position += 1
            elif content[position] == ord("#"):
                newline = content.find(b"\n", position)
                if newline < 0:
                    raise WorkflowError("Unterminated PGM comment", "verification_failed")
                position = newline + 1
            else:
                break
        start = position
        while position < len(content) and content[position] not in b" \t\r\n\v\f#":
            position += 1
        if start == position or position - start > 32:
            raise WorkflowError("Invalid PGM header token", "verification_failed")
        return content[start:position]

    try:
        magic, width, height, maximum = token(), int(token()), int(token()), int(token())
    except ValueError as exc:
        raise WorkflowError("Invalid PGM numeric header", "verification_failed") from exc
    if magic != b"P5" or maximum != 255 or width <= 0 or height <= 0 or width * height > MAX_BUDGET_BYTES:
        raise WorkflowError("Preview must be a bounded 8-bit grayscale P5 image", "verification_failed")
    if position >= len(content) or content[position] not in b" \t\r\n\v\f":
        raise WorkflowError("Missing PGM raster separator", "verification_failed")
    separator = content[position:position + 2]
    position += 2 if separator == b"\r\n" else 1
    pixels = content[position:]
    if len(pixels) != width * height:
        raise WorkflowError("PGM raster size does not match its dimensions", "verification_failed")
    return {"width": width, "height": height, "pixels_base64": base64.b64encode(pixels).decode("ascii")}


class Workspace:
    """A local, bounded workflow around bundled synthetic or opt-in live sources."""

    def __init__(self, root: Path, allow_live=False):
        self.root = Path(root).absolute()
        _no_symlinks(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self._capsule_root = self.root / "capsules"
        self._cache_root = self.root / "cache"
        for directory in (self._capsule_root, self._cache_root):
            _no_symlinks(directory)
            directory.mkdir(exist_ok=True)
        self.allow_live = bool(allow_live)
        self._plans = {}
        self._expired = deque(maxlen=MAX_PENDING_PLANS)
        self._capsule_count = 0
        self._state_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._fixture = resources.load("fixture")
        safe = resources.load("safe")
        self._defaults = {key: deepcopy(value) for key, value in safe.items() if key in REQUEST_FIELDS}
        self._defaults["source_id"] = "synthetic"
        self._defaults["max_decoded_bytes"] = MAX_BUDGET_BYTES
        self._defaults["max_estimated_working_bytes"] = MAX_BUDGET_BYTES

    def _live(self):
        if not self.allow_live:
            raise WorkflowError("Live fetching is disabled; restart locally with explicit live opt-in", "live_disabled")
        try:
            return importlib.import_module("bigbrain_scout.live")
        except ImportError as exc:
            raise WorkflowError("Optional live adapter is unavailable", "live_unavailable") from exc

    @contextmanager
    def _job(self):
        if not self._job_lock.acquire(blocking=False):
            raise WorkflowError("Another local job is running; wait for it to finish", "busy")
        try:
            _no_symlinks(self.root)
            _no_symlinks(self._capsule_root)
            _no_symlinks(self._cache_root)
            yield
        finally:
            self._job_lock.release()

    def config(self):
        """Return fresh copies of server-owned presets, never user source metadata."""
        sources = [{"id": "synthetic", "label": "Synthetic demo",
                    "description": "Nonbiological fixture. No network required.",
                    "defaults": deepcopy(self._defaults),
                    "scales": [{"key": scale["key"], "label": f"{scale['resolution_mm'][0] * 1000:g} µm"}
                               for scale in self._fixture["source"]["scales"]]}]
        if self.allow_live:
            adapter = self._live()
            for identifier in source_ids():
                defaults = _canonical_request(adapter.defaults(identifier))
                if defaults["source_id"] != identifier:
                    raise WorkflowError("Live adapter defaults do not match their fixed section preset", "live_unavailable")
                sources.append({"id": identifier, "label": f"BigBrain section #{identifier.removeprefix('section-')}",
                                "description": "20 µm through-section sampling. Source data stays local.",
                                "defaults": defaults,
                                "scales": [{"key": key, "label": key.removesuffix("um") + " µm"}
                                           for key in getattr(adapter, "SCALES", {defaults["scale"]: None})]})
        return {"sources": sources, "allow_live": self.allow_live, "max_budget_bytes": MAX_BUDGET_BYTES}

    def _prune_plans(self, now):
        for token in list(self._plans):
            if self._plans[token].expires_at <= now:
                del self._plans[token]
                self._expired.append(token)

    def plan(self, request):
        """Store a private request/context/result snapshot under an expiring token."""
        canonical = _canonical_request(request)
        if canonical["source_id"] != "synthetic" and not self.allow_live:
            raise WorkflowError("Live fetching is disabled", "live_disabled")
        with self._job():
            with self._state_lock:
                self._prune_plans(time.monotonic())
                if len(self._plans) >= MAX_PENDING_PLANS:
                    raise WorkflowError("At most 32 pending plans are allowed; build one or wait for expiry", "plan_limit")
            try:
                if canonical["source_id"] == "synthetic":
                    augmented = {key: deepcopy(value) for key, value in canonical.items() if key != "source_id"}
                    augmented["source"] = deepcopy(self._fixture["source"])
                    context = {"request": augmented, "fixture": deepcopy(self._fixture)}
                    result = core.plan(augmented)
                    result["source_id"] = "synthetic"
                else:
                    context, result = self._live().prepare(deepcopy(canonical), self._cache_root)
                snapshot = _Pending(deepcopy(canonical), deepcopy(context), deepcopy(result), time.monotonic() + PLAN_TTL_SECONDS)
            except core.ScoutError as exc:
                if isinstance(exc, WorkflowError):
                    raise
                raise WorkflowError(str(exc)) from exc
            token = secrets.token_hex(16)
            with self._state_lock:
                self._plans[token] = snapshot
            return {"token": token, "plan": deepcopy(result)}

    def build(self, token):
        """Consume a fresh plan once, build locally, and verify before returning."""
        if not isinstance(token, str) or ID_PATTERN.fullmatch(token) is None:
            raise WorkflowError("Unknown plan token", "unknown_plan")
        with self._job():
            with self._state_lock:
                pending = self._plans.get(token)
                if pending is None:
                    code = "expired_plan" if token in self._expired else "unknown_plan"
                    raise WorkflowError("Plan token expired or is unknown; plan again", code)
                if pending.expires_at <= time.monotonic():
                    del self._plans[token]
                    self._expired.append(token)
                    raise WorkflowError("Plan token expired; plan again", "expired_plan")
                if self._capsule_count >= MAX_CAPSULES:
                    raise WorkflowError("This process has reached its 64 local capsule limit", "capsule_limit")
                del self._plans[token]
                self._capsule_count += 1
            identifier = uuid.uuid4().hex
            destination = self._capsule_root / identifier
            if destination.exists() or destination.is_symlink():
                raise WorkflowError("Generated capsule directory already exists; no overwrite allowed", "unsafe_path")
            try:
                if pending.request["source_id"] == "synthetic":
                    with tempfile.TemporaryDirectory(prefix="scout-request-", dir=self.root) as temporary:
                        request_file, fixture_file = Path(temporary) / "request.json", Path(temporary) / "fixture.json"
                        request_file.write_text(json.dumps(pending.context["request"], allow_nan=False), encoding="utf-8")
                        fixture_file.write_text(json.dumps(pending.context["fixture"], allow_nan=False), encoding="utf-8")
                        core.create_demo(request_file, fixture_file, destination)
                else:
                    self._live().fetch(deepcopy(pending.context), deepcopy(pending.plan), destination)
                manifest = self._verified(identifier)
                preview_path = self._artifact_paths(manifest, destination)["preview"]
                preview = read_pgm(preview_path.read_bytes())
            except (core.ScoutError, OSError) as exc:
                if isinstance(exc, WorkflowError):
                    raise
                raise WorkflowError(f"Local capsule build failed: {exc}", "verification_failed") from exc
            return {"id": identifier, "manifest": deepcopy(manifest), "plan": deepcopy(pending.plan),
                    "preview": preview, "verification": self._summary(manifest)}

    def _directory(self, identifier):
        if not isinstance(identifier, str) or ID_PATTERN.fullmatch(identifier) is None:
            raise WorkflowError("Unknown capsule ID", "unknown_capsule")
        directory = self._capsule_root / identifier
        _no_symlinks(directory)
        if not directory.is_dir() or not directory.resolve().is_relative_to(self._capsule_root):
            raise WorkflowError("Unknown capsule ID", "unknown_capsule")
        return directory

    def _artifact_paths(self, manifest, directory):
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts or "preview" not in artifacts:
            raise WorkflowError("Capsule has no recorded preview artifacts", "verification_failed")
        paths, total = {}, 0
        for key, item in artifacts.items():
            if not isinstance(item, dict) or key not in ARTIFACT_PATHS or not isinstance(item.get("path"), str) or item["path"] not in ARTIFACT_PATHS[key]:
                raise WorkflowError("Capsule artifact is outside the fixed download whitelist", "unsafe_path")
            path = directory / item["path"]
            _no_symlinks(path)
            if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
                raise WorkflowError("Capsule artifact is missing or too large", "verification_failed")
            total += path.stat().st_size
            paths[key] = path
        if total > MAX_ARCHIVE_BYTES:
            raise WorkflowError("Recorded local artifacts exceed the archive size limit", "verification_failed")
        return paths

    def _verified(self, identifier):
        directory = self._directory(identifier)
        manifest_path = directory / "manifest.json"
        _no_symlinks(manifest_path)
        if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise WorkflowError("Manifest is missing or too large", "verification_failed")
        try:
            manifest = core.load_json(manifest_path)
        except core.ScoutError as exc:
            raise WorkflowError("Cannot read capsule manifest", "verification_failed") from exc
        if not isinstance(manifest, dict):
            raise WorkflowError("Manifest must be an object", "verification_failed")
        if not isinstance(manifest.get("fixture_type"), str) or manifest["fixture_type"] not in {"synthetic", "real_remote_data"}:
            raise WorkflowError("Unknown capsule fixture type", "verification_failed")
        self._artifact_paths(manifest, directory)
        try:
            verified = core.verify_capsule(directory) if manifest.get("fixture_type") == "synthetic" else self._live().verify(directory)
        except (core.ScoutError, OSError) as exc:
            if isinstance(exc, WorkflowError):
                raise
            raise WorkflowError(f"Capsule verification failed: {exc}", "verification_failed") from exc
        return verified

    @staticmethod
    def _summary(manifest):
        return {"verified": True, "fixture_type": manifest["fixture_type"],
                "warning": manifest.get("provenance", {}).get("warning", "Real-data local capsule; inspect its provenance before scientific use")}

    def verify(self, identifier):
        """Recheck local bytes and geometry without performing any new fetch."""
        with self._job():
            return self._summary(self._verified(identifier))

    def capsule(self, identifier):
        """Return a freshly verified manifest copy for a strict opaque capsule ID."""
        with self._job():
            return deepcopy(self._verified(identifier))

    def archive(self, identifier):
        """Return ZIP bytes for verified recorded files only, never directory globs."""
        with self._job():
            manifest = self._verified(identifier)
            directory = self._directory(identifier)
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
                for key, path in self._artifact_paths(manifest, directory).items():
                    content = path.read_bytes()
                    recorded = manifest["artifacts"][key]
                    if len(content) != recorded["bytes"] or hashlib.sha256(content).hexdigest() != recorded["sha256"]:
                        raise WorkflowError("Artifact changed during local archive creation", "verification_failed")
                    archive.writestr(recorded["path"], content)
            return stream.getvalue()
