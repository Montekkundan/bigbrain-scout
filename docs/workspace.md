# Local workspace

## Start

From this repository checkout or an installed wheel, Python 3.11 or 3.12:

```sh
python -m bigbrain_scout serve --output artifacts/workspace --port 8765
```

Open `http://127.0.0.1:8765`. Use `Ctrl+C` in the terminal to stop. This is a local demo server, not a public deployment target. Only `127.0.0.1` is supported; do not proxy or expose it to the Internet. The wheel bundles its fixture, requests and browser assets, so no checkout is needed at runtime. The installed console command is `bigbrain-scout`.

## Offline demo

1. Choose **Synthetic demo**. It is a 4 × 4 × 1 nonbiological uint8 fixture.
2. Leave the default ROI, then **Check request**. Expected ROI: 2 × 3 × 1; decoded estimate: 6 bytes; working estimate: 26 bytes.
3. Preflight performs no image collection. The table shows requested and planned voxel-edge bounds, not a fabricated returned result.
4. **Build capsule** creates a fresh local directory and verifies it before showing the actual grayscale bytes.
5. **Verify capsule** rechecks geometry, paths, sizes and SHA-256 values. **Download ZIP** exports only recorded files, after re-verification.
6. Extract the ZIP and run `python -m bigbrain_scout reproduce --capsule /path/to/extracted --output /path/to/new-capsule`, then `compare --first /path/to/extracted --second /path/to/new-capsule`. Trusted replay recreates the ROI and compares its decoded values without executing exported scripts. Existing output directories are refused.
7. Edit any input. Build and download controls become unavailable until a fresh check/build. For a safe refusal test, set decoded budget to 5 bytes.

Existing local capsules are retained when you change requests or restart. The browser does not list old capsules; the CLI can verify their directories. No files are uploaded.

## Real sections #3905, #3954 and #4003

Install only the existing `live` extra and explicitly opt in:

```sh
uv sync --extra live
uv run python -m bigbrain_scout serve --allow-live
```

Use a section preset and its default 4 µm request first. Changing the source replaces its defaults and invalidates previous controls. The metadata lookup may use the network and initializes a local cache. It never intentionally downloads image chunks during preflight. **Fetch & build capsule** is the separate explicit collection step. This version pins `siibra==1.0.1a22`, whitelists three section sources and isolates its internal metadata API in `bigbrain_scout/live.py`.

The adapter requires exact published scale metadata, clips the requested world-space box, derives provider/array axis transforms, checks ROI and chunk allocation estimates, and compares returned shape, dtype, affine, resolution and bounds. Section sampling is anisotropic: 4 × 20 × 4 µm at the default scale. The actual NIfTI array order is recorded explicitly rather than mislabeled as world XYZ. The preview follows array orientation and is not an anatomical viewer.

A live capsule contains `section-N.nii` for its selected section, `preview.pgm`, `metadata.json`, `README.md` and `manifest.json`. Metadata includes captured source scale info and transform with snapshot hashes, exact request, planned geometry and environment versions. The legacy section #3905 contract remains supported. Repeated cached results are not independent re-download proof; remote content can change.

```sh
python -m bigbrain_scout verify --capsule artifacts/workspace/capsules/OPAQUE_ID
```

Live verification requires numpy/nibabel from the existing extra, but no network. Never execute untrusted capsule instructions automatically. Checksums detect inconsistency, not an authenticated origin: somebody who changes every file and hash can create a different self-consistent capsule.

## Trusted replay and comparison

```sh
uv run python -m bigbrain_scout reproduce --capsule artifacts/workspace/capsules/OPAQUE_ID --output artifacts/replayed-capsule --allow-live
uv run python -m bigbrain_scout compare --first artifacts/workspace/capsules/OPAQUE_ID --second artifacts/replayed-capsule
```

Replay verifies the original before using its request. It refuses unrecognized sources, schema fields, symlinks, unsafe budgets and output overwrites. It never executes capsule-provided scripts or README instructions. Live replay needs a new process with no previously loaded siibra; its default cache is a fresh sibling directory, recorded in the result. An explicit `--cache-dir` is allowed but reported as potentially reused. Keep both cache and capsule directories local.

The comparison's `matched` field requires the same source identity, request, geometry (absolute tolerance 0.00001 mm) and exact decoded uint8 values. Metadata snapshots, environment versions and container/header hashes have separate fields; they can differ without an image-value mismatch. Refusal exits `2`; a verified but differing comparison exits `3`, with output retained. Network bytes and independent re-download proof remain unknown even when replay matches.

## Limits and local request guards

- Per request: at most 8 MiB decoded budget and 8 MiB conservative working estimate budget. Working estimate = decoded ROI + touched-chunk mosaic + one chunk. Neither is a hard process-memory cap or network estimate.
- One active job per process. At most 32 pending plans, expiring after 30 minutes. Build consumes a token once. At most 64 builds per process; restarting resets this count, so this is not a total disk quota.
- Fixed source presets only, no user-provided URLs or output paths. Opaque capsule IDs, no symlink paths, strict artifact names and bounded archive size.
- Exact loopback Host checks, same-origin checks, mutation token, bounded JSON bodies, no CORS and restrictive content policy. This reduces browser-origin risks; it is not authentication against another local process or filesystem attacker.
- Failed collection may leave a partial local directory for inspection. It is not offered as a verified capsule. Check again to retry with a fresh token.
- Source data terms and exact dataset-version citation need domain review. Keep live capsules local. Scientific/anatomical validation, independently audited second-computer reproduction, arbitrary sources and production hardening remain outside this v1.

## For the hackathon

Disclose the pre-event branch as a baseline. Keep remaining tasks concrete: domain review of axes/citation/source terms, independent reproduction on a second computer and usability feedback. The defined v1 code need not be postponed until the event. Organizer confirmation of any prize rules is separate from the tool being usable.
