# Verification record

## Hackathon CPU handoff, October 18, 2026

The dated sections below this one preserve preparation evidence. Statements that a branch was unpushed describe those historical checks, not its current GitHub state. Check the branch and Actions page for delivery/CI status.

- Python 3.11.13 with the existing scientific environment: **262 tests passed, zero skips**. A separate clean environment containing only the built Scout wheel and its `audit` extra also passed all 262 tests, without siibra, Torch, Cellpose or GPU packages.
- Python 3.12.11 with only the Scout wheel installed: **262 tests discovered, 153 passed, 109 optional scientific tests skipped**. The isolated installed-wheel CLI/HTTP/export/replay smoke check passed outside the source directory. This does not imply that skipped scientific tests ran.
- New `check --output ...` runs the actual synthetic extraction, verifies trusted replay, compares the fixture data with its regression hash and confirms unsafe-request refusal. It uses no runtime packages and performs no remote access by default.
- `check --allow-live` explicitly fetches the fixed native section-3905 request, rather than the 4 micrometre extraction defaults. An actual run in `artifacts/hackathon-live-check-20261018/`, using a newly created cache directory, completed acquisition, capsule verification, CPU auditing and regression comparison successfully.
- That new real audit reproduced the original 400 x 400 crop geometry, decoded/source hashes, counts **69 / 73 / 106** and all three binary-mask hashes. `verify-audit` independently recomputed its masks and deterministic metrics successfully. This is another run on the same computer, not independent second-computer evidence or proof of audited network transfer.
- A clean wheel installation containing the pinned `audit` extra reproduced the real saved-capsule result in isolated Python mode outside the checkout. The installed packages were Scout, NumPy 2.4.6, SciPy 1.17.1, nibabel 5.4.2, packaging, typing-extensions and importlib-resources. No siibra or model runtime was needed for this offline scientific check.
- The packaged aggregate regression record contains counts, settings, geometry and hashes only. It contains no tissue image, candidate mask, checkpoint, private local path or expert annotation. A changed reference comparison returns exit code 3 with its outputs retained; refusal returns 2. The real reference is not biological ground truth.
- The lockfile matches the declared extras. Wheel build, Python compilation, JavaScript syntax and whitespace checks passed. An existing setuptools license-table deprecation warning is non-fatal; no standalone linter/typechecker is configured for Scout. A separate read-only review found no remaining actionable release blocker in the new checks.
- The local first-run checks are not GitHub Actions results. CI now includes offline installation checks and CPU scientific-contract jobs on Linux/macOS with Python 3.11/3.12. CI does not download real tissue or run neural inference.

Remaining external validation: a teammate must run the README quickstart independently and return `check.json` plus the commit tested; a domain expert should assess the region, task and biological interpretation. No real cell-accuracy result, learned-model execution, second-computer outcome, blog publication or organizer endorsement is claimed.

## September 29, 2026

## Offline core

- 33 `unittest` tests passed locally on Python 3.11.13 and Python 3.12.
- The plan/demo/verify sequence passed; a new raw ROI was regenerated from the preserved synthetic fixture and matched the recorded SHA-256.
- The unsafe example was refused before data creation.
- Tests include invalid geometry/units, unavailable scale, clipping, decoded/chunk budgets, overwrites, checksum tampering, path traversal and symlink escape.
- The CLI validates its explicit required manifest contract, not the full JSON Schema specification. Scientific/anatomical validity is not tested.

## Tiny real section #3905

Environment: Python 3.11.13, siibra 1.0.1a22, NumPy 2.4.6, nibabel 5.4.2. Exact dependency resolution is in `uv.lock`.

- Metadata-only inspection passed before the image fetch.
- Fixed requested ROI: `[7.6, 8.0, 7.6]`–`[8.0, 10.0, 8.0]` mm, clipped to the section's metadata bounds.
- Requested available scale: `4,1,4`, provider voxel sizes `[0.004, 0.020, 0.004]` mm.
- Preflight predicted shape `[101, 1, 101]`, decoded ROI bytes `10,201`, chunk mosaic plus one chunk `196,608` bytes, below the 8 MiB example budget. This is not measured network traffic or peak RAM.
- A real fetch completed with actual shape `[101, 1, 101]`, dtype `uint8` and `10,201` decoded bytes. The returned voxel-size check passed.
- A local NIfTI, PGM preview and actual metadata/checksum manifest were saved in ignored `artifacts/section-3905/`. They are not redistributed in this repository.

The live example is not covered by network-free CI and does not establish generic ROI safety, source-data licensing, anatomical relevance, independent re-fetch equivalence or persistence of historical siibra bugs. Those remain explicit contributor/domain-review work.

## Pre-hackathon workspace branch

The starter checks above remain baseline evidence, not a claim about every new source. The `pre-hackathon-mvp` workflow adds an independently checked `live-1.0` capsule contract alongside the synthetic `1.0` contract.

- A new live-adapter run prepared metadata only, fetched the default patch, exported a fresh capsule and verified its source snapshot hashes, artifact hashes, saved NIfTI geometry and exact preview bytes.
- Actual shape `[101, 1, 101]`, uint8, decoded bytes `10,201`. The stricter working model includes the ROI as well as mosaic and one chunk: `206,809` estimated bytes. It is not a network-byte measurement or memory guarantee.
- A local capsule is retained under ignored `artifacts/live-adapter-check-20260929/`. Real image data are not tracked in Git.
- New tests cover one-use/expiring plans, immutable snapshots, job/capsule limits, optional-live refusal, oversized/invalid requests, path/symlink safety, ZIP file whitelist, mocked live geometry/metadata drift and localhost Host/Origin/mutation-token guards.
- Live unit tests use mocked metadata/provider calls and optional numpy/nibabel. They are not network fetch tests. The separate actual fetch is a different proof surface.

## First-MVP regression checks (historical)

- Python 3.11.13: all 98 tests passed with the existing optional live dependencies installed.
- Python 3.12: 98 tests discovered, 72 passed and 26 optional live tests skipped because NumPy/nibabel were not installed in that environment. The offline workspace and HTTP tests passed.
- JavaScript syntax (`node --check`), Python compilation (`compileall`) and `git diff --check` passed. No separate linter is configured.
- An independent review found three edge cases: one-pixel-wide live previews, restoring the download link after successful verification retry, and non-object manifests crashing the CLI. Each was fixed and covered by regression tests. The reviewer's repeat checks found no additional must-fix issues.
- The GitHub Actions configuration was updated for the workspace and mocked live contract tests, but has not run on GitHub for these unpushed changes. Local tests are not CI evidence.

## Browser and exported capsule checks

The native Codex in-app browser exercised the local application, not a deployed service.

- Offline: check request, build capsule, preview actual fixture bytes, reverify and download ZIP completed. The downloaded ZIP was extracted locally and passed CLI verification. Its preserved reproduction script regenerated the raw ROI with the same SHA-256: `69c71336771f104a66f9aa5fe5aa098feae18ad12d7ca599e2a83336a77780c7`.
- Live: metadata-only preflight, explicit fetch/build and reverify completed for section #3905. The final browser capsule is `artifacts/mvp-workspace/capsules/ead5e19374894ce09f9bde0f4177f63e/`, with actual shape `[101, 1, 101]` and `10,201` decoded bytes.
- The UI shows requested versus saved returned bounds, separate planned and returned sampling/shape, the actual grayscale preview, and verification/redistribution caveats. Editing a request invalidates its plan and capsule controls.
- Narrow-screen live and offline workflows were checked at a measured CSS viewport of 390 x 844. Document scroll width was also 390, with no page-level horizontal overflow. The precision table has its own horizontal scroll area. This is responsive browser testing, not testing on a physical phone.
- The temporary viewport override was reset. Final desktop evidence measured 1159 x 1298 CSS pixels with matching document scroll width. Final desktop and mobile screenshots were inspected; no browser warning/error logs were present after the final live verification.

Changes remain uncommitted and unpushed on `pre-hackathon-mvp`. `main` and `origin/main` remain at baseline commit `17ab60284760218f9dfe0ec6363ec4ceacedd12d`. Real capsules and cache files are ignored, not tracked in Git.

## Expanded v1 final verification

The expanded v1 is implemented locally, not deployed or organizer-approved. The first-MVP checks above remain historical evidence; the following checks cover the integrated v1.

### Tests and installed package

- Python 3.11.13 with the pinned live environment: all **156 tests passed**, zero skips.
- Python 3.12.11 without NumPy/nibabel: **156 tests discovered, 122 passed, 34 optional tests skipped**. This is not full live-dependency coverage on Python 3.12.
- New coverage includes packaged fixture/default integrity, all three source presets, explicit live refusal, trusted replay that never executes capsule code, exact decoded-value comparison, metadata/environment/header differences, strict finite/duplicate-free JSON, limits, mismatches/exit codes and retained outputs.
- HTTP regressions cover ZIP extraction plus trusted verification, exclusion of unrecorded private files, tampering refusal for verify/manifest/download, failed collection followed by a fresh-plan retry, and preserved capsules versus lost pending plans after restart.
- Independent integration review found a missing synthetic `clipped_bbox_mm` comparison. The fix now requires equality with the trusted planned bounds; a forged-bounds regression refuses both verification and replay before output creation. The reviewer confirmed the finding closed.
- JavaScript syntax, Python compilation and `git diff --check` passed. No separate lint/typecheck tool is configured.
- `uv build --wheel` succeeded. The wheel contains the Python code, browser assets and default JSON resources. `scripts/smoke_installed.py` passed with the final wheel in a dependency-free Python 3.12.11 environment, outside the checkout with isolated Python mode: console entry point, safe/refused planning, catalog, demo/verify/replay/compare, output-overwrite/live-opt-in refusal, HTTP assets, verified ZIP export and extracted-capsule CLI verification.
- The standard macOS `venv` attempt produced a broken `/install` Python prefix with this uv-managed interpreter. A fresh `uv venv` environment worked and was used for the successful installed-package check. This was a local interpreter setup issue, not a Scout runtime dependency.
- Building emits an existing setuptools warning about the deprecated license-table form; it does not fail the build. Runtime dependencies remain unchanged. The package was not published.
- GitHub Actions now includes trusted replay/comparison and isolated-wheel smoke checks. **Those unpushed changes have not run on GitHub.**

### Actual data and fresh-process replay

- Catalog URLs, source `/info` and transforms for registered sections #3954 and #4003 were checked against primary source metadata; #3905 remains supported.
- Both additional default requests completed metadata preflight, tiny image collection, local capsule creation and verification in the pinned environment. Each saved uint8 shape is `[101, 1, 101]`, with `10,201` decoded bytes and a `206,809`-byte working estimate. The default sampling is 4 x 20 x 4 micrometres, not isotropic 4 micrometres.
- Capsules are retained in ignored `artifacts/v1-section-3954-check-20260929/` and `artifacts/v1-section-4003-check-20260929/`. The stricter original #3905 capsule in `artifacts/live-adapter-check-20260929/` still verifies.
- The actual CLI replay of the #3954 capsule completed in a fresh process with a new default cache, into `artifacts/v1-replayed-3954-20260929/`. Trusted comparison returned `matched: true`, exact decoded-byte equality and matching geometry/source/request. Decoded-data SHA-256 for both: `487188393f4bd5a77e8484057fd927d846fd9da4f20ccc42ac03e6a539d83e71`.
- The replay reports `fresh_directory: true`, `network_bytes: null` and `independent_download_proven: false`. This is local fresh-cache replay evidence, not second-computer reproduction or independently audited network transfer.

### Browser QA

The browser checks below targeted the local application at `http://127.0.0.1:8765/`, title `BigBrain Scout | Local workspace`, not a deployed service.

| Required check | Result |
| --- | --- |
| Page identity | Correct localhost URL and title |
| Nonblank page | Request, preflight and capsule controls rendered |
| Framework overlay | None |
| Console health | No warning/error logs after final workflows |
| Screenshot evidence | Desktop and narrow-screen screenshots captured and visually inspected outside the repository |
| Interaction proof | Source changes, preflight, real fetch/build, reverify, unsafe refusal and successful offline recovery |

- #3954 and #4003 browser flows showed their correct default Y bounds and section-specific preview labels. Each real build/reverify completed; their capsule IDs are `bbd16bb7cc7d4c109cb1ec8c2b3d1288` and `db3c2f9de8f54ca2818f121bd6d6f5ec` under ignored `artifacts/mvp-workspace/capsules/`.
- Changing source removed the old preview/download state and disabled build until a new check. Synthetic decoded budget `5` was refused because the ROI needs `6` bytes. Restoring the budget, checking, building and reverifying succeeded.
- Desktop screenshot viewport measured 1159 x 1338 CSS pixels; document scroll width 1159. Narrow-screen viewport measured 390 x 844; document scroll width 390. No page-level horizontal overflow; the precision table scrolls internally. Not a physical-phone or cross-browser test.
- The temporary viewport override was reset, measuring 1159 x 1298, and the working local app tab was retained. Screenshots are `scout-v1-desktop-4003.jpg` and `scout-v1-mobile-4003.jpg` in the task's visualization directory, not Git.

### Handoff boundary

Changes remain **uncommitted, unstaged and unpushed** on `pre-hackathon-mvp`. `main`, `origin/main` and HEAD remain at `17ab60284760218f9dfe0ec6363ec4ceacedd12d`. No real capsule, cache or wheel is tracked in Git. No project submission, public issue update, email or deployment was made during implementation.

Source licensing, anatomical correctness and independent live re-download reproduction remain unverified. The adapter is pinned to siibra 1.0.1a22 and three fixed registered sections, not arbitrary datasets. Byte budgets reject decoded ROI/working estimates, not guaranteed process-memory or network-transfer use. The localhost application is not a production or multi-user service. These boundaries do not require leaving the defined v1 code unfinished before the event.

## CPU resolution audit and held model adapter, October 13, 2026

- Python 3.11.13 with the existing scientific environment: **253 tests passed, zero skips**. Python 3.12.11 without optional numerical dependencies: **253 discovered, 144 passed, 109 skipped**. These are local checks; the unpushed GitHub Actions configuration has not run.
- The real native section-3905 capsule contains uint8 shape `[401, 1, 401]`, 160,801 decoded bytes, and sampling `[0.001, 0.020, 0.001]` mm. The exact top-left X/Z crop becomes 400 x 400 pixels, with Z rows and X columns. Crop SHA-256: `4d1125f4687b2bc30df94d5c93587a20df39eddfec21eadc5b8f7861abe7fc2c`.
- A fresh local process with a fresh cache reproduced matching physical geometry and every decoded value in `artifacts/scout-replay-section-3905-1um-20261013/`. Decoded SHA-256: `bb261af1189f86b6de62d2569680537d7ebebde8a1857c1495f90a6f1bd6a7ed`. Independent network transfer and second-machine reproduction are **not established**.
- The actual CPU audit at `artifacts/scout-cpu-audit-20261013/` verifies by recomputing every mask and deterministic metric. A separately produced local audit gives identical source/mask hashes. Fixed settings: Gaussian sigma 1.5 micrometres, dark threshold 90, inclusive component area 16..600 square micrometres, 8-connectivity, exclusion of outer-border components. No parameter was retuned by scale.
- Native 1 micrometre and derived 2/4 micrometre candidate counts are **69 / 73 / 106**. Native-relative count changes are **+5.797% / +53.623%**. These are untrained dark components, not expert-labeled cells or biological counts. Recorded runtimes are single local wall-clock observations, not benchmarks. Exact recomputation requires recorded NumPy 2.4.6 and SciPy 1.17.1.
- Instance evaluation uses maximum-cardinality one-to-one IoU matching and validates bounded label files, hashes, geometry and declared held-out provenance. No real reference labels were created, no real accuracy was computed, and a declared review field does not certify expert review. Unknown model training sections remain unknown.
- The frozen CALHippo/Cellpose adapter is prepared and tested with mocks only. `model-plan` is dependency-free; `infer` refuses before imports or file access without exact explicit GPU opt-in. Checkpoint identity, missing-path download refusal, CUDA/version guards, mask geometry and finite reports are covered. **No checkpoint was downloaded, no model library installed, and no actual inference or training ran.** Hardware compatibility and GPU measurements are held for hackathon day.
- The final wheel builds offline and matches the adapter source byte-for-byte. A clean dependency-free Python 3.12 environment passed isolated CLI/HTTP/export/replay checks and the held-model metadata/refusal checks. Python compilation, JavaScript syntax and whitespace checks passed. The existing setuptools license-table deprecation warning remains non-fatal.
- m-26's exporter has **17 passing regression tests**. Native image/provenance and new CPU-mask/outline/replay exports all pass read-only byte checks against their verified capsules. Public exports omit private local paths and preserve source attribution/license terms.
- m-26's **341 site tests**, lint, typecheck and local webpack production build passed. Browser controls changed resolution/count drift, showed and hid candidate outlines, and displayed matched replay values. Both actual images loaded; mobile viewport 390 x 844 and desktop 1176 x 1338 had no page-level horizontal overflow. No warning/error console entries were present in the checked session. Screenshots were visually inspected against the existing blog style.
- The BaseHub draft readback contains the measured CPU results and explicit GPU hold. The committed-content query returns no article at this slug. No BaseHub commit, Git commit, push, deployment or publication occurred. Scout remains on `pre-hackathon-mvp` with HEAD `17ab60284760218f9dfe0ec6363ec4ceacedd12d`.
