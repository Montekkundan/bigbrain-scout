# BigBrain Scout

An early hackathon prototype for **reproducible brain-tissue resolution audits**: extract a bounded region, run the same CPU method at 1, 2 and 4 micrometres, and preserve the data, geometry, settings and outputs needed to repeat the comparison. These are dark-component candidates, not validated cell counts.

Prepared for the [BigBrain Hackathon 2026](https://tfunck.github.io/bigbrainhack2026/), Montréal, October 18–19. Project lead: [Montek Kundan](https://github.com/Montekkundan). This is an independent proposal, not an official siibra tool or an organizer-endorsed project.

## Test on a second computer, no GPU needed

Use Python 3.11 or 3.12. Clone this branch, not `main`:

```sh
git clone --branch pre-hackathon-mvp https://github.com/Montekkundan/bigbrain-scout.git
cd bigbrain-scout
python3 -m bigbrain_scout check --output artifacts/teammate-offline
```

On Windows, use your Python 3.11/3.12 command in place of `python3`. This first check needs no runtime packages, downloads no brain images and does not start a server. Expected: `passed: true`, `synthetic.matched: true`, and `real_audit.executed: false`. It extracts the bundled nonbiological fixture, refuses an unsafe request, then replays and compares the actual saved data. Results are written to `artifacts/teammate-offline/check.json`.

For the **real CPU experiment**, install the existing optional live dependencies in a virtual environment. No Torch, model weights or GPU packages are required:

```sh
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e '.[live]'
python -m bigbrain_scout check --output artifacts/teammate-real --allow-live
```

If you already use uv, the equivalent locked command is:

```sh
uv run --locked --python 3.11 --extra live python -m bigbrain_scout check --output artifacts/teammate-real --allow-live
```

The live check explicitly fetches the fixed native 1 micrometre section-3905 preset, verifies it, and recomputes the 1/2/4 micrometre CPU audit. It compares crop geometry, source hashes, pinned scientific versions, candidate counts **69 / 73 / 106**, and all three binary-mask hashes against the packaged aggregate regression record. That record contains no tissue image or mask data. It is a software regression reference, not biological ground truth, and applies only to this exact crop and settings. A mismatch returns exit code `3` and keeps the evidence; do not change the expected counts merely to make it pass.

Send the collaborator **only `artifacts/teammate-real/check.json`**, the commit tested, and a note that you ran it on your own computer. This JSON contains software/platform details and aggregate signatures, not raw tissue or private local paths. A successful run on Montek's machine is not a second-computer result. The command itself cannot certify who ran it or independently audited network traffic. Keep native images, masks and caches local until redistribution terms are reviewed. Network/source failures return exit code `2`; retain the error and use the offline check while investigating. Existing output directories are never overwritten, so choose a new output name for each retry.

## Local extraction workspace

The `pre-hackathon-mvp` branch includes preparation built before the event, plus hackathon-day reproduction checks. `main` remains the starter baseline. From the checkout:

```sh
git switch pre-hackathon-mvp
python -m bigbrain_scout serve
```

Open [the local workspace](http://127.0.0.1:8765). Choose **Synthetic demo**, click **Check request**, then **Build capsule**. Inspect the actual preview and returned bounds, click **Verify capsule**, and **Download ZIP**. Change any input to invalidate the checked request. For a deliberate refusal, set the decoded budget to `5` bytes and check again.

The offline workspace has no runtime dependencies and uses no remote data. To explicitly enable the optional live adapter:

```sh
uv sync --extra live
uv run python -m bigbrain_scout serve --allow-live
```

Choose **BigBrain section #3905**, **#3954** or **#4003**, check its metadata, then click **Fetch & build capsule**. Each preset has its own default bounds. First-run metadata initialization can take a moment. No image fetch occurs during preflight. Live scale labels describe in-plane sampling; through-section sampling remains 20 µm. Results are kept under `artifacts/workspace/capsules/`, not uploaded. Exact source terms remain unverified; do not publicly share live image capsules yet.

See [workspace usage and limitations](docs/workspace.md) and [verification evidence](docs/verification.md).

### Install without a checkout

The wheel includes the synthetic fixture, default requests and browser assets. Build/install locally, or install this branch explicitly when it is published:

```sh
uv build --wheel
python -m pip install dist/bigbrain_scout-0.1.0-py3-none-any.whl
bigbrain-scout serve
```

The console command and `python -m bigbrain_scout` have the same interface. Install the existing `live` extra when real data access is needed. No public package release or deployed service is claimed.

## Try the offline CLI

Python 3.11 or 3.12 recommended. These commands need no runtime dependencies or network:

```sh
python -m unittest discover -s tests -v
python -m bigbrain_scout plan --example safe
# Expected refusal: the unsafe request's decoded-byte budget is too small.
python -m bigbrain_scout plan --example unsafe
python -m bigbrain_scout demo --output artifacts/demo
python -m bigbrain_scout verify --capsule artifacts/demo
python -m bigbrain_scout reproduce --capsule artifacts/demo --output artifacts/replayed-demo
python -m bigbrain_scout compare --first artifacts/demo --second artifacts/replayed-demo
```

Run the tests from a checkout; the remaining CLI commands also work from an installed wheel in a clean directory. The offline fixture is **synthetic, nonbiological data**, not a BigBrain image. Trusted replay verifies the capsule, checks its source metadata against Scout's packaged fixture, and recreates the ROI from preserved fixture values, never executing capsule-provided Python or Markdown. Existing output directories are never overwritten.

## Bounded real sections and replay

The optional adapter targets small ROIs in three registered sections, requests an exact published scale, and checks ROI/chunk allocation estimates before fetching. September 29 tests fetched a **101 × 1 × 101 uint8 patch (10,201 decoded bytes)** from each preset. Section thickness is not 4 µm; the default sampling is 4 × 20 × 4 µm. This is a fixed, version-pinned catalog, not a generic fetch service. See [verification details](docs/verification.md).

```sh
uv sync --extra live
uv run python -m bigbrain_scout sources
uv run python -m bigbrain_scout plan --source section-3954 --allow-live
uv run python -m bigbrain_scout fetch --source section-3954 --output artifacts/section-3954 --allow-live
uv run python -m bigbrain_scout reproduce --capsule artifacts/section-3954 --output artifacts/replayed-3954 --allow-live
uv run python -m bigbrain_scout compare --first artifacts/section-3954 --second artifacts/replayed-3954
```

`sources` is offline catalog inspection. `plan` may retrieve metadata but does not collect image chunks. Live `fetch` and `reproduce` require explicit opt-in. Replay uses a fresh process and, by default, a new cache directory. Exact decoded values and geometry are compared separately from environment, source snapshot and file/header hashes. A mismatch returns exit code `3` and retains the new output for inspection; refusal returns `2`. Matching values are not proof of a second-computer or independently audited network re-download. See [the live adapter notes](docs/live-adapter.md).

## CPU candidate-resolution audit

The `audit` command works offline on a verified, native 1 µm section-3905 capsule. Install `.[audit]` for the pinned NumPy/SciPy/nibabel environment without siibra; `.[live]` includes the same versions plus the fetch adapter. It compares the same physical crop at native 1 µm and derived 2/4 µm block averages. Gaussian smoothing is expressed in micrometres, component area in square micrometres, and one fixed intensity threshold is used at every resolution. Components touching the crop edge are excluded. These are classical dark-structure **candidates**, not independently labeled cells or neurons.

```sh
# The real teammate check above creates artifacts/teammate-real/native.
python -m bigbrain_scout audit --capsule artifacts/teammate-real/native --output artifacts/candidate-audit
python -m bigbrain_scout verify-audit --audit artifacts/candidate-audit
# Or check an existing native capsule and the regression reference entirely offline:
python -m bigbrain_scout check --capsule artifacts/teammate-real/native --output artifacts/teammate-local
```

The report records settings, source geometry and hashes, candidate masks, counts, area, count drift and measured CPU runtime. `verify-audit` recomputes masks and deterministic metrics offline from the bounded saved source. Runtime is not expected to reproduce exactly. Outputs are never overwritten. No model weights, image requests or GPU jobs are triggered by either command.

### Independent instance labels

```sh
python -m bigbrain_scout evaluate --prediction prediction.json --reference reference.json --iou-threshold 0.5
```

Both strict JSON files declare schema version 1, section ID, coordinate space, row/column spacing and origin in micrometres, signed axis directions, rectangular integer instance labels (0 is background), a SHA-256 of compact JSON label rows, and role-specific provenance. Prediction provenance declares `type: predicted`, `independent_expert_reviewed: false` and `training_section_ids`. Reference provenance declares `type: independent_expert_reviewed`, `independent_expert_reviewed: true` and `evaluation_split: held_out_section`. Geometry must match; a declared training section cannot be used as its held-out reference. These declarations are caller assertions, not authentication of expert review or a complete training registry.

Matching maximizes the number of one-to-one pairs at or above the declared IoU threshold. Precision, recall, F1 and absolute count error are reported only for the supplied labels. The real demo crop has no independent cell-instance reference annotations, so its segmentation accuracy remains unmeasured. Another model's output or cortical-layer annotations must not be presented as independent cell ground truth.

### Model execution is on hold

`model-plan` is an offline inspection command. `infer` additionally requires an explicitly supplied, hash-checked official checkpoint, the pinned model runtime, a CUDA device and `--allow-gpu`. It never starts automatically from the website or CPU audit. Model installation, checkpoint acquisition, GPU inference and hardware compatibility validation are held for hackathon day; no GPU performance or learned-model result is claimed now. Training from scratch is not part of this prepared workflow.

The checkpoint is CALHippo's `finetune_v4_astrocytes_big_brain`, not a general neuron detector. The adapter pins [the official weights](https://huggingface.co/AImageLab-Zip/CALHippo-Framework-Models/blob/46d13ceec0233e5c32b7d7cec0a0bd65edd5bcd2/segmentation/cellpose/finetune_v4_astrocytes_big_brain) at revision `46d13ceec0233e5c32b7d7cec0a0bd65edd5bcd2`: 609,391,431 bytes, SHA-256 `98f1b62df3db68ad21a54725caacc3b718f7d46c1c1a40eafef9932674a3e534`, CC BY-NC-SA 4.0. It requires Cellpose 4.1.1 and records a Scout-specific frozen 30 µm physical-diameter protocol, not reproduction of the full published CALHippo benchmark. Upstream supports CPU execution; CUDA-only is this adapter's deliberate hackathon execution policy. Training-section provenance is incomplete, so these weights cannot establish an independent held-out accuracy claim. No dependency is installed and no checkpoint is downloaded automatically.

```sh
python -m bigbrain_scout model-plan
# Run only on hackathon day, after model/runtime setup and resource allocation:
python -m bigbrain_scout infer --audit artifacts/candidate-audit --checkpoint /absolute/path/to/official-checkpoint --output artifacts/model-audit --allow-gpu
```

The existing m-26 blog is a local presentation layer for verified saved crop, audit and replay records. Its controls do not run Python. The BaseHub article remains a draft; publishing this code does not publish that article.

## Why this project?

The discussions in siibra [#704](https://github.com/FZJ-INM1-BDA/siibra-python/issues/704), [#705](https://github.com/FZJ-INM1-BDA/siibra-python/issues/705), and [#706](https://github.com/FZJ-INM1-BDA/siibra-python/issues/706) motivate making ROI bounds, available scales, size limits and returned metadata visible. **They are reported failure modes, not bugs independently reproduced here.** September 29 review found maintainer updates saying the scale-key problem was fixed, a resolution warning was remedied, and an ROI issue could not be reproduced on newer versions. Open issue status alone is not proof of a current defect.

Scout's proposed value is the user-facing planning and provenance workflow, even when the underlying client works correctly. This v1 supports a small, explicit source catalog.

## What v1 does and does not promise

- Offline preflight: reject invalid coordinates, unavailable scales and requests beyond a decoded-byte/chunk budget; record clipping explicitly.
- Offline capsule: data, preview, request/returned metadata, environment and SHA-256 verification.
- Local browser workflow: fixed source presets, immutable one-use preflight tokens, actual-byte preview, requested/returned geometry, verification and recorded-files-only ZIP export.
- Optional real adapter: inspect source metadata, manually clip bounds, record actual NIfTI shape/affine/voxel sizes, reject unexpected returned shape, affine, bounds or scale.
- Trusted replay and comparison: verify inputs, reconstruct a bounded request without executing capsule code, compare exact decoded uint8 values and geometry, and report metadata/environment differences independently.
- CPU audit: fixed-physical-unit candidate masks and count drift at 1/2/4 µm, with offline recomputation and no cell-accuracy claim.
- Independent-label evaluator: strict geometry/provenance inputs and maximum-cardinality instance matching; it does not create or certify reference annotations.

Decoded-byte estimates are **not network-transfer estimates or a guaranteed peak-memory ceiling**. siibra caching, concurrency, compression and temporary arrays affect resource use. Bounds/shape checks are not anatomical validation or a scientific quality score. Live reproduction cannot promise unchanged remote bytes. No AI model is required.

## Contribute during the hackathon

Start with the teammate checks above and check the [good-first-issue board](https://github.com/Montekkundan/bigbrain-scout/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) against the current branch before taking an issue. See [CONTRIBUTING.md](CONTRIBUTING.md). Scientific/source-term review, usability feedback and independent reproduction on a second computer remain meaningful team work. GPU inference is optional, not a prerequisite for completing the bounded CPU workflow. A scientifically useful region/measurement and independent reference annotations still require domain input; this prototype does not establish segmentation accuracy.

## License and credit

Scout code is MIT. This does **not** relicense BigBrain/siibra data. The synthetic fixture is generated for this project; real downloads must retain source terms and citations separately. The initial live adapter does not redistribute image data in Git. Review [BigBrain access information](https://bigbrainproject.org/) and [siibra](https://github.com/FZJ-INM1-BDA/siibra-python) before redistributing derived data.
