# BigBrain Scout

An early hackathon prototype for **safe, reproducible high-resolution ROI extraction**: inspect a request before fetching data, compare requested and returned metadata, and retain a provenance capsule.

Prepared for the [BigBrain Hackathon 2026](https://tfunck.github.io/bigbrainhack2026/), Montréal, October 18–19. Project lead: [Montek Kundan](https://github.com/Montekkundan). This is an independent proposal, not an official siibra tool or an organizer-endorsed project.

## Try the offline starter

Python 3.11 or 3.12 recommended. These commands need no runtime dependencies or network:

```sh
git clone https://github.com/Montekkundan/bigbrain-scout.git
cd bigbrain-scout
python -m unittest discover -s tests -v
python -m bigbrain_scout plan --request examples/safe-request.json
# Expected refusal: the unsafe request's decoded-byte budget is too small.
python -m bigbrain_scout plan --request examples/unsafe-request.json
python -m bigbrain_scout demo --output artifacts/demo
python -m bigbrain_scout verify --capsule artifacts/demo
python artifacts/demo/reproduce.py --output artifacts/reproduced-demo.raw
```

The offline fixture is **synthetic, nonbiological data**, not a BigBrain image. Its reproduction script rebuilds the ROI from the preserved fixture and checks its hash; it does not independently re-download a dataset. Existing capsule directories and reproduced files are never overwritten.

## Tiny real-section example

The optional adapter targets one small ROI in section #3905, requests an exact 4 µm in-plane scale, and checks ROI/chunk allocation estimates before fetching. A September 29 test fetched a **101 × 1 × 101 uint8 patch (10,201 decoded bytes)** successfully. Section thickness is not 4 µm; the source scale is anisotropic. It is a deliberately narrow, version-pinned example, not a generic fetch service. See [verification details](docs/verification.md).

```sh
uv sync --extra live
uv run python examples/fetch_tiny_roi.py
uv run python examples/fetch_tiny_roi.py --fetch --output artifacts/section-3905
uv run python examples/investigate_siibra.py
```

Without `--fetch`, the first command reads metadata only. The investigation script does not attempt the reported full-section download. See [the live adapter notes](docs/live-adapter.md) for limits and verification status.

## Why this project?

The discussions in siibra [#704](https://github.com/FZJ-INM1-BDA/siibra-python/issues/704), [#705](https://github.com/FZJ-INM1-BDA/siibra-python/issues/705), and [#706](https://github.com/FZJ-INM1-BDA/siibra-python/issues/706) motivate making ROI bounds, available scales, size limits and returned metadata visible. **They are reported failure modes, not bugs independently reproduced here.** September 29 review found maintainer updates saying the scale-key problem was fixed, a resolution warning was remedied, and an ROI issue could not be reproduced on newer versions. Open issue status alone is not proof of a current defect.

Scout's proposed value is the user-facing planning and provenance workflow, even when the underlying client works correctly. The two-day MVP should support one well-defined fetch path before expanding.

## What this starter does and does not promise

- Offline preflight: reject invalid coordinates, unavailable scales and requests beyond a decoded-byte/chunk budget; record clipping explicitly.
- Offline capsule: data, preview, request/returned metadata, environment and SHA-256 verification.
- Optional real adapter: inspect source metadata, manually clip bounds, record actual NIfTI shape/affine/voxel sizes, reject unexpected returned scale.
- Preparation: [dated checklist](docs/preparation.md), [demo plan](docs/demo.md), [two-minute pitch](docs/pitch.md), [submission draft](docs/project-pitch.md), and [illustrative report](docs/report-mockup.html).

Decoded-byte estimates are **not network-transfer estimates or a guaranteed peak-memory ceiling**. siibra caching, concurrency, compression and temporary arrays affect resource use. Bounds/shape checks are not anatomical validation or a scientific quality score. Live reproduction cannot promise unchanged remote bytes. No AI model is required.

## Contribute during the hackathon

Start with the [good-first-issue board](https://github.com/Montekkundan/bigbrain-scout/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22): preview, regression tests, metadata/domain review, and documentation. See [CONTRIBUTING.md](CONTRIBUTING.md). The UI, broader source support, scientific review and independent live reproduction are intentionally left for the team.

## License and credit

Scout code is MIT. This does **not** relicense BigBrain/siibra data. The synthetic fixture is generated for this project; real downloads must retain source terms and citations separately. The initial live adapter does not redistribute image data in Git. Review [BigBrain access information](https://bigbrainproject.org/) and [siibra](https://github.com/FZJ-INM1-BDA/siibra-python) before redistributing derived data.
