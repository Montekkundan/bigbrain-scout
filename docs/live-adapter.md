# Live adapter: narrow and explicitly gated

The v1 adapter in `bigbrain_scout/live.py` isolates a pinned internal siibra API. Its offline catalog in `bigbrain_scout/sources.py` permits only registered sections #3905, #3954 and #4003. The earlier `examples/fetch_tiny_roi.py` remains a separate narrow investigation helper, not the current capsule workflow. Requested default provider voxel sizes: `[0.004, 0.020, 0.004]` mm; section thickness is not isotropic 4 µm.

| Preset | Default lower bound (mm) | Default upper bound (mm) |
| --- | --- | --- |
| section-3905 | 7.6, 8.0, 7.6 | 8.0, 10.0, 8.0 |
| section-3954 | 7.6, 9.04, 7.6 | 8.0, 9.08, 8.0 |
| section-4003 | 7.6, 10.02, 7.6 | 8.0, 10.06, 8.0 |

Each request is manually intersected with the selected section's metadata bounds. Exact scale availability is verified at preflight, not inferred from the preset label.

## Resource and fallback boundaries

- Set `SIIBRA_MAX_FETCH_SIZE_BYTES` before importing siibra; pass explicit `max_bytes` too.
- Read `/info` and `/transform.json`, require an exact available scale, and never silently choose a coarser scale in the adapter.
- Inspect `get_boundingbox(clip=False)` and manually intersect bounds; do not call `Volume.intersection()` as a planning operation because its implementation can fetch image data.
- Estimate decoded ROI bytes and a working model of decoded ROI plus rounded chunk mosaic plus one chunk; each configured budget is at most 8 MiB.
- Metadata preflight alone does not guarantee process peak RAM, network transfer volume, source stability or scientific correctness.
- Record returned array-axis shape, affine and voxel sizes directly. Do not label the returned array xyz merely because the provider uses xyz.
- Keep real image outputs in ignored `artifacts/`. Do not publish source data until license/citation terms are reviewed.

## Verification status

September 29: source and live metadata inspected. Historical issues are not independently reproduced defects. Actual example runtime results are recorded separately in `docs/verification.md` after testing.

The earlier investigation helper's manifest is `live-example-0.1`. The v1 workspace/CLI uses the stricter `live-1.0` contract alongside synthetic `1.0`; trusted verification/replay intentionally refuses the older investigation contract. Current live capsules preserve fixed source identity, scale/transform snapshots and hashes, array-axis geometry, decoded bytes and exact previews. All three default patches have been fetched and verified locally; see [verification](verification.md).

```sh
uv run python -m bigbrain_scout plan --source section-4003 --allow-live
uv run python -m bigbrain_scout fetch --source section-4003 --output artifacts/new-4003 --allow-live
uv run python -m bigbrain_scout reproduce --capsule artifacts/new-4003 --output artifacts/replayed-4003 --allow-live
uv run python -m bigbrain_scout compare --first artifacts/new-4003 --second artifacts/replayed-4003
```

Trusted replay never executes capsule code. It requires a fresh process for live access and defaults to a fresh local cache; explicit cache reuse is reported. Comparison requires exact decoded uint8 values and matching source/request/geometry, while reporting metadata/environment/container differences independently. A mismatch exits `3` and preserves the output. Fresh-cache matching does not establish second-computer reproduction or independently measured network traffic. Arbitrary sources/affines, source-term approval, anatomical validation and hard process-memory limits remain outside v1.

Sources: [siibra 1.0.1a22](https://pypi.org/project/siibra/1.0.1a22/), [Neuroglancer implementation](https://github.com/FZJ-INM1-BDA/siibra-python/blob/main/siibra/volumes/providers/neuroglancer.py), [Volume implementation](https://github.com/FZJ-INM1-BDA/siibra-python/blob/main/siibra/volumes/volume.py), [section configuration](https://github.com/FZJ-INM1-BDA/siibra-configurations/blob/master/features/images/sections/cellbody/73c1fa55-d099-4854-8cda-c9a403c6080a_bigbrain_1um_3905.json), [source info](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3905/precomputed/info), [source transform](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3905/precomputed/transform.json).

Additional official source metadata checked for the catalog: [section #3954 info](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3954/precomputed/info), [section #3954 transform](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3954/precomputed/transform.json), [section #4003 info](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_4003/precomputed/info), [section #4003 transform](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_4003/precomputed/transform.json). These links identify data sources, not a legal conclusion about redistribution rights.
