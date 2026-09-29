# Live adapter: narrow and explicitly gated

`examples/fetch_tiny_roi.py` isolates a pinned internal siibra API. Target: section #3905, the maintainer-discussed tiny ROI `[7.6, 8, 7.6]` to `[8, 10, 8]` mm, manually intersected with the section's metadata bounds. Requested provider voxel sizes: `[0.004, 0.020, 0.004]` mm; section thickness is not isotropic 4 µm.

## Resource and fallback boundaries

- Set `SIIBRA_MAX_FETCH_SIZE_BYTES` before importing siibra; pass explicit `max_bytes` too.
- Read `/info` and `/transform.json`, require an exact available scale, and never silently choose a coarser scale in the adapter.
- Inspect `get_boundingbox(clip=False)` and manually intersect bounds; do not call `Volume.intersection()` as a planning operation because its implementation can fetch image data.
- Estimate decoded ROI bytes and rounded chunk mosaic plus one chunk; both must fit an 8 MiB example budget.
- Metadata preflight alone does not guarantee process peak RAM, network transfer volume, source stability or scientific correctness.
- Record returned array-axis shape, affine and voxel sizes directly. Do not label the returned array xyz merely because the provider uses xyz.
- Keep real image outputs in ignored `artifacts/`. Do not publish source data until license/citation terms are reviewed.

## Verification status

September 29: source and live metadata inspected. Historical issues are not independently reproduced defects. Actual example runtime results are recorded separately in `docs/verification.md` after testing.

The live example's manifest is currently `live-example-0.1`, distinct from the offline synthetic schema. Unifying both, automating independent re-fetch comparison, handling arbitrary affines/sections and adding stronger process resource limits are deliberately out of the starter's scope.

Sources: [siibra 1.0.1a22](https://pypi.org/project/siibra/1.0.1a22/), [Neuroglancer implementation](https://github.com/FZJ-INM1-BDA/siibra-python/blob/main/siibra/volumes/providers/neuroglancer.py), [Volume implementation](https://github.com/FZJ-INM1-BDA/siibra-python/blob/main/siibra/volumes/volume.py), [section configuration](https://github.com/FZJ-INM1-BDA/siibra-configurations/blob/master/features/images/sections/cellbody/73c1fa55-d099-4854-8cda-c9a403c6080a_bigbrain_1um_3905.json), [source info](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3905/precomputed/info), [source transform](https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3905/precomputed/transform.json).
