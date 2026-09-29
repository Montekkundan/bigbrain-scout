# Verification record — September 29, 2026

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
