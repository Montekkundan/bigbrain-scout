# Five-minute demonstration

## Offline fallback: repeatable without internet

```sh
python -m unittest discover -s tests -v
python -m bigbrain_scout plan --request examples/safe-request.json
python -m bigbrain_scout demo --output artifacts/demo
python -m bigbrain_scout verify --capsule artifacts/demo
python artifacts/demo/reproduce.py --output artifacts/reproduced-demo.raw
```

Use fresh capsule and reproduced-file names for each run. The tiny fixture is synthetic, not brain tissue. Open the preview and manifest; show the distinction between requested parameters and returned metadata. The last command rebuilds the ROI from the preserved fixture and checks its hash, not an independent remote fetch.

## Proposed final live story

1. **0:00–0:45:** State the source, space and requested scale; label the tool a prototype.
2. **0:45–1:30:** Show a deliberately oversized *planned* request being blocked before image download. Never deliberately trigger an uncontrolled full-section fetch.
3. **1:30–2:30:** Refine to the tiny section #3905 patch; show the available anisotropic scale and decoded/chunk estimates.
4. **2:30–3:30:** Fetch or use an explicitly cached patch; inspect actual shape, affine and voxel sizes. Explain any fallback or failure honestly.
5. **3:30–4:30:** Open capsule metadata, source terms/citations and checksum report.
6. **4:30–5:00:** State what was verified, what is not scientifically validated, and next contributor/upstream work.

The real adapter currently exports a separate live manifest; unifying it with the offline schema and independent re-fetch comparison are contributor work. Do not call a checksum-only check independent reproduction.
