# Contributing

The `pre-hackathon-mvp` branch includes pre-event preparation and hackathon-day teammate checks: browser workflow, three fixed sections, installed CLI, trusted replay, comparison and CPU auditing. Keep its origin transparent. Domain/source-term review, second-computer reproduction and usability feedback remain team work.

1. Run the [teammate quickstart](README.md#test-on-a-second-computer-no-gpu-needed). Return the resulting `check.json`, commit tested and whether you used another computer. Do not attach real tissue or masks to public issues. Check contributor issues against the current branch before choosing work; old issues may describe code already present.
2. Use Python 3.11/3.12 and run `python -m unittest discover -s tests -v`. The dependency-free suite skips optional scientific tests. Install `.[audit]` for numerical/mocked source tests without siibra, or use `uv run --locked --extra live python -m unittest discover -s tests -v`. Tests use mocks, not uncontrolled downloads. See `scripts/smoke_installed.py` for clean-directory wheel checks.
3. Keep tests offline by default. Gate real downloads explicitly and cap the ROI/chunk estimates.
4. Label synthetic data, reported bugs and verified results distinctly.
5. Never commit data caches, authentication material, registration emails or private correspondence. Keep downloaded artifacts in ignored `artifacts/`.
6. Preserve actual returned shape, affine, units and source terms; do not infer anatomical validity from a successful download.

Follow the event's [code of conduct](https://github.com/tfunck/bigbrainhack2026/blob/main/CODE_OF_CONDUCT.md). Give contributors and upstream tools credit. Issue reports should state library version, source, request, observed result and reproduction limits.
