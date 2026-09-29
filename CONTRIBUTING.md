# Contributing

This is a small, pre-hackathon foundation. Please leave the broader demo and implementation open to the team rather than solving every task beforehand.

1. Choose one contributor issue and comment on the intended scope.
2. Use Python 3.11/3.12 and run `python -m unittest discover -s tests -v`.
3. Keep tests offline by default. Gate real downloads explicitly and cap the ROI/chunk estimates.
4. Label synthetic data, mock-ups, reported bugs and verified results distinctly.
5. Never commit data caches, authentication material, registration emails or private correspondence. Keep downloaded artifacts in ignored `artifacts/`.
6. Preserve actual returned shape, affine, units and source terms; do not infer anatomical validity from a successful download.

Follow the event's [code of conduct](https://github.com/tfunck/bigbrainhack2026/blob/main/CODE_OF_CONDUCT.md). Give contributors and upstream tools credit. Issue reports should state library version, source, request, observed result and reproduction limits.
