# Rendered-output index

Rendered experiment folders use the same diagnostic vocabulary:

- `teacher_forced.mp4` measures one-step prediction with correct source history.
- `free_rollout.mp4` is the actual autoregressive dream after the source cutoff.
- `comparison.mp4` aligns target, teacher prediction, free rollout, and error.
- `error_maps.mp4` uses red for false white pixels and cyan for missed white
  pixels.
- `error_curve.csv` and `error_curve.png` preserve frame-level measurements.

Beginning with renders made after the V4.2 code change, `rollout-ar` also writes
a local `report.md` with headline metrics and memory/cut-gate diagnostics.
Existing key V4 results were backfilled manually.

The completed `hybrid_v4_2_long_horizon` folder also contains
`findings_report.html`, a validated self-contained technical report. Its main
finding is that V4.2 reduces true-polarity rollout error from 10.16% to 4.93%,
while the separate polarity head regresses to 88.47% accuracy and masks that
gain in the raw render.

Do not compare full-video mean error directly with a 15-second mean without also
examining the same 45-60 second slice. The full run contains many more scene
transitions and was trained with a different optimization budget.
