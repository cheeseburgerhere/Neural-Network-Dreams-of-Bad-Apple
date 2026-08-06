# Hybrid V4.2 long-horizon findings

**Status:** Training, full-video evaluation, and comparison video complete

**Checkpoint:** `prototype_runs/hybrid_v4_2_long_horizon/model_best.pt`

## Main conclusion

V4.2 substantially reduces long-horizon collapse in canonical scene content,
but its learned polarity head fails. The raw binary render therefore looks
worse than V4.1 even though the underlying autoregressive trajectory is much
better.

With the known target polarity applied, V4.2 reduces full-video free-rollout
binary error from **10.16% to 4.93%** and reduces the teacher-to-rollout
accumulation gap from **7.42% to 1.72%**. Normalized rollout latent MSE falls
from **0.464 to 0.337**.

The learned polarity head reaches only **88.47%** accuracy and predicts **23
switches** for three real switches. This raises actual raw rollout error to
**15.26%**.

## Headline comparison

| Post-cutoff metric | V4.1 full | V4.2 raw | V4.2 with true polarity |
| --- | ---: | ---: | ---: |
| Teacher binary error | 2.73% | 14.04% | 3.21% |
| Free-rollout binary error | 10.16% | 15.26% | **4.93%** |
| Accumulation gap | 7.42% | 1.23% | **1.72%** |
| Rollout latent MSE | 0.464 | 0.337 | 0.337 |
| Polarity accuracy | 100% | 88.47% | 100% by construction |

The true-polarity result is an exact binary counterfactual, not an estimate.
When polarity is wrong, inverting every prediction bit changes binary error
from `e` to `1-e`.

## Long-context recovery by interval

| Interval | V4.1 rollout | V4.2 true polarity | Interpretation |
| --- | ---: | ---: | --- |
| 0-5 s | **5.21%** | 6.17% | Early-timeline regression |
| 5-15 s | 11.67% | **3.89%** | Large recovery gain |
| 45-60 s | 10.89% | **7.12%** | Better than V4.1 full, below dedicated short model |
| 53-55 s hands/wings | 9.43% | **6.92%** | Local content is present; raw output is inverted |
| 105-120 s | 16.14% | **6.18%** | Previous collapse region largely repaired |
| 210 s-end | 22.04% | **6.88%** | Late-horizon drift strongly reduced |

The corrected peak is 38.52% at frame 358, or 11.93 seconds. This and the
0-5 second regression indicate that random long burn-in under-samples direct
supervision near the beginning of the timeline.

## Why the raw 45-60 second output looks catastrophic

The longest wrong-polarity interval runs from frame 1,263 through frame 1,685,
or 42.1-56.17 seconds. It covers 74.67% of the original 45-60 second prototype
window and all of the 53-55 second hand-and-wing interval.

Consequently:

- 45-60 second raw rollout error is 71.00%; correct polarity makes it 7.12%.
- 53-55 second raw rollout error is 93.08%; correct polarity makes it 6.92%.

This is a representation-head failure rather than loss of the silhouette or
local movement.

## Training behavior

Training completed all 12 epochs in 27,087.8 seconds, approximately 7.52
hours, with an empty stderr log. Validation rollout latent MSE improved every
epoch from 0.604 to 0.337 while the rollout curriculum expanded from four to
32 frames. There is no evidence of a GPU-memory failure or optimizer collapse.

The rollout effective memory gate averages 22.89%, compared with 9.10% for
V4.1 full. This improves rollout stability, but teacher latent MSE rises by
27.8%. V4.2 therefore trades some one-step precision for much better
free-running recovery.

Cut-gate activity averages 14.53% during rollout and is not confined to a few
isolated cuts. Its positive correlation with error shows that it activates on
difficult frames; it does not prove that the gate causes those errors.

## Recommended next experiment

Do not retrain the full motion model yet.

1. Give polarity a separate low-frequency normalized-time encoder.
2. Fine-tune only that small polarity branch from the saved V4.2 checkpoint.
3. Require exactly three predicted switches and at least 99.5% frame accuracy.
4. Re-render and check whether raw rollout error approaches the current 4.93%
   true-polarity result.
5. Add deliberate zero/short-burn batches or left-padded early windows before
   any later full retraining.
6. After polarity is fixed, test a lower transition cap at inference to see
   whether teacher precision can be recovered without reopening collapse.

## Artifacts

- `comparison.mp4`: full 3:39 target, teacher, rollout, and error comparison.
- `findings_report.html`: self-contained technical report with charts and
  tables.
- `analysis.json`: reviewed calculations and metric definitions.
- `analysis.sqlite`: bounded 37-row report snapshot.
- `artifact.json`: validated report source.
- `drift_summary.json`: raw renderer summary.
- `error_curve.csv`: all 6,573 frame-level measurements.
- `error_curve.png`: raw binary-error curve, dominated by polarity failures.
- `teacher_forced`, `free_rollout`, `error_maps`, and `comparison`: metrics-run
  PNG frames. Only the combined comparison was encoded as MP4.

The HTML report passed artifact validation and structural packaging. Browser
QA was unavailable because no compatible Chromium headless shell is installed;
the report includes the generated semantic fallback.
