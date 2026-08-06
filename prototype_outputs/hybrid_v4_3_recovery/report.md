# Hybrid V4.3 recovery — final full-video report

## Outcome

V4.3 epoch 1 is the final selected checkpoint. The full 6,573-frame render completed,
all four diagnostic MP4s tail-decode successfully, and the original soundtrack was
muxed into the two viewing deliverables.

## Headline comparison against V4.2

| Metric | V4.2 | V4.3 | Change |
| --- | ---: | ---: | ---: |
| Post-cutoff rollout binary error | 0.049289 | 0.044500 | 9.72% lower |
| Post-cutoff accumulation gap | 0.017169 | 0.012760 | 25.68% lower |
| Post-cutoff rollout IoU | 0.843972 | 0.854195 | +0.010223 |
| Peak rollout error | 0.385223 | 0.346054 | 10.17% lower |

V4.3 has lower rollout error on 65.77% of frames and every
evaluated non-overlapping five-second window.

## Silhouette / local-motion finding

At 53–55 s, rollout error improves from 0.069229 to
0.064644 (6.62% lower),
but the V4.3 teacher error remains only 0.022039. The large
teacher-to-rollout split confirms that small hand/wing motion is still lost mainly
through autonomous state drift, not because the autoencoder cannot represent it.

## Training decision

- Baseline rollout latent MSE: 0.337478
- Epoch 1: 0.326779 — selected
- Epoch 2: 0.328261 — regressed

Do not continue the same recovery schedule. Keep `model_best.pt` (epoch 1).

## Deliverables

- `comparison_with_audio.mp4` — target / teacher / rollout / error with soundtrack
- `free_rollout_with_audio.mp4` — standalone autoregressive dream with soundtrack
- `comparison.mp4` — silent diagnostic comparison
- `teacher_forced.mp4` — silent teacher-forced diagnostic
- `free_rollout.mp4` — silent free rollout
- `error_maps.mp4` — silent false-positive / false-negative maps
- `error_curve.csv` — complete frame-level metrics
- `drift_summary.json` — aggregate render metrics and settings
- `analysis.json` — V4.2/V4.3 derived comparison data
- `artifact.json` — portable report source
- `findings_report.html` — self-contained interactive technical report

## Recommended next step

Ship this checkpoint for the prototype and review the two audio videos artistically.
If one more technical experiment is approved, isolate it to a high-resolution
temporal residual/boundary head for small moving parts. Preserve free-running error
bleed; do not hard-reset the rollout to target latents.
