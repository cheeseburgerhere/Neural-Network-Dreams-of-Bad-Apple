# V4.2 rollout silhouette diagnostics

**Status:** Greedy diagnosis complete

## Main finding

Teacher-good/rollout-bad behavior comes from a training-target mismatch. During free-running burn-in and rollout, the model state is its own imperfect prediction, but velocity supervision remains `target - true_previous`. Correct recovery requires `target - predicted_previous`.

The latent loss asks the model to remove accumulated state error. The velocity losses simultaneously ask it to reproduce only true scene motion. Teacher forcing satisfies both because predicted and true previous states coincide. Free rollout does not.

Inference-only gain and bleed changes were rejected:

- Fast scaling amplifies directionally wrong velocity.
- Stronger memory bleed pulls toward interpolated anchors and blurs local contours.
- Memory-only output confirms anchors are recovery references, not finished frames.

## Greedy ablation

| Variant | Sample error | Sample boundary F1 | 53-55s error | 53-55s boundary F1 | Mean gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0516 | 0.4430 | 0.0693 | 0.6383 | 0.2289 |
| fast-1.5 | 0.0578 | 0.4237 | 0.0738 | 0.6404 | 0.2310 |
| fast-2.0 | 0.0685 | 0.3889 | 0.0786 | 0.6317 | 0.2333 |
| recovery-0.25 | 0.0769 | 0.3673 | 0.0786 | 0.6193 | 0.2726 |
| moving-0.5 | 0.0837 | 0.3525 | 0.0828 | 0.6093 | 0.2828 |
| memory-only | 0.1345 | 0.3030 | 0.0999 | 0.5820 | 1.0000 |

## 53-55 second target mismatch

- True scene-velocity RMS: 0.4054.
- Required state-relative recovery RMS: 0.7503 (2.04x larger).
- Previous rollout-state MSE: 0.5755.
- Predicted velocity MSE versus training target: 0.1835.
- Predicted velocity MSE versus required recovery: 0.5870.

## Causal oracle test

For diagnosis only, subtracting a known fraction of previous state error from each next prediction isolates the missing recovery behavior.

| Correction fraction | Pixel error | Boundary F1 | Latent MSE |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.0693 | 0.6384 | 0.5776 |
| 0.10 | 0.0518 | 0.7340 | 0.4089 |
| 0.25 | 0.0388 | 0.8183 | 0.3075 |
| 0.50 | 0.0262 | 0.8996 | 0.2250 |
| 1.00 | 0.0187 | 0.9462 | 0.1845 |

This oracle uses true previous state and is not deployable. Its monotonic improvement establishes causality and defines the next training change.

## Recommended implementation

Fine-tune motion path with state-relative velocity supervision:

`recovery_velocity = target - latent_history[:, -1]`

Keep original true-scene velocity as a smaller auxiliary term so local motion character remains. Freeze polarity spline and scene memory. Train on self-generated burn-ins, then validate teacher precision and full rollout together.

## Definitions

- Sample metrics: every configured stride plus every 53-55s frame.
- Boundary F1: silhouette edges matched within two pixels.
- Memory-only: two nearest learned time anchors, no autoregression.
- Moving bleed: restores anchor correction where motion mask normally suppresses it.
- Recovery: adds anchor correction when latent disagreement grows.
