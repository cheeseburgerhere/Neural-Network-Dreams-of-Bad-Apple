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
| baseline | 0.0480 | 0.4598 | 0.0648 | 0.6595 | 0.2180 |

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
