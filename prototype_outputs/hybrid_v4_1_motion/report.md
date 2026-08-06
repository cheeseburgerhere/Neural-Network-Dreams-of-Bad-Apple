# Hybrid V4.1 dual-motion render log

**Status:** Complete

**Checkpoint:** `prototype_runs/hybrid_v4_1_motion/model_best.pt`

## Headline metrics

| Metric | Value |
| --- | ---: |
| Frames | 450 |
| FPS | 30 |
| Post-cutoff teacher binary error | 3.4258% |
| Post-cutoff free-rollout binary error | 5.9226% |
| Accumulation gap | 2.4968% |
| Post-cutoff mean binary IoU | 0.84114 |
| Peak free-rollout binary error | 25.7304% at frame 165 |
| Final-frame binary error | 12.3103% |

## Internal behavior

The post-cutoff effective memory gate averaged 13.22%. Slow latent velocity
averaged 0.05278 and fast velocity averaged 0.02127, confirming that the added
branch was active. Polarity matched the single target inversion.

V4.1 is the strongest 15-second result so far. It should remain the control
when evaluating V4.2: compare both the full-video aggregate and the identical
45-60 second slice.
