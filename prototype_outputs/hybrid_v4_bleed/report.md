# Hybrid V4 bleed render log

**Status:** Complete

**Checkpoint:** `prototype_runs/hybrid_v4_bleed/model_best.pt`

## Headline metrics

| Metric | Value |
| --- | ---: |
| Post-cutoff teacher binary error | 3.62% |
| Post-cutoff free-rollout binary error | 7.29% |
| Accumulation gap | 3.66% |
| Mean binary IoU | 0.805 |
| Peak free-rollout binary error | 26.76% |
| Final-frame binary error | 14.98% |

## Interpretation

The character remains recognizable and scene errors recover gradually. In the
hand-and-wing interval, frames 240-300, predicted frame-to-frame pixel motion
was 0.70% versus the target's 1.78%. The moving-pixel union was 25.5% versus
35.2% in the target. V4 therefore moved rather than froze, but visibly damped
fine motion; that observation motivated V4.1.
