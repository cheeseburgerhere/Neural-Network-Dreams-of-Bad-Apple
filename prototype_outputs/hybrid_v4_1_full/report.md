# Hybrid V4.1 full-timeline render log

**Status:** Complete

**Checkpoint:** `prototype_runs/hybrid_v4_1_full/model_best.pt`

## Headline metrics

| Metric | Value |
| --- | ---: |
| Frames | 6,573 |
| Duration | 219.1 seconds |
| FPS | 30 |
| Post-cutoff teacher binary error | 2.7323% |
| Post-cutoff free-rollout binary error | 10.1554% |
| Accumulation gap | 7.4231% |
| Post-cutoff mean binary IoU | 0.73269 |
| Peak free-rollout binary error | 82.0221% |
| Peak location | Frame 3,325 / 110.83 seconds |

The final-frame metric is zero because the source video ends in an empty frame;
it is not evidence of stable recovery.

## Internal behavior

There are 220 memory tokens and 219 dominant-token changes. The post-cutoff
effective memory gate averaged 9.10% during rollout, slow velocity 0.03129, and
fast velocity 0.01543. Polarity was correct, including all three target
switches, so the main peak is a content-recovery failure rather than an
inversion-label failure.

## Interpretation

The error curve contains high-frequency spikes inside broad high-error regions.
Teacher forcing repeatedly steps across difficult silhouettes and cuts using
slightly different true histories, so individual local transitions alternate
between easy and hard. The free rollout adds a broader state mismatch beneath
those spikes. V4.2 should reduce the broad region by training on its own
corrupted states and should sharpen transition recovery without turning it into
a hard reset.
