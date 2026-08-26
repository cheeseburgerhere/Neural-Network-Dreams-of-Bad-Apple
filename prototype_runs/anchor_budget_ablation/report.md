# Anchor-budget ablation

**Status:** Prepared; no training started

## Question

How much explicit spatial scene memory is needed for long-horizon rollout, and
where does additional anchor capacity stop buying useful quality?

## Controlled design

- Same V4.2 temporal U-Net, time features, dual velocity, cut gate, loss,
  6,573 frames, seed `7`, and 12-epoch schedule for every new run.
- Only `anchor_count` changes.
- Every checkpoint is rendered once before polarity repair, then receives the
  same polarity-spline fit and a final metrics-only rollout.
- Primary comparison: final rollout binary error and accumulation gap.
- Secondary comparison: IoU, peak error, scene-cut windows, and 53–55 s
  boundary quality.
- The existing 220-anchor V4.2 polarity-fixed run is the reference; it does
  not need to be retrained.

This deliberately tests V4.2 before the V4.3 recovery fine-tune. Recovery only
trains the motion path and would add cost plus a second experimental variable.
Run recovery on the Pareto-best smaller budget only if the memory curve is
promising.

## Parameter budgets

One anchor stores `64 × 24 × 32 = 49,152` trainable values. The shared
non-memory predictor contains 133,190 parameters.

| Anchors | Anchor parameters | Total predictor parameters | Status |
| ---: | ---: | ---: | --- |
| 0 | 0 | 133,190 | Prepared control |
| 16 | 786,432 | 919,622 | Prepared |
| 32 | 1,572,864 | 1,706,054 | Prepared |
| 55 | 2,703,360 | 2,836,550 | Prepared |
| 110 | 5,406,720 | 5,539,910 | Prepared |
| 220 | 10,813,440 | 10,946,630 | Existing reference |

## Run order

Start with `0`, `32`, and the existing `220` reference. That is enough to show
whether memory matters and whether 32 anchors recover a meaningful fraction of
the quality. Run `16`, `55`, and `110` afterward to locate the knee of the
curve. This order can stop early without weakening the first conclusion.

Each prepared variant folder contains its exact train and evaluation commands.
Nothing in this folder launches training automatically.

## Result table

Fill this from each final `drift_summary.json` after the runs finish.

| Anchors | Rollout error | Accumulation gap | IoU | Peak error | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | — | — | — | — | Pending |
| 16 | — | — | — | — | Pending |
| 32 | — | — | — | — | Pending |
| 55 | — | — | — | — | Pending |
| 110 | — | — | — | — | Pending |
| 220 | 0.049289 | 0.017169 | 0.843972 | 0.385223 | Reference |

## Hardware note

The experiment is ready for either `torch-gpu` locally or a Colab A100. Keep
the batch size identical across variants for the cleanest comparison. If an
A100 batch-size benchmark motivates changing it, change it for every new
variant and treat the existing 220 run as a reference rather than a perfectly
controlled training replicate.
