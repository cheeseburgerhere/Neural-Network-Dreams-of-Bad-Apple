# Hybrid V4.1 dual-motion training log

**Status:** Complete

## Purpose and architecture

V4.1 kept V4's soft scene-memory bleed and split velocity into two branches:

- A slow branch capped at 0.5 latent units preserves stable whole-shape motion.
- A fast branch capped at 2.0 latent units is multiplied by the learned motion
  mask and targets hands, wings, edges, and other local changes.

This experiment warm-started the successful V4 checkpoint. Epochs 1-2 trained
the new fast head and mask, epochs 3-6 opened the temporal motion path, and
epochs 7-8 also opened the spatial memory gate. Timeline addressing, polarity,
and scene memories stayed protected.

## Reproduction

```powershell
python prototype.py train-hybrid-v4 --run-dir prototype_runs/hybrid_v4_1_motion --warm-start-checkpoint prototype_runs/hybrid_v4_bleed/model_best.pt --dual-velocity --epochs 8 --minimum-rollout-steps 16 --rollout-steps 32 --truncated-backprop-steps 4 --base-channels 8 --anchors 16 --batch-size 2 --maximum-anchor-gate 0.35 --fast-head-only-epochs 2 --motion-only-epochs 6 --learning-rate 0.0001 --slow-velocity-loss-weight 0.25 --fast-velocity-loss-weight 0.25 --fast-velocity-dynamic-weight 2.0
```

## Result

The best epoch reached latent rollout MSE 0.40030 with perfect polarity
classification. In rendered binary frames, post-cutoff teacher error was 3.43%,
free-rollout error was 5.92%, accumulation gap was 2.50%, and mean IoU was
0.841.

During frames 240-300, predicted motion rose from V4's 0.70% to 0.94% per
frame, 53% of target motion. The moving-pixel union reached 28.4%, 81% of the
target. This improved local motion without sacrificing the chosen soft bleed.
