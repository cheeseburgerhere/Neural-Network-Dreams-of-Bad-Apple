# Hybrid V4 bleed training log

**Status:** Complete

## Purpose

V4 replaced hard scene retrieval with a spatially gated correction that can
only move the prediction partway toward a time-addressed scene anchor. The
intended visual behavior is error that bleeds out rather than a clean reset.

The motion path is a causal temporal U-Net over 16 latent frames. It predicts
latent velocity and a motion mask. Sixteen timeline-covering scene anchors use
deterministic top-2 addressing; static areas receive more memory correction and
moving areas receive less. The normal memory gate is capped at 0.35.

## Reproduction

```powershell
python prototype.py train-hybrid-v4 --run-dir prototype_runs/hybrid_v4_bleed --epochs 12 --minimum-rollout-steps 4 --rollout-steps 32 --truncated-backprop-steps 4 --base-channels 8 --anchors 16 --batch-size 2 --maximum-anchor-gate 0.35
```

## Result

On the 45-60 second prototype, post-cutoff teacher error was 3.62%, free-rollout
error was 7.29%, and mean IoU was 0.805. Peak rollout error was 26.76%.

V4 established that soft memory correction can substantially reduce
autoregressive drift without removing the desired dream-like deformation. Its
main remaining weakness was underpowered local movement in hands and wings.
