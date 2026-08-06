# Hybrid V4.1 full-timeline training log

**Status:** Complete; diagnostic baseline for V4.2

## Experiment

The V4.1 architecture was trained from scratch on all 6,573 frames (219.1
seconds) using 220 scene anchors, a 16-frame history, and a 4-to-32-frame
rollout curriculum. It used 10,946,452 parameters and trained for six epochs.
This was not an equal-budget copy of the 15-second V4.1 run: the short model
benefited from a 12-epoch V4 base followed by eight staged fine-tuning epochs.

```powershell
python prototype.py train-hybrid-v4 --data-dir prototype_data/full_source_frames --run-dir prototype_runs/hybrid_v4_1_full --dual-velocity --anchors 220 --epochs 6 --minimum-rollout-steps 4 --rollout-steps 32 --truncated-backprop-steps 4 --base-channels 8 --batch-size 2 --maximum-anchor-gate 0.35 --slow-velocity-loss-weight 0.25 --fast-velocity-loss-weight 0.25 --fast-velocity-dynamic-weight 2.0
```

## Training result

Latent rollout MSE improved from 1.00050 after epoch 1 to 0.46427 after epoch
6. The last epoch took about 31.8 minutes; all six epochs took about 1.96 hours.
Polarity accuracy reached 99.985% during training.

## Collapse diagnosis

- Rendered post-cutoff teacher error was 2.73%, but rollout error was 10.16%.
  The model can predict locally; much of the failure appears only on its own
  states.
- Collapse is episodic around difficult transitions, not a smooth monotonic
  decay. Peak binary error was 82.02% near 110.83 seconds.
- Resetting to true context every 15 seconds improved mean latent MSE only from
  0.46422 to 0.45378, about 2.25%. Old accumulated error is therefore secondary
  to failure to recover at new scenes.
- The fixed normalized anchor temperature expanded its physical bandwidth from
  roughly 0.45 seconds in the short run to about 6.57 seconds here. Neighboring
  anchors became nearly equally weighted.
- Normalized Fourier time bands also stretched with video duration; the finest
  temporal period changed from roughly 0.47 seconds to 6.85 seconds.
- On the identical 45-60 second content, the full model had about 10.89%
  rollout error versus 5.92% for the dedicated short model.

The evidence points to scale-sensitive time/memory coordinates, insufficient
exposure to corrupted rollout states, and scene-transition recovery as the
primary issues. It does not point to VRAM capacity as the root cause.
