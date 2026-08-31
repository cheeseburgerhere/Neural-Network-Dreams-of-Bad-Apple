# Hybrid V4.3 state-recovery fine-tune

**Status:** Training complete

## Purpose

Correct the teacher-good/rollout-bad objective mismatch without retraining scene memory, polarity, or the full temporal backbone.

- Primary velocity target is relative to the model's current predicted state.
- Frozen memory fusion is algebraically included in that target.
- Clean true-scene velocity remains a smaller auxiliary objective.
- Decoder-middle, decoder-high, velocity heads, and motion-mask head are trainable.
- Trainable parameters: 44,121.

## Safe stopping

- Atomic `resume.pt` saved every configured interval.
- `model_last.pt` saved after every completed epoch or graceful stop.
- Create `STOP` in this folder for graceful stop after current batch.
- Automatic runtime cutoff: 100.0 minutes.
- Hard shutdown loses at most one checkpoint interval.

```powershell
New-Item "C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\hybrid_v4_3_recovery\STOP" -ItemType File
```

Remove `STOP`, then rerun the same command to resume.

## Configuration

```json
{
  "checkpoint": "C:\\Users\\cheeseburgerhere\\OneDrive\\Belgeler\\GitHub\\Neural-Network-Dreams-of-Bad-Apple\\prototype_runs\\hybrid_v4_2_polarity_fix\\model_best.pt",
  "latent_cache": "C:\\Users\\cheeseburgerhere\\OneDrive\\Belgeler\\GitHub\\Neural-Network-Dreams-of-Bad-Apple\\prototype_data\\cache\\v4_2_canonical_latents_fp16.pt",
  "run_dir": "C:\\Users\\cheeseburgerhere\\OneDrive\\Belgeler\\GitHub\\Neural-Network-Dreams-of-Bad-Apple\\prototype_runs\\hybrid_v4_3_recovery",
  "epochs": 2,
  "history_length": 16,
  "rollout_steps": 16,
  "truncated_backprop_steps": 4,
  "minimum_burn_in_steps": 32,
  "burn_in_steps": 128,
  "clean_batch_interval": 5,
  "batch_size": 2,
  "learning_rate": 5e-05,
  "recovery_velocity_weight": 0.25,
  "scene_velocity_weight": 0.05,
  "dynamic_loss_weight": 0.5,
  "motion_mask_loss_weight": 0.05,
  "checkpoint_every_minutes": 5.0,
  "max_runtime_minutes": 100.0,
  "max_batches_per_epoch": 0,
  "seed": 7,
  "device": "cuda"
}
```

## Baseline

- Rollout latent MSE: 0.337478.
- Peak latent MSE: 1.078125.

## Completed epochs

| Epoch | Loss | Latent | Recovery velocity | Scene velocity | Rollout MSE | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.113354 | 0.353692 | 0.264731 | 0.131928 | 0.326779 | 1478.2 |
| 2 | 1.104957 | 0.351048 | 0.261518 | 0.130424 | 0.328261 | 1478.6 |

## Final selection

Epoch 1 is the final checkpoint. Epoch 2 lowered training loss but slightly
worsened full-rollout latent MSE and did not improve the important 53-55
second silhouette interval.

| Metric | V4.2 | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: |
| Full rollout latent MSE | 0.337478 | **0.326779** | 0.328261 |
| Sample binary error | 0.051616 | **0.047529** | 0.047996 |
| Sample boundary F1 | 0.443035 | 0.459802 | **0.459844** |
| 53-55s binary error | 0.069343 | **0.064762** | 0.064782 |
| 53-55s boundary F1 | 0.638335 | **0.661605** | 0.659498 |

Teacher precision is preserved. Sample teacher error changes from 0.032282 to
0.031811. In the 53-55 second interval it changes from 0.021808 to 0.021946,
only +0.014 percentage points.

Use `model_best.pt`, which contains epoch 1. `model_last.pt` retains epoch 2
for comparison. No third epoch is recommended.

## Reproduction

```powershell
python prototype.py fine-tune-recovery --run-dir "prototype_runs\hybrid_v4_3_recovery" --epochs 2 --max-runtime-minutes 100.0
```
