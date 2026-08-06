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
- Automatic runtime cutoff: 10.0 minutes.
- Hard shutdown loses at most one checkpoint interval.

```powershell
New-Item "D:\Code_archive\Bad_apple\prototype_runs\hybrid_v4_3_recovery_smoke\STOP" -ItemType File
```

Remove `STOP`, then rerun the same command to resume.

## Configuration

```json
{
  "checkpoint": "D:\\Code_archive\\Bad_apple\\prototype_runs\\hybrid_v4_2_polarity_fix\\model_best.pt",
  "latent_cache": "D:\\Code_archive\\Bad_apple\\prototype_data\\cache\\v4_2_canonical_latents_fp16.pt",
  "run_dir": "D:\\Code_archive\\Bad_apple\\prototype_runs\\hybrid_v4_3_recovery_smoke",
  "epochs": 1,
  "history_length": 16,
  "rollout_steps": 4,
  "truncated_backprop_steps": 4,
  "minimum_burn_in_steps": 0,
  "burn_in_steps": 4,
  "clean_batch_interval": 5,
  "batch_size": 2,
  "learning_rate": 5e-05,
  "recovery_velocity_weight": 0.25,
  "scene_velocity_weight": 0.05,
  "dynamic_loss_weight": 0.5,
  "motion_mask_loss_weight": 0.05,
  "checkpoint_every_minutes": 1.0,
  "max_runtime_minutes": 10.0,
  "max_batches_per_epoch": 2,
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
| 1 | 1.314594 | 0.406740 | 0.294051 | 0.108198 | 0.337003 | 24.2 |

## Reproduction

```powershell
python prototype.py fine-tune-recovery --run-dir "prototype_runs\hybrid_v4_3_recovery_smoke" --epochs 1 --max-runtime-minutes 10.0
```
