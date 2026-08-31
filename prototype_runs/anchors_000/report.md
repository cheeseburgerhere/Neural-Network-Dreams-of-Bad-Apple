# Neural Network Dreams Bad Apple — Hybrid v4.2

**Status:** Training complete

## Purpose and architecture

- A causal temporal U-Net predicts slow and fast latent velocity from the previous latent window.
- Scene memory is disabled for the zero-anchor control.
- Time features use the `seconds` basis across 219.07 seconds.
- Anchor temperature is `spacing` and resolved to 0.030000.
- A learned cut gate can temporarily raise scene correction to 0.65; its disagreement signal naturally decays as the rollout approaches memory.
- Each supervised window starts after a randomly sampled 0-128 frame free-running burn-in with no gradient retained through it.
- Scene-memory tensors remain frozen for the first 6 epochs.

## Reproduction command

```powershell
/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 0 --run-dir "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_000" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4 --anchor-loss-weight 0
```

## Configuration

```json
{
  "autoencoder_checkpoint": "/content/bad_apple/prototype_runs/basic_full/model_best.pt",
  "frame_dir": "/content/bad_apple/prototype_data/full_source_frames",
  "run_dir": "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_000",
  "history_length": 16,
  "minimum_rollout_steps": 4,
  "rollout_steps": 32,
  "truncated_backprop_steps": 4,
  "base_channels": 8,
  "anchor_count": 0,
  "anchor_temperature": 0.03,
  "anchor_temperature_mode": "spacing",
  "anchor_temperature_ratio": 0.45,
  "maximum_anchor_gate": 0.35,
  "maximum_transition_gate": 0.65,
  "anchor_minimum_distance": 8,
  "fourier_frequencies": 6,
  "time_basis": "seconds",
  "timeline_seconds": 219.06666666666666,
  "frames_per_second": 30.0,
  "time_fourier_base_frequency": 0.0625,
  "max_velocity_step": 0.5,
  "use_dual_velocity": true,
  "use_cut_gate": true,
  "max_fast_velocity_step": 2.0,
  "velocity_loss_weight": 0.5,
  "slow_velocity_loss_weight": 0.25,
  "fast_velocity_loss_weight": 0.25,
  "fast_velocity_dynamic_weight": 2.0,
  "dynamic_loss_weight": 0.5,
  "motion_mask_loss_weight": 0.05,
  "cut_gate_loss_weight": 0.05,
  "anchor_loss_weight": 0.0,
  "polarity_loss_weight": 0.2,
  "polarity_calibration_steps": 500,
  "polarity_calibration_learning_rate": 0.03,
  "canonicalize_polarity": true,
  "polarity_tracking_method": "temporal",
  "polarity_switch_penalty": 0.05,
  "latent_noise_standard_deviation": 0.03,
  "epochs": 12,
  "batch_size": 16,
  "learning_rate": 0.0003,
  "warm_start_checkpoint": null,
  "fast_head_only_epochs": 0,
  "motion_only_epochs": 0,
  "minimum_burn_in_steps": 0,
  "burn_in_steps": 128,
  "freeze_memory_epochs": 6,
  "architecture_version": "v4.2",
  "reproduction_command": "/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 0 --run-dir \"/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_000\" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4 --anchor-loss-weight 0",
  "seed": 7,
  "device": "cuda",
  "anchor_initialization_indices": [],
  "resolved_device": "cuda",
  "latent_shape": [
    64,
    24,
    32
  ],
  "parameter_count": 133094
}
```

## Checkpoint

`/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_000/model_best.pt`

## Training history

| Epoch | Stage | Burn-in | Rollout | Train loss | Rollout MSE | Peak MSE | Seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | memory-disabled | 62.6825 | 4 | 4.052898 | 137.467834 | 431.359406 | 170.281903 |
| 2 | memory-disabled | 64.27 | 7 | 2.680215 | 36.666386 | 112.359978 | 190.265232 |
| 3 | memory-disabled | 62.0075 | 9 | 2.573597 | 74.713593 | 332.128448 | 198.554750 |
| 4 | memory-disabled | 65.185 | 12 | 2.516048 | 19.010210 | 63.309658 | 222.035590 |
| 5 | memory-disabled | 60.83 | 14 | 2.442921 | 19.248598 | 62.407581 | 226.626559 |
| 6 | memory-disabled | 64.255 | 17 | 2.407243 | 18.245838 | 57.179703 | 249.012622 |
| 7 | memory-disabled | 63.6075 | 19 | 2.394047 | 14.403286 | 45.862835 | 261.367935 |
| 8 | memory-disabled | 63.51 | 22 | 2.349621 | 30.197701 | 92.918465 | 279.079366 |
| 9 | memory-disabled | 62.765 | 24 | 2.334590 | 11.172194 | 31.947638 | 293.254020 |
| 10 | memory-disabled | 62.2375 | 27 | 2.298635 | 17.343817 | 60.908772 | 308.452795 |
| 11 | memory-disabled | 62.8875 | 29 | 2.277669 | 12.460485 | 46.157825 | 322.330277 |
| 12 | memory-disabled | 62.6225 | 32 | 2.246688 | 10.186395 | 41.642319 | 344.344568 |

## Notes

- Polarity calibration accuracy: 0.8876.
- Use the matching output-folder report for binary-error and visual rollout metrics.
