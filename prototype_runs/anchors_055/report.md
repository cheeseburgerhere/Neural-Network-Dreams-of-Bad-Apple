# Neural Network Dreams Bad Apple — Hybrid v4.2

**Status:** Training complete

## Purpose and architecture

- A causal temporal U-Net predicts slow and fast latent velocity from the previous latent window.
- 55 time-local scene memories bleed into the motion candidate with a normal cap of 0.35.
- Time features use the `seconds` basis across 219.07 seconds.
- Anchor temperature is `spacing` and resolved to 0.007190.
- A learned cut gate can temporarily raise scene correction to 0.65; its disagreement signal naturally decays as the rollout approaches memory.
- Each supervised window starts after a randomly sampled 0-128 frame free-running burn-in with no gradient retained through it.
- Scene-memory tensors remain frozen for the first 6 epochs.

## Reproduction command

```powershell
/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 55 --run-dir "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_055" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4
```

## Configuration

```json
{
  "autoencoder_checkpoint": "/content/bad_apple/prototype_runs/basic_full/model_best.pt",
  "frame_dir": "/content/bad_apple/prototype_data/full_source_frames",
  "run_dir": "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_055",
  "history_length": 16,
  "minimum_rollout_steps": 4,
  "rollout_steps": 32,
  "truncated_backprop_steps": 4,
  "base_channels": 8,
  "anchor_count": 55,
  "anchor_temperature": 0.007189592532813549,
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
  "anchor_loss_weight": 0.01,
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
  "reproduction_command": "/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 55 --run-dir \"/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_055\" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4",
  "seed": 7,
  "device": "cuda",
  "anchor_initialization_indices": [
    0,
    253,
    358,
    466,
    480,
    506,
    758,
    820,
    1011,
    1264,
    1471,
    1509,
    1517,
    1525,
    1578,
    1769,
    1884,
    2022,
    2246,
    2275,
    2503,
    2528,
    2780,
    2862,
    3033,
    3272,
    3286,
    3462,
    3471,
    3515,
    3539,
    3616,
    3672,
    3792,
    4044,
    4297,
    4394,
    4550,
    4566,
    4576,
    4592,
    4602,
    4803,
    5023,
    5055,
    5308,
    5335,
    5343,
    5561,
    5628,
    5814,
    6066,
    6277,
    6319,
    6572
  ],
  "resolved_device": "cuda",
  "latent_shape": [
    64,
    24,
    32
  ],
  "parameter_count": 2836454
}
```

## Checkpoint

`/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_055/model_best.pt`

## Training history

| Epoch | Stage | Burn-in | Rollout | Train loss | Rollout MSE | Peak MSE | Seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | memory-frozen | 62.6825 | 4 | 2.490116 | 0.946071 | 2.104908 | 199.370604 |
| 2 | memory-frozen | 64.27 | 7 | 2.200702 | 0.867592 | 1.869673 | 222.332728 |
| 3 | memory-frozen | 62.0075 | 9 | 2.121811 | 0.803718 | 2.109248 | 231.655871 |
| 4 | memory-frozen | 65.185 | 12 | 2.069030 | 0.767672 | 1.838913 | 257.236758 |
| 5 | memory-frozen | 60.83 | 14 | 2.010925 | 0.748930 | 1.763189 | 262.490208 |
| 6 | memory-frozen | 64.255 | 17 | 1.965443 | 0.716978 | 1.587657 | 287.978994 |
| 7 | full | 63.6075 | 19 | 1.901983 | 0.651925 | 1.659595 | 300.325306 |
| 8 | full | 63.51 | 22 | 1.833063 | 0.618922 | 1.431560 | 320.563940 |
| 9 | full | 62.765 | 24 | 1.766570 | 0.579923 | 1.404845 | 332.590601 |
| 10 | full | 62.2375 | 27 | 1.701929 | 0.571320 | 1.569073 | 351.794146 |
| 11 | full | 62.8875 | 29 | 1.644071 | 0.532086 | 1.298025 | 365.150991 |
| 12 | full | 62.6225 | 32 | 1.594089 | 0.497831 | 1.334374 | 385.717848 |

## Notes

- Polarity calibration accuracy: 0.8854.
- Use the matching output-folder report for binary-error and visual rollout metrics.
