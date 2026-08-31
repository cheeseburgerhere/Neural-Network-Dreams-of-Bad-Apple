# Neural Network Dreams Bad Apple — Hybrid v4.2

**Status:** Training complete

## Purpose and architecture

- A causal temporal U-Net predicts slow and fast latent velocity from the previous latent window.
- 110 time-local scene memories bleed into the motion candidate with a normal cap of 0.35.
- Time features use the `seconds` basis across 219.07 seconds.
- Anchor temperature is `spacing` and resolved to 0.003766.
- A learned cut gate can temporarily raise scene correction to 0.65; its disagreement signal naturally decays as the rollout approaches memory.
- Each supervised window starts after a randomly sampled 0-128 frame free-running burn-in with no gradient retained through it.
- Scene-memory tensors remain frozen for the first 6 epochs.

## Reproduction command

```powershell
/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 110 --run-dir "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_110" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4
```

## Configuration

```json
{
  "autoencoder_checkpoint": "/content/bad_apple/prototype_runs/basic_full/model_best.pt",
  "frame_dir": "/content/bad_apple/prototype_data/full_source_frames",
  "run_dir": "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_110",
  "history_length": 16,
  "minimum_rollout_steps": 4,
  "rollout_steps": 32,
  "truncated_backprop_steps": 4,
  "base_channels": 8,
  "anchor_count": 110,
  "anchor_temperature": 0.0037659823894500735,
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
  "reproduction_command": "/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 110 --run-dir \"/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_110\" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4",
  "seed": 7,
  "device": "cuda",
  "anchor_initialization_indices": [
    0,
    122,
    243,
    357,
    365,
    443,
    458,
    466,
    479,
    487,
    529,
    609,
    730,
    820,
    852,
    974,
    1095,
    1217,
    1339,
    1365,
    1460,
    1471,
    1508,
    1516,
    1524,
    1574,
    1582,
    1683,
    1704,
    1714,
    1826,
    1884,
    1910,
    1947,
    2069,
    2191,
    2246,
    2312,
    2434,
    2503,
    2556,
    2677,
    2732,
    2799,
    2862,
    2921,
    2939,
    2948,
    3043,
    3164,
    3272,
    3286,
    3408,
    3462,
    3471,
    3515,
    3529,
    3577,
    3616,
    3624,
    3651,
    3672,
    3688,
    3773,
    3895,
    4016,
    4138,
    4187,
    4260,
    4381,
    4394,
    4403,
    4503,
    4566,
    4576,
    4592,
    4602,
    4625,
    4746,
    4868,
    4990,
    5023,
    5112,
    5233,
    5320,
    5335,
    5343,
    5355,
    5477,
    5598,
    5628,
    5660,
    5720,
    5842,
    5946,
    5963,
    6085,
    6207,
    6277,
    6285,
    6298,
    6319,
    6329,
    6343,
    6356,
    6368,
    6383,
    6450,
    6514,
    6572
  ],
  "resolved_device": "cuda",
  "latent_shape": [
    64,
    24,
    32
  ],
  "parameter_count": 5539814
}
```

## Checkpoint

`/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_110/model_best.pt`

## Training history

| Epoch | Stage | Burn-in | Rollout | Train loss | Rollout MSE | Peak MSE | Seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | memory-frozen | 62.6825 | 4 | 2.272068 | 0.823605 | 1.871681 | 197.278059 |
| 2 | memory-frozen | 64.27 | 7 | 2.068582 | 0.762138 | 2.003773 | 219.003838 |
| 3 | memory-frozen | 62.0075 | 9 | 1.983964 | 0.713451 | 1.606473 | 227.469748 |
| 4 | memory-frozen | 65.185 | 12 | 1.925422 | 0.682951 | 2.091113 | 252.551540 |
| 5 | memory-frozen | 60.83 | 14 | 1.858980 | 0.646052 | 1.677462 | 258.511980 |
| 6 | memory-frozen | 64.255 | 17 | 1.836639 | 0.620207 | 1.495560 | 284.189815 |
| 7 | full | 63.6075 | 19 | 1.766629 | 0.578402 | 1.549748 | 296.389581 |
| 8 | full | 63.51 | 22 | 1.690435 | 0.543309 | 1.842762 | 315.765982 |
| 9 | full | 62.765 | 24 | 1.616552 | 0.511282 | 1.261619 | 328.313746 |
| 10 | full | 62.2375 | 27 | 1.569679 | 0.496332 | 1.308200 | 347.454337 |
| 11 | full | 62.8875 | 29 | 1.525805 | 0.467391 | 1.292570 | 360.560214 |
| 12 | full | 62.6225 | 32 | 1.454358 | 0.445190 | 1.343319 | 380.228156 |

## Notes

- Polarity calibration accuracy: 0.8851.
- Use the matching output-folder report for binary-error and visual rollout metrics.
