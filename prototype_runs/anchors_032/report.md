# Neural Network Dreams Bad Apple — Hybrid v4.2

**Status:** Training complete

## Purpose and architecture

- A causal temporal U-Net predicts slow and fast latent velocity from the previous latent window.
- 32 time-local scene memories bleed into the motion candidate with a normal cap of 0.35.
- Time features use the `seconds` basis across 219.07 seconds.
- Anchor temperature is `spacing` and resolved to 0.012667.
- A learned cut gate can temporarily raise scene correction to 0.65; its disagreement signal naturally decays as the rollout approaches memory.
- Each supervised window starts after a randomly sampled 0-128 frame free-running burn-in with no gradient retained through it.
- Scene-memory tensors remain frozen for the first 6 epochs.

## Reproduction command

```powershell
/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 32 --run-dir "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_032" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4
```

## Configuration

```json
{
  "autoencoder_checkpoint": "/content/bad_apple/prototype_runs/basic_full/model_best.pt",
  "frame_dir": "/content/bad_apple/prototype_data/full_source_frames",
  "run_dir": "/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_032",
  "history_length": 16,
  "minimum_rollout_steps": 4,
  "rollout_steps": 32,
  "truncated_backprop_steps": 4,
  "base_channels": 8,
  "anchor_count": 32,
  "anchor_temperature": 0.012667375802993775,
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
  "reproduction_command": "/usr/bin/python3 prototype.py train-hybrid-v42 --anchors 32 --run-dir \"/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_032\" --epochs 12 --batch-size 16 --seed 7 --device cuda --learning-rate 3e-4",
  "seed": 7,
  "device": "cuda",
  "anchor_initialization_indices": [
    0,
    358,
    438,
    480,
    820,
    876,
    1314,
    1516,
    1524,
    1578,
    1753,
    1884,
    2191,
    2503,
    2629,
    3067,
    3462,
    3471,
    3505,
    3515,
    3672,
    3943,
    4381,
    4566,
    4576,
    4592,
    4602,
    4819,
    5258,
    5696,
    6134,
    6572
  ],
  "resolved_device": "cuda",
  "latent_shape": [
    64,
    24,
    32
  ],
  "parameter_count": 1705958
}
```

## Checkpoint

`/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_032/model_best.pt`

## Training history

| Epoch | Stage | Burn-in | Rollout | Train loss | Rollout MSE | Peak MSE | Seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | memory-frozen | 62.6825 | 4 | 2.555557 | 0.954857 | 1.906691 | 217.729463 |
| 2 | memory-frozen | 64.27 | 7 | 2.248275 | 0.889934 | 1.966528 | 240.940149 |
| 3 | memory-frozen | 62.0075 | 9 | 2.154405 | 0.887594 | 2.125355 | 248.951851 |
| 4 | memory-frozen | 65.185 | 12 | 2.098457 | 0.799267 | 1.672847 | 273.819690 |
| 5 | memory-frozen | 60.83 | 14 | 2.024195 | 0.752848 | 1.556193 | 280.573623 |
| 6 | memory-frozen | 64.255 | 17 | 1.983439 | 0.726540 | 1.920747 | 304.975741 |
| 7 | full | 63.6075 | 19 | 1.897231 | 0.688988 | 1.825481 | 317.931852 |
| 8 | full | 63.51 | 22 | 1.830465 | 0.635719 | 1.498305 | 337.295068 |
| 9 | full | 62.765 | 24 | 1.758667 | 0.599020 | 1.629019 | 348.832822 |
| 10 | full | 62.2375 | 27 | 1.693763 | 0.562851 | 1.394906 | 368.016766 |
| 11 | full | 62.8875 | 29 | 1.642229 | 0.530208 | 1.317646 | 382.663165 |
| 12 | full | 62.6225 | 32 | 1.595495 | 0.496081 | 1.288868 | 400.925279 |

## Notes

- Polarity calibration accuracy: 0.8865.
- Use the matching output-folder report for binary-error and visual rollout metrics.
