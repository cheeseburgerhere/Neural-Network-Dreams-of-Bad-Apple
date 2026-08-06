# Neural Network Dreams Bad Apple — Hybrid v4.2

**Status:** Training complete

## Purpose and architecture

- A causal temporal U-Net predicts slow and fast latent velocity from the previous latent window.
- 220 time-local scene memories bleed into the motion candidate with a normal cap of 0.35.
- Time features use the `seconds` basis across 219.07 seconds.
- Anchor temperature is `spacing` and resolved to 0.001575.
- A learned cut gate can temporarily raise scene correction to 0.65; its disagreement signal naturally decays as the rollout approaches memory.
- Each supervised window starts after a randomly sampled 0-128 frame free-running burn-in with no gradient retained through it.
- Scene-memory tensors remain frozen for the first 6 epochs.

## Reproduction command

```powershell
C:\Users\alt_user\miniconda3\envs\torch-gpu\python.exe prototype.py train-hybrid-v42
```

## Configuration

```json
{
  "autoencoder_checkpoint": "D:\\Code_archive\\Bad_apple\\prototype_runs\\basic_full\\model_best.pt",
  "frame_dir": "D:\\Code_archive\\Bad_apple\\prototype_data\\full_source_frames",
  "run_dir": "D:\\Code_archive\\Bad_apple\\prototype_runs\\hybrid_v4_2_long_horizon",
  "history_length": 16,
  "minimum_rollout_steps": 4,
  "rollout_steps": 32,
  "truncated_backprop_steps": 4,
  "base_channels": 8,
  "anchor_count": 220,
  "anchor_temperature": 0.0015748590230941772,
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
  "batch_size": 2,
  "learning_rate": 0.0001,
  "warm_start_checkpoint": null,
  "fast_head_only_epochs": 0,
  "motion_only_epochs": 0,
  "minimum_burn_in_steps": 0,
  "burn_in_steps": 128,
  "freeze_memory_epochs": 6,
  "architecture_version": "v4.2",
  "reproduction_command": "C:\\Users\\alt_user\\miniconda3\\envs\\torch-gpu\\python.exe prototype.py train-hybrid-v42",
  "seed": 7,
  "device": "auto",
  "anchor_initialization_indices": [
    0,
    60,
    121,
    181,
    241,
    301,
    362,
    422,
    443,
    458,
    466,
    474,
    482,
    529,
    543,
    603,
    663,
    724,
    784,
    797,
    820,
    844,
    904,
    965,
    1025,
    1060,
    1076,
    1085,
    1102,
    1146,
    1206,
    1220,
    1266,
    1293,
    1326,
    1365,
    1387,
    1447,
    1471,
    1507,
    1516,
    1524,
    1568,
    1578,
    1628,
    1688,
    1714,
    1722,
    1730,
    1749,
    1798,
    1809,
    1854,
    1869,
    1884,
    1892,
    1910,
    1929,
    1990,
    2050,
    2110,
    2171,
    2208,
    2231,
    2246,
    2254,
    2291,
    2307,
    2315,
    2323,
    2351,
    2412,
    2472,
    2495,
    2503,
    2532,
    2593,
    2653,
    2713,
    2724,
    2732,
    2774,
    2820,
    2834,
    2851,
    2862,
    2894,
    2922,
    2930,
    2938,
    2946,
    2954,
    3015,
    3075,
    3135,
    3196,
    3256,
    3272,
    3299,
    3316,
    3376,
    3437,
    3454,
    3462,
    3471,
    3497,
    3505,
    3515,
    3529,
    3537,
    3557,
    3577,
    3618,
    3640,
    3649,
    3668,
    3678,
    3688,
    3738,
    3755,
    3798,
    3809,
    3817,
    3825,
    3859,
    3919,
    3950,
    3979,
    3994,
    4040,
    4100,
    4160,
    4187,
    4221,
    4248,
    4261,
    4281,
    4341,
    4364,
    4393,
    4401,
    4418,
    4462,
    4522,
    4549,
    4566,
    4582,
    4592,
    4602,
    4621,
    4643,
    4675,
    4683,
    4694,
    4703,
    4763,
    4792,
    4823,
    4844,
    4870,
    4884,
    4944,
    5004,
    5023,
    5036,
    5065,
    5125,
    5185,
    5214,
    5246,
    5306,
    5320,
    5335,
    5343,
    5354,
    5366,
    5405,
    5426,
    5487,
    5547,
    5607,
    5628,
    5660,
    5668,
    5676,
    5684,
    5692,
    5700,
    5708,
    5728,
    5788,
    5824,
    5833,
    5848,
    5909,
    5946,
    5959,
    5969,
    6029,
    6090,
    6108,
    6135,
    6150,
    6164,
    6210,
    6271,
    6280,
    6298,
    6319,
    6331,
    6343,
    6356,
    6368,
    6383,
    6391,
    6451,
    6459,
    6470,
    6512,
    6572
  ],
  "resolved_device": "cuda",
  "latent_shape": [
    64,
    24,
    32
  ],
  "parameter_count": 10946534
}
```

## Checkpoint

`D:\Code_archive\Bad_apple\prototype_runs\hybrid_v4_2_long_horizon\model_best.pt`

## Training history

| Epoch | Stage | Burn-in | Rollout | Train loss | Rollout MSE | Peak MSE | Seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | memory-frozen | 63.30697092841513 | 4 | 1.874571 | 0.603592 | 2.058060 | 1488.902949 |
| 2 | memory-frozen | 63.674273210378246 | 7 | 1.729391 | 0.562032 | 1.826638 | 1602.027639 |
| 3 | memory-frozen | 63.78899656142544 | 9 | 1.670833 | 0.530509 | 1.583976 | 2007.990115 |
| 4 | memory-frozen | 64.89402938418256 | 12 | 1.604543 | 0.509577 | 1.543456 | 1902.893940 |
| 5 | memory-frozen | 64.66301969365426 | 14 | 1.561807 | 0.502281 | 1.631557 | 1935.781050 |
| 6 | memory-frozen | 64.57986870897156 | 17 | 1.523537 | 0.486394 | 1.410486 | 2028.997817 |
| 7 | full | 63.85589246639575 | 19 | 1.474021 | 0.442811 | 1.448142 | 2160.958332 |
| 8 | full | 63.904657705532976 | 22 | 1.415197 | 0.421580 | 1.599957 | 2426.858422 |
| 9 | full | 63.87402313222882 | 24 | 1.361555 | 0.394353 | 1.129929 | 3915.212002 |
| 10 | full | 63.480150046889655 | 27 | 1.309443 | 0.372906 | 1.206604 | 2736.007549 |
| 11 | full | 65.01625507971241 | 29 | 1.273045 | 0.361736 | 1.115185 | 2432.191405 |
| 12 | full | 63.93372929040325 | 32 | 1.215874 | 0.337478 | 1.078125 | 2449.951395 |

## Notes

- Polarity calibration accuracy: 0.8847.
- Use the matching output-folder report for binary-error and visual rollout metrics.

## Outcome

Training completed all 12 epochs in approximately 7.52 hours with no stderr
output. Best validation rollout latent MSE was 0.33748 at epoch 12, 27.3% below
the V4.1 full-timeline baseline.

The canonical content rollout is substantially more stable, but the polarity
head predicts 23 switches instead of three. See
`prototype_outputs/hybrid_v4_2_long_horizon/report.md` and
`findings_report.html` for the polarity-corrected evaluation and recommended
next experiment.
