# Neural Network Dreams Bad Apple — Hybrid v4.2 render

**Checkpoint:** `/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_055/model_best.pt`

## What is in this folder

- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.
- MP4 generation was disabled for this metrics-only run.

## Headline metrics

| Metric | Value |
| --- | ---: |
| `frame_count` | 6573 |
| `fps` | 30.000000 |
| `post_cutoff_mean_teacher_binary_error` | 0.139710 |
| `post_cutoff_mean_rollout_binary_error` | 0.194167 |
| `post_cutoff_mean_accumulation_gap` | 0.054457 |
| `post_cutoff_mean_rollout_iou` | 0.664830 |
| `peak_error_frame` | 1480 |
| `peak_error_seconds` | 49.333333 |
| `peak_rollout_binary_error` | 0.994359 |
| `final_rollout_binary_error` | 0.187002 |

## Memory and motion diagnostics

```json
{
  "token_count": 55,
  "mean_gate": 0.11805884540081024,
  "post_cutoff_mean_gate": 0.11807394027709961,
  "minimum_gate": 0.06258796155452728,
  "maximum_gate": 0.16920152306556702,
  "mean_address_entropy": 0.12864114539535346,
  "dominant_token_changes": 54,
  "post_cutoff_teacher_effective_gate": 0.09692954272031784,
  "post_cutoff_rollout_effective_gate": 0.08011984825134277,
  "post_cutoff_teacher_motion_mask": 0.23476949334144592,
  "post_cutoff_rollout_motion_mask": 0.20340147614479065,
  "post_cutoff_rollout_slow_velocity": 0.038360998034477234,
  "post_cutoff_rollout_fast_velocity": 0.023233562707901,
  "post_cutoff_teacher_cut_gate": 0.07213075459003448,
  "post_cutoff_rollout_cut_gate": 0.0463998056948185,
  "post_cutoff_rollout_anchor_disagreement": 0.6209083795547485
}
```

## Interpretation notes

- Teacher-forced output measures one-step reconstruction with correct latent history.
- Free rollout measures the actual autoregressive dream and therefore includes accumulated state error.
