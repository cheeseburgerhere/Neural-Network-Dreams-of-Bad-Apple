# Neural Network Dreams Bad Apple — Hybrid v4.2 render

**Checkpoint:** `/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_032/model_best.pt`

## What is in this folder

- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.
- MP4 generation was disabled for this metrics-only run.

## Headline metrics

| Metric | Value |
| --- | ---: |
| `frame_count` | 6573 |
| `fps` | 30.000000 |
| `post_cutoff_mean_teacher_binary_error` | 0.139138 |
| `post_cutoff_mean_rollout_binary_error` | 0.189876 |
| `post_cutoff_mean_accumulation_gap` | 0.050738 |
| `post_cutoff_mean_rollout_iou` | 0.663682 |
| `peak_error_frame` | 1312 |
| `peak_error_seconds` | 43.733333 |
| `peak_rollout_binary_error` | 0.990122 |
| `final_rollout_binary_error` | 0.402705 |

## Memory and motion diagnostics

```json
{
  "token_count": 32,
  "mean_gate": 0.08721671998500824,
  "post_cutoff_mean_gate": 0.08722647279500961,
  "minimum_gate": 0.048283349722623825,
  "maximum_gate": 0.13272903859615326,
  "mean_address_entropy": 0.13557901969977146,
  "dominant_token_changes": 31,
  "post_cutoff_teacher_effective_gate": 0.08628144860267639,
  "post_cutoff_rollout_effective_gate": 0.061443645507097244,
  "post_cutoff_teacher_motion_mask": 0.17601731419563293,
  "post_cutoff_rollout_motion_mask": 0.20186084508895874,
  "post_cutoff_rollout_slow_velocity": 0.0339190810918808,
  "post_cutoff_rollout_fast_velocity": 0.021135393530130386,
  "post_cutoff_teacher_cut_gate": 0.07890605926513672,
  "post_cutoff_rollout_cut_gate": 0.05004370957612991,
  "post_cutoff_rollout_anchor_disagreement": 0.6805664896965027
}
```

## Interpretation notes

- Teacher-forced output measures one-step reconstruction with correct latent history.
- Free rollout measures the actual autoregressive dream and therefore includes accumulated state error.
