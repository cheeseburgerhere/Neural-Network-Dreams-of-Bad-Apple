# Neural Network Dreams Bad Apple — Hybrid v4.2-polarity-fix render

**Checkpoint:** `/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_000_polarity/model_best.pt`

## What is in this folder

- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.
- MP4 generation was disabled for this metrics-only run.

## Headline metrics

| Metric | Value |
| --- | ---: |
| `frame_count` | 6573 |
| `fps` | 30.000000 |
| `post_cutoff_mean_teacher_binary_error` | 0.031732 |
| `post_cutoff_mean_rollout_binary_error` | 0.445350 |
| `post_cutoff_mean_accumulation_gap` | 0.413618 |
| `post_cutoff_mean_rollout_iou` | 0.330777 |
| `peak_error_frame` | 6468 |
| `peak_error_seconds` | 215.600000 |
| `peak_rollout_binary_error` | 0.766368 |
| `final_rollout_binary_error` | 0.197657 |

## Memory and motion diagnostics

```json
{
  "token_count": 0,
  "mean_gate": 0.0,
  "post_cutoff_mean_gate": 0.0,
  "minimum_gate": 0.0,
  "maximum_gate": 0.0,
  "mean_address_entropy": 0.0,
  "dominant_token_changes": 0,
  "post_cutoff_teacher_effective_gate": 0.0,
  "post_cutoff_rollout_effective_gate": 0.0,
  "post_cutoff_teacher_motion_mask": 0.1784825325012207,
  "post_cutoff_rollout_motion_mask": 0.1839407980442047,
  "post_cutoff_rollout_slow_velocity": 0.007790878415107727,
  "post_cutoff_rollout_fast_velocity": 0.005968281999230385,
  "post_cutoff_teacher_cut_gate": 0.0,
  "post_cutoff_rollout_cut_gate": 0.0,
  "post_cutoff_rollout_anchor_disagreement": 0.0
}
```

## Interpretation notes

- Teacher-forced output measures one-step reconstruction with correct latent history.
- Free rollout measures the actual autoregressive dream and therefore includes accumulated state error.
