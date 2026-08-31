# Neural Network Dreams Bad Apple — Hybrid v4.3-state-recovery render

**Checkpoint:** `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\hybrid_v4_3_recovery\model_best.pt`

## What is in this folder

- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.
- MP4 generation was disabled for this metrics-only run.

## Headline metrics

| Metric | Value |
| --- | ---: |
| `frame_count` | 6573 |
| `fps` | 30.000000 |
| `post_cutoff_mean_teacher_binary_error` | 0.032147 |
| `post_cutoff_mean_rollout_binary_error` | 0.081682 |
| `post_cutoff_mean_accumulation_gap` | 0.049535 |
| `post_cutoff_mean_rollout_iou` | 0.775055 |
| `peak_error_frame` | 822 |
| `peak_error_seconds` | 27.400000 |
| `peak_rollout_binary_error` | 0.768799 |
| `final_rollout_binary_error` | 0.000000 |

## Memory and motion diagnostics

```json
{
  "token_count": 220,
  "mean_gate": 0.11999956518411636,
  "post_cutoff_mean_gate": 0.11994514614343643,
  "minimum_gate": 0.06801880151033401,
  "maximum_gate": 0.17506268620491028,
  "mean_address_entropy": 0.08437451719028977,
  "dominant_token_changes": 219,
  "post_cutoff_teacher_effective_gate": 0.23683199286460876,
  "post_cutoff_rollout_effective_gate": 0.26331648230552673,
  "post_cutoff_teacher_motion_mask": 0.3577769100666046,
  "post_cutoff_rollout_motion_mask": 0.26181530952453613,
  "post_cutoff_rollout_slow_velocity": 0.03788946568965912,
  "post_cutoff_rollout_fast_velocity": 0.031713489443063736,
  "post_cutoff_teacher_cut_gate": 0.24920965731143951,
  "post_cutoff_rollout_cut_gate": 0.20085662603378296,
  "post_cutoff_rollout_anchor_disagreement": 0.3021511137485504
}
```

## Interpretation notes

- Teacher-forced output measures one-step reconstruction with correct latent history.
- Free rollout measures the actual autoregressive dream and therefore includes accumulated state error.
