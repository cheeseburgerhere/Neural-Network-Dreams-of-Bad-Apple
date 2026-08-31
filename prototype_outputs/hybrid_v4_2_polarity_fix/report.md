# Neural Network Dreams Bad Apple — Hybrid v4.2-polarity-fix render

**Checkpoint:** `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\hybrid_v4_2_polarity_fix\model_best.pt`

## What is in this folder

- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.
- MP4 generation was disabled for this metrics-only run.

## Headline metrics

| Metric | Value |
| --- | ---: |
| `frame_count` | 6573 |
| `fps` | 30.000000 |
| `post_cutoff_mean_teacher_binary_error` | 0.032120 |
| `post_cutoff_mean_rollout_binary_error` | 0.049289 |
| `post_cutoff_mean_accumulation_gap` | 0.017169 |
| `post_cutoff_mean_rollout_iou` | 0.843972 |
| `peak_error_frame` | 358 |
| `peak_error_seconds` | 11.933333 |
| `peak_rollout_binary_error` | 0.385223 |
| `final_rollout_binary_error` | 0.000000 |

## Memory and motion diagnostics

```json
{
  "token_count": 220,
  "mean_gate": 0.11999956518411636,
  "post_cutoff_mean_gate": 0.11993807554244995,
  "minimum_gate": 0.06801880151033401,
  "maximum_gate": 0.17506268620491028,
  "mean_address_entropy": 0.08437451719028977,
  "dominant_token_changes": 219,
  "post_cutoff_teacher_effective_gate": 0.2383929342031479,
  "post_cutoff_rollout_effective_gate": 0.22885136306285858,
  "post_cutoff_teacher_motion_mask": 0.2683411240577698,
  "post_cutoff_rollout_motion_mask": 0.20127174258232117,
  "post_cutoff_rollout_slow_velocity": 0.04636275768280029,
  "post_cutoff_rollout_fast_velocity": 0.0282468032091856,
  "post_cutoff_teacher_cut_gate": 0.22026030719280243,
  "post_cutoff_rollout_cut_gate": 0.1453084498643875,
  "post_cutoff_rollout_anchor_disagreement": 0.34678035974502563
}
```

## Interpretation notes

- Teacher-forced output measures one-step reconstruction with correct latent history.
- Free rollout measures the actual autoregressive dream and therefore includes accumulated state error.
