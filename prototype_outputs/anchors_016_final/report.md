# Neural Network Dreams Bad Apple — Hybrid v4.2-polarity-fix render

**Checkpoint:** `D:\Code_archive\Bad_apple\prototype_runs\anchor_budget_ablation\anchors_016_polarity\model_best.pt`

## What is in this folder

- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.
- `comparison.mp4`: target, teacher-forced prediction, free rollout, and error map.
- `free_rollout.mp4`: uninterrupted autoregressive dream.
- `teacher_forced.mp4`: one-step control using true history.
- `error_maps.mp4`: false positives in red and misses in cyan.

## Headline metrics

| Metric | Value |
| --- | ---: |
| `frame_count` | 6573 |
| `fps` | 30.000000 |
| `post_cutoff_mean_teacher_binary_error` | 0.032089 |
| `post_cutoff_mean_rollout_binary_error` | 0.144104 |
| `post_cutoff_mean_accumulation_gap` | 0.112015 |
| `post_cutoff_mean_rollout_iou` | 0.641834 |
| `peak_error_frame` | 6334 |
| `peak_error_seconds` | 211.133333 |
| `peak_rollout_binary_error` | 0.671992 |
| `final_rollout_binary_error` | 0.245921 |

## Memory and motion diagnostics

```json
{
  "token_count": 16,
  "mean_gate": 0.059118643403053284,
  "post_cutoff_mean_gate": 0.05904931202530861,
  "minimum_gate": 0.01990467496216297,
  "maximum_gate": 0.12858565151691437,
  "mean_address_entropy": 0.16292576968705344,
  "dominant_token_changes": 15,
  "post_cutoff_teacher_effective_gate": 0.06634022295475006,
  "post_cutoff_rollout_effective_gate": 0.042975250631570816,
  "post_cutoff_teacher_motion_mask": 0.19742555916309357,
  "post_cutoff_rollout_motion_mask": 0.1930636614561081,
  "post_cutoff_rollout_slow_velocity": 0.0273685771971941,
  "post_cutoff_rollout_fast_velocity": 0.015757819637656212,
  "post_cutoff_teacher_cut_gate": 0.07569092512130737,
  "post_cutoff_rollout_cut_gate": 0.04156109690666199,
  "post_cutoff_rollout_anchor_disagreement": 0.7943358421325684
}
```

## Interpretation notes

- Teacher-forced output measures one-step reconstruction with correct latent history.
- Free rollout measures the actual autoregressive dream and therefore includes accumulated state error.
