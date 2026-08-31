# Neural Network Dreams Bad Apple — Hybrid v4.2 render

**Checkpoint:** `/content/drive/MyDrive/neural_bad_apple (1)/prototype_runs/anchor_budget_ablation/anchors_110/model_best.pt`

## What is in this folder

- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.
- MP4 generation was disabled for this metrics-only run.

## Headline metrics

| Metric | Value |
| --- | ---: |
| `frame_count` | 6573 |
| `fps` | 30.000000 |
| `post_cutoff_mean_teacher_binary_error` | 0.139364 |
| `post_cutoff_mean_rollout_binary_error` | 0.177643 |
| `post_cutoff_mean_accumulation_gap` | 0.038279 |
| `post_cutoff_mean_rollout_iou` | 0.693210 |
| `peak_error_frame` | 1480 |
| `peak_error_seconds` | 49.333333 |
| `peak_rollout_binary_error` | 0.998225 |
| `final_rollout_binary_error` | 0.000000 |

## Memory and motion diagnostics

```json
{
  "token_count": 110,
  "mean_gate": 0.14735017716884613,
  "post_cutoff_mean_gate": 0.14731758832931519,
  "minimum_gate": 0.07181580364704132,
  "maximum_gate": 0.2224881500005722,
  "mean_address_entropy": 0.10412801789763575,
  "dominant_token_changes": 109,
  "post_cutoff_teacher_effective_gate": 0.14486907422542572,
  "post_cutoff_rollout_effective_gate": 0.13223156332969666,
  "post_cutoff_teacher_motion_mask": 0.21741655468940735,
  "post_cutoff_rollout_motion_mask": 0.19976146519184113,
  "post_cutoff_rollout_slow_velocity": 0.04518534988164902,
  "post_cutoff_rollout_fast_velocity": 0.0252887811511755,
  "post_cutoff_teacher_cut_gate": 0.09340819716453552,
  "post_cutoff_rollout_cut_gate": 0.0672498494386673,
  "post_cutoff_rollout_anchor_disagreement": 0.5016087293624878
}
```

## Interpretation notes

- Teacher-forced output measures one-step reconstruction with correct latent history.
- Free rollout measures the actual autoregressive dream and therefore includes accumulated state error.
