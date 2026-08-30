# Anchor-budget ablation — 55 anchors

**Status:** Training and full evaluation completed on Colab; summary and CSV received locally

Anchor parameters: **2,703,360**. Total predictor parameters: **2,836,550**.

## Train

```powershell
python prototype.py train-hybrid-v42 --anchors 55 --run-dir prototype_runs/anchor_budget_ablation/anchors_055 --epochs 12 --batch-size 2 --seed 7 --device cuda
```

## Evaluate and repair polarity

```powershell
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_055/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_055_raw --batch-size 16 --device cuda --fps 30 --no-video
python prototype.py fix-polarity --checkpoint prototype_runs/anchor_budget_ablation/anchors_055/model_best.pt --target-csv prototype_outputs/anchor_budget_ablation/anchors_055_raw/error_curve.csv --run-dir prototype_runs/anchor_budget_ablation/anchors_055_polarity --device cuda
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_055_polarity/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_055_final --batch-size 16 --device cuda --fps 30 --no-video
```

## Full-video result

| Metric | Value |
| --- | ---: |
| Teacher-forced binary error | 0.032004 |
| Free-rollout binary error | 0.096726 |
| Accumulation gap | 0.064721 |
| Rollout IoU | 0.745313 |
| Final-frame error | 0.187002 |
| Peak error | 0.592834 at 110.67 s |
| Polarity accuracy | 1.000000 |

The supplied frame-level curve is preserved at
`prototype_outputs/anchor_budget_ablation/anchors_055_final/error_curve.csv`.

## Distribution and failure diagnosis

- Median rollout error: 0.077423; p90: 0.189911; p95: 0.228968; p99: 0.333623.
- 554 frames (8.43%) exceed 0.20 error; 109 frames (1.66%) exceed 0.30.
- Removing the worst 1% of frames lowers mean error from 0.096490 to 0.093381. This is slightly below the untrimmed 32-anchor mean of 0.093853, although a proper distribution comparison still requires the 32-anchor CSV.
- The main acute failure lasts 24 frames from 110.30-111.07 s: mean error 0.442940, peak 0.592834. Error recovers to 0.020981 by 111.67 s, so this is a short transition failure rather than persistent collapse.
- Memory token 26 is the worst region (109.30-112.43 s): rollout error 0.21 versus teacher error 0.03, leaving a 0.18 accumulation gap.
- Rollout error correlates strongly with teacher difficulty (0.68), cut gate (0.65), motion mask (0.64), and fast velocity (0.55), but weakly with memory gate (0.13) and anchor disagreement (0.08).

## Interpretation

Fifty-five anchors improve ordinary-frame fidelity and endpoint recovery, but introduce or expose a heavier transition-error tail. Extra memory capacity is no longer the main bottleneck. Hard cuts and fast local dynamics dominate the remaining large failures. Complete the controlled 110-anchor run before changing the architecture; afterward test cut-triggered hard routing or a brief anchor snap instead of stronger continuous bleeding.
