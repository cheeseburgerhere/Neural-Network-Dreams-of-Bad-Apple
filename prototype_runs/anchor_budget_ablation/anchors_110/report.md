# Anchor-budget ablation — 110 anchors

**Status:** Training and full evaluation completed on Colab; summary received locally

Anchor parameters: **5,406,720**. Total predictor parameters: **5,539,910**.

## Train

```powershell
python prototype.py train-hybrid-v42 --anchors 110 --run-dir prototype_runs/anchor_budget_ablation/anchors_110 --epochs 12 --batch-size 2 --seed 7 --device cuda
```

## Evaluate and repair polarity

```powershell
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_110/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_110_raw --batch-size 16 --device cuda --fps 30 --no-video
python prototype.py fix-polarity --checkpoint prototype_runs/anchor_budget_ablation/anchors_110/model_best.pt --target-csv prototype_outputs/anchor_budget_ablation/anchors_110_raw/error_curve.csv --run-dir prototype_runs/anchor_budget_ablation/anchors_110_polarity --device cuda
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_110_polarity/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_110_final --batch-size 16 --device cuda --fps 30 --no-video
```

## Full-video result

| Metric | Value |
| --- | ---: |
| Teacher-forced binary error | 0.031397 |
| Free-rollout binary error | 0.077569 |
| Accumulation gap | 0.046171 |
| Rollout IoU | 0.776510 |
| Final-frame error | 0.000000 |
| Peak error | 0.437637 at 152.50 s |
| Polarity accuracy | 1.000000 |

## Comparison with 55 anchors

- Rollout error improves by 19.8%: 0.096726 to 0.077569.
- Accumulation gap improves by 28.7%: 0.064721 to 0.046171.
- IoU rises by 0.031197: 0.745313 to 0.776510.
- Peak error falls by 26.2%, and final-frame error falls from 0.187002 to zero.
- Effective rollout memory gate rises from 0.080120 to 0.132232 while anchor disagreement falls from 0.620908 to 0.501609.

## Interpretation and control caveat

The apparent 32/55-anchor plateau was not true saturation. Doubling memory to 110 materially improves average drift, tail stability, and endpoint recovery. Returns are strongly diminishing, however: 110 anchors use 5,539,910 parameters, roughly half of the 220-anchor model, and recover about 92.9% of the zero-to-220 rollout-error improvement.

The existing 220-anchor reference is not a perfectly controlled continuation of the Colab ablation. Its recorded training used batch size 2 and learning rate 1e-4, whereas the Colab budget runs used batch size 16 and learning rate 3e-4. Therefore the remaining 110-to-220 quality gap cannot be assigned to anchor count alone without retraining one side under the same optimization recipe.
