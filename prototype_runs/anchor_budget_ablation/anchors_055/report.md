# Anchor-budget ablation — 55 anchors

**Status:** Prepared; no training started

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
