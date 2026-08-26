# Anchor-budget ablation — 16 anchors

**Status:** Prepared; no training started

Anchor parameters: **786,432**. Total predictor parameters: **919,622**.

## Train

```powershell
python prototype.py train-hybrid-v42 --anchors 16 --run-dir prototype_runs/anchor_budget_ablation/anchors_016 --epochs 12 --batch-size 2 --seed 7 --device cuda
```

## Evaluate and repair polarity

```powershell
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_016/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_016_raw --batch-size 16 --device cuda --fps 30 --no-video
python prototype.py fix-polarity --checkpoint prototype_runs/anchor_budget_ablation/anchors_016/model_best.pt --target-csv prototype_outputs/anchor_budget_ablation/anchors_016_raw/error_curve.csv --run-dir prototype_runs/anchor_budget_ablation/anchors_016_polarity --device cuda
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_016_polarity/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_016_final --batch-size 16 --device cuda --fps 30 --no-video
```
