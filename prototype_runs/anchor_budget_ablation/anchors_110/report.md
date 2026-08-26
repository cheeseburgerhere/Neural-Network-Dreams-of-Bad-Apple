# Anchor-budget ablation — 110 anchors

**Status:** Prepared; no training started

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
