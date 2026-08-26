# Anchor-budget ablation — 0 anchors

**Status:** Prepared; no training started

This is the real memory-disabled control. The temporal U-Net, time input,
motion heads, and polarity path remain; memory weights and gates are forced to
zero. Total predictor parameters: **133,190**.

## Train

```powershell
python prototype.py train-hybrid-v42 --anchors 0 --run-dir prototype_runs/anchor_budget_ablation/anchors_000 --epochs 12 --batch-size 2 --seed 7 --device cuda
```

## Evaluate and repair polarity

```powershell
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_000/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_000_raw --batch-size 16 --device cuda --fps 30 --no-video
python prototype.py fix-polarity --checkpoint prototype_runs/anchor_budget_ablation/anchors_000/model_best.pt --target-csv prototype_outputs/anchor_budget_ablation/anchors_000_raw/error_curve.csv --run-dir prototype_runs/anchor_budget_ablation/anchors_000_polarity --device cuda
python prototype.py rollout-ar --checkpoint prototype_runs/anchor_budget_ablation/anchors_000_polarity/model_best.pt --data-dir prototype_data/full_source_frames --output-dir prototype_outputs/anchor_budget_ablation/anchors_000_final --batch-size 16 --device cuda --fps 30 --no-video
```
