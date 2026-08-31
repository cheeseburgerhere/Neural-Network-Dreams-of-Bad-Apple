# Hybrid V4.2 polarity fix

**Status:** Complete

## Result

- Selected 96-knot normalized-time linear spline.
- Accuracy: 100.0000%.
- Switches: 3 predicted / 3 target.
- Mismatch frames: 0.
- Temporal U-Net, latent heads, scene memory, and gates unchanged.

## Candidate sweep

| Knots | Accuracy | Mismatches | Switches | BCE |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.994827 | 34 | 3 | 0.037348 |
| 24 | 0.999544 | 3 | 3 | 0.014985 |
| 32 | 0.999239 | 5 | 3 | 0.011527 |
| 48 | 0.999544 | 3 | 3 | 0.007586 |
| 64 | 0.999696 | 2 | 3 | 0.005419 |
| 96 | 1.000000 | 0 | 3 | 0.003592 |

## Artifacts

- Checkpoint: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\anchor_budget_ablation\anchors_016_polarity\model_best.pt`
- Metrics: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\anchor_budget_ablation\anchors_016_polarity\calibration.json`
- Source checkpoint: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\anchor_budget_ablation\anchors_016\model_best.pt`
- Polarity targets: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_outputs\anchors_000_raw\error_curve.csv`

## Reproduction

```powershell
python prototype.py fix-polarity --checkpoint "prototype_runs\anchor_budget_ablation\anchors_016\model_best.pt" --target-csv "prototype_outputs\anchors_000_raw\error_curve.csv" --run-dir "prototype_runs\anchor_budget_ablation\anchors_016_polarity"
```
