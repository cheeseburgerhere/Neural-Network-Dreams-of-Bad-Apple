# Hybrid V4.2 polarity fix

**Status:** Complete

## Result

- Selected 96-knot normalized-time linear spline.
- Accuracy: 100.0000%.
- Switches: 3 predicted / 3 target.
- Mismatch frames: 0.
- Temporal U-Net, latent heads, scene memory, and gates unchanged.

## Full-video validation

- Raw post-cutoff teacher binary error: 3.2120%.
- Raw post-cutoff rollout binary error: 4.9289%.
- Accumulation gap: 1.7169%.
- Polarity accuracy: 100%.
- Predicted polarity switches: 3 / 3 target switches.
- Peak rollout error: 38.5223% at frame 358 (11.93 seconds).

The raw render now exactly matches the earlier true-polarity counterfactual.
This confirms the spline fix changes only presentation polarity and preserves
V4.2 content dynamics.

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

- Checkpoint: `D:\Code_archive\Bad_apple\prototype_runs\hybrid_v4_2_polarity_fix\model_best.pt`
- Metrics: `D:\Code_archive\Bad_apple\prototype_runs\hybrid_v4_2_polarity_fix\calibration.json`
- Full render metrics: `D:\Code_archive\Bad_apple\prototype_outputs\hybrid_v4_2_polarity_fix\drift_summary.json`
- Source checkpoint: `D:\Code_archive\Bad_apple\prototype_runs\hybrid_v4_2_long_horizon\model_best.pt`
- Polarity targets: `D:\Code_archive\Bad_apple\prototype_outputs\hybrid_v4_2_long_horizon\error_curve.csv`

## Reproduction

```powershell
python prototype.py fix-polarity --checkpoint "prototype_runs\hybrid_v4_2_long_horizon\model_best.pt" --target-csv "prototype_outputs\hybrid_v4_2_long_horizon\error_curve.csv" --run-dir "prototype_runs\hybrid_v4_2_polarity_fix"
```
