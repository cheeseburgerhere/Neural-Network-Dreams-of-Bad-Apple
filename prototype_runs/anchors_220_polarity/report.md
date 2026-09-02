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

- Checkpoint: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\anchors_220_polarity\model_best.pt`
- Metrics: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\anchors_220_polarity\calibration.json`
- Source checkpoint: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_runs\anchors_220\model_best.pt`
- Polarity targets: `C:\Users\cheeseburgerhere\OneDrive\Belgeler\GitHub\Neural-Network-Dreams-of-Bad-Apple\prototype_outputs\blog_work\anchors_220\metrics.csv`

## Reproduction

```powershell
python prototype.py fix-polarity --checkpoint "prototype_runs\anchors_220\model_best.pt" --target-csv "prototype_outputs\blog_work\anchors_220\metrics.csv" --run-dir "prototype_runs\anchors_220_polarity"
```


## Verification and final blog results

The correction was checked against the original checkpoint, not just against its rendered scores.

- All 63 original state tensors are bitwise unchanged; only `polarity_spline_logits` was added (96 scalars).
- Raw checkpoint SHA-256 remains `102a62c7e551adcbb51ff98ef5e3643a7f53345fbbc8c12e26d5b051210cfe07`.
- Corrected checkpoint SHA-256: `4502f0674d98020e3a5571e250e97212f2404111eeb784f5fdfaaf2717153ba2`.
- Latent normalization is unchanged. Sampled forward latents and gates are bitwise identical.
- Every one of the 6,573 rollout memory gates matches the original run.
- For teacher, rollout and memory-only outputs, per-frame pixel-error changes match exactly the expected polarity inversion (maximum floating-point discrepancy below 1.2e-16).
- 41 regression tests pass, including checkpoint preservation and unchanged latent dynamics.
- All 14 exported videos passed decoded frame-count and duration checks. Four figures and sampled corrected video frames were visually inspected.

| Metric, excluding 16 warmup frames | Raw 220 | Calibrated 220 |
| --- | --- | --- |
| Teacher pixel error | 13.82% | 3.13% |
| Rollout pixel error | 15.73% | 5.63% |
| Memory-only pixel error | 22.59% | 14.24% |
| Polarity errors / 6,557 frames | 749 | 0 |

The corrected full system reduces pixel error by 60.4% relative to decoding the same jointly trained memories directly. This is a post-hoc removal comparison, not an independently trained memory-only baseline. The polarity fix does not remove the remaining silhouette and motion errors.

The polarity spline receives normalized time only during generation. Its fitting uses labels from this known video, as did the smaller-budget calibrations. These are reconstruction results, not held-out generalization results.

Exact preservation checks: [verification.json](verification.json).
Publishable media, tables and caveats: [blog asset index](../../blog_assets/README.md).
The raw checkpoint and inference results remain in their original folders; the entire previous blog pack was moved intact to `prototype_outputs/blog_assets_before_220_polarity`.

No main-model training was performed for this fix.

