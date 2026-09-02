All quality scores use frames 16–6572 (6,557 frames) at 384×512, before video compression. Polarity handling and checkpoint identity are recorded in source_summaries.json; the calibrated main comparison uses the same spline method across budgets. Memory-only is post-hoc removal, not a separately trained baseline.

## 01 quality

| Model / anchors | Teacher error | Rollout error | Gap (pp) | Class-mean IoU | Polarity accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | 3.17% | 44.82% | 41.65 | 0.329 | 100.00% |
| 16 | 3.21% | 14.35% | 11.14 | 0.643 | 100.00% |
| 32 | 3.27% | 9.38% | 6.11 | 0.741 | 100.00% |
| 55 | 3.20% | 9.67% | 6.47 | 0.745 | 100.00% |
| 110 | 3.14% | 7.76% | 4.62 | 0.776 | 100.00% |
| 220 | 3.13% | 5.63% | 2.50 | 0.828 | 100.00% |


## 02 memory only

| Anchors / variant | Memory-only error | Full error | Relative reduction | Frames full wins |
| --- | --- | --- | --- | --- |
| 16 | 45.84% | 14.35% | 68.7% | 95.0% |
| 32 | 37.20% | 9.38% | 74.8% | 94.6% |
| 55 | 32.11% | 9.67% | 69.9% | 91.2% |
| 110 | 23.49% | 7.76% | 67.0% | 87.8% |
| 220 | 14.24% | 5.63% | 60.4% | 86.9% |


## 03 parameters

| Anchors / variant | Anchor parameters | Temporal U-Net | Other parameters | Predictor total | Frozen AE |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 116,256 | 16,934 | 133,190 | 167,665 |
| 16 | 786,432 | 116,256 | 16,934 | 919,622 | 167,665 |
| 32 | 1,572,864 | 116,256 | 16,934 | 1,706,054 | 167,665 |
| 55 | 2,703,360 | 116,256 | 16,934 | 2,836,550 | 167,665 |
| 110 | 5,406,720 | 116,256 | 16,934 | 5,539,910 | 167,665 |
| 220 | 10,813,440 | 116,256 | 16,934 | 10,946,630 | 167,665 |


## 04 training

| Anchors / variant | Batch | LR | Epochs | Seed | Burn-in max | Rollout max | Memory frozen epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 0.0003 | 12 | 7 | 128 | 32 | 6 |
| 16 | 16 | 0.0003 | 12 | 7 | 128 | 32 | 6 |
| 32 | 16 | 0.0003 | 12 | 7 | 128 | 32 | 6 |
| 55 | 16 | 0.0003 | 12 | 7 | 128 | 32 | 6 |
| 110 | 16 | 0.0003 | 12 | 7 | 128 | 32 | 6 |
| 220 | 16 | 0.0003 | 12 | 7 | 128 | 32 | 6 |


## 05 anchor spacing

| Anchors / variant | Video frames | Mean gap (s) | Median gap (s) | Max gap (s) |
| --- | --- | --- | --- | --- |
| 16 | 6573 | 14.60 | 11.80 | 31.30 |
| 32 | 6573 | 7.07 | 6.17 | 14.63 |
| 55 | 6573 | 4.06 | 3.55 | 8.43 |
| 110 | 6573 | 2.01 | 1.83 | 4.07 |
| 220 | 6573 | 1.00 | 0.77 | 2.03 |

