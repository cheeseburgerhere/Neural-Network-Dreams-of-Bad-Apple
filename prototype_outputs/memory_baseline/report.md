# Does the current model do more than play back its anchors?

## Outcome

Yes, for the existing 32- and 55-anchor checkpoints, the complete model substantially outperforms directly decoding the same blended memories. This is a post-hoc removal experiment, not a comparison against an independently trained memory-only model.

| Anchors | Memory-only pixel error | Full-model pixel error | Reduction with full model | Frames where full model wins | Memory-only IoU | Full-model IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.371977 | 0.094082 | 74.7% | 94.6% | 0.403546 | 0.741114 |
| 55 | 0.321062 | 0.096726 | 69.9% | 91.2% | 0.459035 | 0.745313 |

Both evaluations use 6,573 frames at 30 FPS. Scores exclude the first 16 frames, exactly as the reference rollout scores do. The memory-only path itself receives no source frames, even during that excluded interval.

## What the baseline does

At each timestamp it selects the same two nearest anchor times and uses the same distance-softmax weights as the original model. It forms `memory_latent = sum(weight_i * learned_anchor_i)`, reverses the checkpoint's latent normalization, then uses the frozen autoencoder decoder. Pixel threshold and learned polarity restoration are unchanged.

The temporal U-Net, predicted slow/fast velocity, recurrent history, and memory-fusion gates are not used. The model checkpoint is never modified and no optimization is performed.

The full-model side is taken from each checkpoint's existing, frame-aligned rollout CSV. Frame count, frame order, FPS, image dimensions, and anchor count are checked where available in the reference summary. Full-model inference was not rerun on this machine.

## Where the difference is largest

| Anchors | Distance to nearest anchor | Frames | Memory-only error | Full-model error |
| ---: | --- | ---: | ---: | ---: |
| 32 | At most 0.25 s | 430 | 0.203000 | 0.138491 |
| 32 | 0.25-1.0 s | 1,047 | 0.280922 | 0.102380 |
| 32 | More than 1.0 s | 5,080 | 0.405046 | 0.088613 |
| 55 | At most 0.25 s | 764 | 0.195957 | 0.142146 |
| 55 | 0.25-1.0 s | 1,682 | 0.277012 | 0.096697 |
| 55 | More than 1.0 s | 4,111 | 0.362335 | 0.088297 |

Direct memory playback becomes particularly inaccurate between anchors. In inspected 32-anchor previews at 10, 90, 110.7 and 180 seconds, it often shows a different pose or scene rather than merely a blurred version of the correct one.

Do not interpret the full model's higher error near anchors as evidence that anchors cause errors. Anchor selection favors difficult changes, so the distance bins contain different scene difficulties. The meaningful comparison is between the two methods on the same frames within each bin.

## What this establishes, and what it does not

- It establishes that the current output cannot be reproduced simply by decoding the current time-blended anchors. The rest of the trained system makes a large contribution.
- It does not isolate the temporal U-Net from recurrent state, learned time features, or gated correction; all are removed together in this diagnostic.
- The anchors were optimized jointly with the full system. They may be useful correction references rather than the best possible independently decodable keyframes. A separately trained memory-only baseline could perform much better.
- This remains reconstruction of one known video with explicit time input, not evidence of general video prediction or research novelty.
- The matched large-batch 220-anchor checkpoint was not present locally and was not used. These two diagnostics did not need it. Repeat the same evaluation after its weights and rollout CSV are available before making claims about that checkpoint.

## Artifacts and reproduction

Each of `anchors_032/` and `anchors_055/` contains `comparison.mp4`, `error_curve.csv`, `summary.json` and `report.md`. Videos show the source target on the left and memory-only output on the right; they do not contain a newly rendered full-model panel. Both MP4s were verified to contain 6,573 frames and 219.1 seconds.

Run from the repository root, choosing a new output directory because existing results are never overwritten:

```powershell
python -m neural_bad_apple.memory_baseline --checkpoint prototype_runs/anchors_032_polarity/model_best.pt --reference-csv prototype_outputs/anchors_032_final/error_curve.csv --output-dir prototype_outputs/memory_baseline/anchors_032_repeat --device cuda --batch-size 8
```

Replace `032` with `055` for the second run. The module is independent of the existing training and rendering entry points. All 34 tests passed, including four new checks for memory-only execution, zero-anchor rejection, reference alignment, and warmup exclusion. Source frames were regenerated from the existing local video; the missing pinned FFmpeg wrapper was installed in the existing `torch-gpu` environment. The user's notebook modification was not touched.
