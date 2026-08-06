# Prototype data log

This folder contains the source material used by **Neural network dreams Bad
Apple**. Extracted PNG frames are intentionally ignored by Git; the small JSON
manifests are the reproducible record.

## Datasets

| Dataset | Source interval | Frames | FPS | Purpose |
| --- | ---: | ---: | ---: | --- |
| `source_frames` | 00:45-01:00 | 450 | 30 | Fast 15-second prototype and artistic evaluation |
| `full_source_frames` | 00:00-03:39.1 | 6,573 | 30 | Long-horizon collapse experiment |

Both datasets preserve the original 512x384 aspect ratio and are stored as
grayscale frames. Training resizes them through the frozen autoencoder.

## Reproduction

```powershell
python prototype.py extract
python prototype.py extract --start 0 --end 219.1 --output-dir prototype_data/full_source_frames --manifest prototype_data/full_manifest.json
```

The full extraction already exists. Re-extraction is unnecessary unless the
source video or preprocessing changes.

## Artistic boundary

The data pipeline makes no aesthetic decisions beyond grayscale conversion and
the existing black/white threshold. Intentional distortion, timing changes,
inversions, or alternate silhouettes require an explicit artistic decision.
