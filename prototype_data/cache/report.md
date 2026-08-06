# V4.2 diagnostic latent cache

**Status:** Complete

`v4_2_canonical_latents_fp16.pt` contains the 6,573 canonical, normalized
autoencoder latents used by the silhouette ablation runner.

- Shape before storage: 6,573 x 64 x 24 x 32.
- Storage dtype: float16.
- Polarity labels: uint8.
- Size: approximately 616 MiB.
- Source checkpoint:
  `prototype_runs/hybrid_v4_2_polarity_fix/model_best.pt`.
- Source frames: `prototype_data/full_source_frames`.

The cache avoids encoding the complete video again for every inference
ablation. It is reproducible and may be deleted without losing trained models
or reports; `diagnose-silhouette` recreates it automatically.
