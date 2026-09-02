# Notes for editing the blog draft

The article is [BLOG_POST.md](BLOG_POST.md). It is written in your voice as a starting draft, not something you need to publish unchanged. It is intentionally a personal project story, not a corporate report or a novelty claim.

## Before publishing

- Replace the local video links with whichever public uploads you choose. No media or article has been published by this task. Keep the figures alongside the Markdown, or update their URLs when cross-posting.
- Add the exact original music and animation credits and a working source-video link. The inspiration article links to https://www.youtube.com/watch?v=FtutLA63Cp8, but that upload could not be verified live during this draft. Do not treat this unresolved credit/link check as finished.
- Check the equations, code blocks, images, and video links in each platform's preview. The article uses ordinary Markdown links, PNG figures, and LaTeX equations; no platform-specific embed syntax is assumed.
- Read the first-person passages and change anything that doesn't sound like you. The motivation, concern about compute, desire to see drift, and choice to let scenes bleed are grounded in our conversation; no personal anecdotes were invented.
- Consider an assistance credit if you want one: “I used Codex throughout the implementation, experiments, diagnostics, and the first draft of this post.”
- The draft is around 3,400 words. If you want a shorter version, the recovery-target section can become a follow-up post. Keep the memory-only caveat and polarity correction even in a shorter version.

## What the article must not accidentally claim

The main table compares the matched anchor-budget runs, all with the separately fitted 96-knot polarity spline. The 220 row is the corrected checkpoint, not its raw result.

The separate v4.3 state-recovery experiment is described as a lesson from development. Its improvements are not attributed to the main anchor-budget table. The latter uses the v4.2-style long-horizon recipe.

“Teacher” and “rollout” are evaluation modes of the same checkpoint, not two separately trained models.

The 16 source warmup frames are all black in the full-video experiment. Every later source frame is withheld during free rollout, but the predictor is trained on the whole sequence and still receives time.

The autoencoder's basic_full name means full resolution, not full-video training. Its config points to the 450-frame 45–60 s prototype. It is frozen and reused for full-video prediction. Do not turn the prototype's interleaved-frame validation score into a held-out-video result.

Memory-only uses the full model's jointly trained anchors. It removes history, temporal processing, velocity prediction, and fusion together. It is neither an independently trained baseline nor proof that the U-Net alone contributes the entire gain.

The recovery equation treats the current gate and memory candidate as fixed. It describes the local supervision target, not a closed-form guarantee about the nonlinear recurrent model.

There is one training seed per budget. The study does not establish statistical significance, general-video generation, or codec efficiency. Pixel error also does not fully measure contour quality.

## Evidence and source map

All local paths below are relative to this file.

| Article topic | Controlling evidence |
| --- | --- |
| Original 45–60 s segment | [Prototype manifest](../prototype_data/manifest.json) |
| Autoencoder scope and configuration | [Autoencoder config](../prototype_runs/basic_full/config.json), [initial project notes](../README.md) |
| Full video: 6,573 frames, 30 fps, 219.1 s | [Full manifest](../prototype_data/full_manifest.json) |
| Actual temporal model and memory fusion | [Model implementation](../neural_bad_apple/hybrid_v4.py), [architecture notes](architecture.md) |
| Long-horizon training | [220 config](../prototype_runs/anchors_220/config.json), [long-horizon run report](../prototype_runs/hybrid_v4_2_long_horizon/report.md) |
| Recovery equation and result | [Recovery code](../neural_bad_apple/recovery.py), [completed recovery report](../prototype_runs/hybrid_v4_3_recovery/report.md) |
| Main quality and memory comparison | [Quality CSV](tables/01_quality.csv), [memory CSV](tables/02_memory_only.csv), [source summaries](tables/source_summaries.json) |
| Parameter counts and anchor spacing | [Parameter CSV](tables/03_parameters.csv), [spacing CSV](tables/05_anchor_spacing.csv) |
| Polarity correction and unchanged weights | [Correction report](220_polarity_correction.md), [verification JSON](220_polarity_verification.json) |
| Whole-video per-frame measurements | [220 corrected curve](tables/anchors_220_polarity_per_frame.csv), corresponding CSVs for the other budgets |
| Media identity and frame verification | [Asset manifest](asset_manifest.json) |

The run-level reports and saved results take precedence over the old training-run index, which still calls the now-completed v4.2 experiment untrained. The index was not edited as part of writing this draft.

The initial attention comparison is described qualitatively. Its prototype validation is not mixed into the final full-video scores.

External sources checked:

- [Valérien Braye's original project article](https://brayevalerien.com/blog/bad-apple-but-its-gpt2/): frozen GPT-2 XL with optimized per-frame input embeddings. The blog draft only summarizes the inspiration, not its performance claims.
- [The original Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1r5lra1/bad_apple_but_its_gpt2_xl_attention_maps/): contains the previous-frame/latent generation suggestion. These are an article and discussion of the same project, not two independent implementations.
- Original video upload/credits: not verified; see the publishing check above.

## Validation and visual notes

Validation completed: all six quality rows and five memory-only rows were independently recomputed from their per-frame CSVs. The displayed rounding, relative reductions, and parameter arithmetic match. Every local Markdown link resolves and code/math delimiters are balanced. Hash checks confirm all six checkpoints are unchanged and all fourteen current videos match the asset manifest. On September 2, the anchor-budget montage was intentionally re-encoded with attached header bars and panel borders; its underlying seven streams and model results did not change.

The draft uses the corrected, fresh local inference measurements consistently. It does not mix those with older Colab summary values. It is ready for author editing with the stated experimental caveats; exact original-media credits and publishing-platform previews remain pre-publication checks.

Highest-impact calculations to retain:

- Scored frames: 6,573 − 16 = 6,557.
- One anchor: 64 × 24 × 32 = 49,152 parameters.
- 220 anchors: 220 × 49,152 = 10,813,440 parameters.
- Memory share of predictor parameters: 10,813,440 / 10,946,630 ≈ 98.8%.
- Full-system relative error reduction versus memory-only:
  - 32 anchors: 1 − 0.0937821256036309 / 0.37197653816164575 ≈ 74.8%.
  - 220 anchors: 1 − 0.05634751677919152 / 0.14237929532235945 ≈ 60.4%.
- The separate recovery report compares full-rollout latent MSE 0.337478 → 0.326779. These are not pixel-error percentages.

The article reuses two existing figures without editing their data or appearance:

| Article section | Figure | Why it is here |
| --- | --- | --- |
| Is this just memory playback? | [Memory-only comparison](figures/02_memory_only_ablation.png) | Grouped bars compare the same two evaluation paths at five budgets; zero baseline; blue/gold plus hatching. |
| Is this just memory playback? | [Error accumulation](figures/03_error_accumulation.png) | Unsmoothed per-frame errors for 32 and 220; same time and error scales expose failures hidden by averages. |

Exact parameter counts use a table rather than another chart. The architecture uses text diagrams, not generated artwork. Additional budget/parameter figures remain in the asset pack but are not embedded redundantly.

The technical-report roles were adapted to the requested personal blog format: the opening supplies the result, the development story supplies methods and model specification, “What I measured” defines the experiment before the comparison, limitations stay beside claims, and the ending combines further questions with reflection. No corporate executive-summary section, dashboard, HTML report, or publishing action was added because the user selected an editable blog draft.

## Files changed for the article

The article draft and these editorial notes were added. No model, notebook, training pipeline, checkpoint, or result table was changed. The anchor-budget montage was later re-encoded to fix ambiguous label placement; the previous version is preserved in `../prototype_outputs/blog_work/anchor_budget_full_before_header_fix.mp4`.



