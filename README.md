# Neural network dreams Bad Apple

Proof of concept for reconstructing **00:45–01:00** of Bad Apple from neural
network activations.

The pipeline is intentionally made of small, replaceable blocks:

1. `extract` writes the source segment to `prototype_data/source_frames`.
2. `train` learns either the plain or attention autoencoder.
3. `reconstruct` saves raw output-neuron probabilities, thresholded black/white
   activations, and (for the attention model) the learned attention maps.
4. `train-ar` freezes that autoencoder and trains a recurrent next-latent
   predictor.
5. `rollout-ar` removes the source after a short context window and visualizes
   accumulated prediction error.

Every generated pixel is the sigmoid activation of one decoder location. A
pixel is white when that activation is at least `0.5`, otherwise it is black.
The optional `attention` model also learns a spatial gate in its bottleneck.
This tests attention without coupling the rest of the pipeline to it.

## Setup

Use a Python environment with PyTorch, then install the small remaining
dependencies:

```powershell
python -m pip install -r requirements.txt
```

In this workspace the intended environment is `torch-gpu`.

## Run

Extract the original 30 fps frames:

```powershell
python prototype.py extract
```

Train and render the plain baseline:

```powershell
python prototype.py train --model basic --epochs 8
python prototype.py reconstruct --checkpoint prototype_runs/basic/model_best.pt
```

Run the attention experiment by changing `basic` to `attention`.

For a first smoke test, use:

```powershell
python prototype.py train --model basic --epochs 1 --height 96 --width 128 --base-channels 8 --latent-channels 32
```

Run the lightweight shape/data checks with:

```powershell
python -m unittest discover -s tests
```

The working resolution is independent from the source extraction. Defaults are
`192x256` to keep experiments quick. Height and width are freely configurable;
the network pads and crops internally when they are not multiples of 16.

```powershell
python prototype.py train --model basic --height 384 --width 512 --batch-size 4 --run-dir prototype_runs/basic_full
```

## Prototype results

All results use every tenth frame for validation.

| Model | Resolution | Validation loss | Pixel accuracy | Mean binary IoU |
| --- | ---: | ---: | ---: | ---: |
| attention | 192×256 | 0.0414 | 0.9897 | 0.9739 |
| basic | 192×256 | 0.0414 | 0.9896 | 0.9736 |
| basic | 384×512 | **0.0097** | **0.9967** | **0.9909** |

Attention works, but its same-resolution improvement is only `0.0003` IoU and
the learned focus maps are diffuse. The plain model is therefore the current
default. The attention block remains available for future experiments.

## Autoregressive dream

The autoregressive block observes 16 true latent frames (0.5 seconds), then the
source is cut off. Every later latent is predicted from the previous predicted
latent and the ConvGRU state. It is never reset during the remaining rollout.

Train the predictor against the frozen full-resolution autoencoder:

```powershell
python prototype.py train-ar --autoencoder-checkpoint prototype_runs/basic_full/model_best.pt --run-dir prototype_runs/autoregressive_warm16 --epochs 8 --sequence-length 16 --rollout-warmup-frames 16
```

Render the drift experiment:

```powershell
python prototype.py rollout-ar --checkpoint prototype_runs/autoregressive_warm16/model_best.pt --output-dir prototype_outputs/autoregressive_drift
```

The comparison video shows four synchronized panels:

1. Target frame.
2. Teacher-forced prediction, which receives the correct previous latent.
3. Free rollout, which only receives its own previous predictions after cutoff.
4. Error map: red is dreamed white content and cyan is missed white content.

The teacher-forced/free-rollout difference isolates accumulated error from
ordinary one-step prediction error. In the tested run, teacher-forced binary
error averaged `3.97%`; free rollout averaged `28.62%`, with a `24.65%`
accumulation gap. Peak free-rollout error was `60.53%` at frame 337, during a
major inversion transition.

## Temporal U-Net with learned scene memory

The hybrid predictor keeps the same 16-frame causal context but replaces the
ConvGRU motion model with a temporal U-Net. Skip concatenations preserve
short-term motion at high temporal resolution while the bottleneck summarizes
the full history window. Fourier time features address 12 learned spatial
memory tokens, and a learned gate fuses remembered scene content with the
motion forecast.

Train the tested configuration:

```powershell
python prototype.py train-hybrid --run-dir prototype_runs/hybrid_memory --epochs 12 --base-channels 8 --history-length 16 --rollout-steps 4 --batch-size 4 --memory-tokens 12
```

Render it through the same diagnostics:

```powershell
python prototype.py rollout-ar --checkpoint prototype_runs/hybrid_memory/model_best.pt --output-dir prototype_outputs/hybrid_memory
```

| Free-rollout metric after cutoff | ConvGRU | Hybrid memory | Change |
| --- | ---: | ---: | ---: |
| Binary error | 29.66% | **17.32%** | -41.6% |
| Accumulation gap | 25.75% | **13.36%** | -48.1% |
| Mean binary IoU | 0.442 | **0.666** | +50.8% |
| Peak binary error | 60.53% | **51.08%** | -15.6% |

The memory gate averages `0.274` after cutoff and changes its dominant token 35
times. Memory weights, gate values, and address entropy are included per frame
in `error_curve.csv`. The hybrid remembers recognizable characters and scene
structure much longer. The remaining largest failure is the major black/white
inversion near frame 335.

### Hybrid v2: polarity, scene cuts, and rollout curriculum

Hybrid v2 canonicalizes every training frame to a black background before
encoding. A separate time-conditioned polarity head restores black/white
orientation after decoding. Memory tokens are initialized from high-change
scene frames, addressing uses temperature `0.5` plus entropy regularization,
and the training rollout expands from 4 to 16 steps.

```powershell
python prototype.py train-hybrid --run-dir prototype_runs/hybrid_v2 --epochs 12 --base-channels 8 --history-length 16 --minimum-rollout-steps 4 --rollout-steps 16 --batch-size 4 --memory-tokens 12 --memory-temperature 0.5
python prototype.py rollout-ar --checkpoint prototype_runs/hybrid_v2/model_best.pt --output-dir prototype_outputs/hybrid_v2
```

| Free-rollout metric after cutoff | ConvGRU | Hybrid v1 | Hybrid v2 |
| --- | ---: | ---: | ---: |
| Binary error | 29.66% | 17.32% | **14.19%** |
| Accumulation gap | 25.75% | 13.36% | **9.04%** |
| Mean binary IoU | 0.442 | 0.666 | **0.686** |
| Final-frame error | 31.68% | 25.13% | **15.48%** |

Polarity accuracy reaches `99.1%`. The inversion around frame 337 is corrected:
free-rollout error there falls from roughly `60%` in v1 to `14.5%` in v2.
The remaining peak is `77.46%` at frame 380, where polarity is correct but the
model fails to retrieve the new scene content. Memory address entropy remains
high (`0.877`), making sharper scene selection the next bottleneck.

### Hybrid v3: temporal polarity tracking

The v2 border heuristic becomes unreliable when a large silhouette touches the
frame edges. Hybrid v3 anchors polarity once, then compares each low-resolution
frame with both orientations. It only switches when the inverted orientation is
more temporally consistent by a configurable margin.

```powershell
python prototype.py train-hybrid --run-dir prototype_runs/hybrid_v3_temporal --epochs 12 --base-channels 8 --history-length 16 --minimum-rollout-steps 4 --rollout-steps 16 --batch-size 4 --memory-tokens 12 --memory-temperature 0.5
python prototype.py rollout-ar --checkpoint prototype_runs/hybrid_v3_temporal/model_best.pt --output-dir prototype_outputs/hybrid_v3_temporal
```

The old detector changes polarity eight times. The temporal path changes once,
at the real global inversion near frame 336. In the problematic frame 356–388
window, teacher error falls from `22.63%` to `4.66%`, all seven teacher spikes
above `20%` disappear, and free-rollout error falls from `33.28%` to `15.22%`.

| Post-cutoff metric | Hybrid v2 | Hybrid v3 temporal |
| --- | ---: | ---: |
| Teacher binary error | 5.15% | **3.71%** |
| Free-rollout binary error | 14.19% | **14.10%** |
| Peak free-rollout error | 77.46% | **33.03%** |
| Accumulation gap | **9.04%** | 10.38% |
| Mean binary IoU | 0.675 | **0.690** |
| Final-frame error | **15.48%** | 17.43% |

The repaired representation removes catastrophic inversions and cuts the worst
failure substantially. Average long-horizon quality is nearly unchanged,
showing that scene-content retrieval is now the main limitation rather than
polarity detection.

### Hybrid v4: velocity and bleeding scene anchors

Hybrid v4 keeps scene recovery soft so errors bleed into the next character
instead of causing a clean reset. It adds explicit latent velocity, a supervised
motion mask, 16 timeline-covering anchors with deterministic top-2 addressing,
and a spatial anchor gate capped at `35%`. Moving regions suppress the anchor
gate, while static regions receive a gradual scene-memory correction.

Training uses motion-weighted latent losses and truncated backpropagation,
expanding from 4 to 32 rollout steps. A final linear calibration exposes the
polarity head to every timestamp equally; it reaches one predicted switch
matching the one temporal-polarity target switch.

```powershell
python prototype.py train-hybrid-v4 --run-dir prototype_runs/hybrid_v4_bleed --epochs 12 --minimum-rollout-steps 4 --rollout-steps 32 --truncated-backprop-steps 4 --base-channels 8 --anchors 16 --batch-size 2 --maximum-anchor-gate 0.35
python prototype.py rollout-ar --checkpoint prototype_runs/hybrid_v4_bleed/model_best.pt --output-dir prototype_outputs/hybrid_v4_bleed
```

| Post-cutoff metric | Hybrid v3 temporal | Hybrid v4 bleed |
| --- | ---: | ---: |
| Teacher binary error | 3.71% | **3.62%** |
| Free-rollout binary error | 14.10% | **7.29%** |
| Accumulation gap | 10.38% | **3.66%** |
| Mean binary IoU | 0.690 | **0.805** |
| Peak free-rollout error | 33.03% | **26.76%** |
| Final-frame error | 17.43% | **14.98%** |

In the hand-and-wing motion interval at frames 240–300, rollout error falls
from `14.14%` to `7.32%`. Mean frame-to-frame pixel motion rises from `0.34%`
to `0.70%`, compared with the target's `1.78%`, and the union of pixels that
participate in motion rises from `14.5%` to `25.5%` versus the target's
`35.2%`. Motion is therefore no longer frozen, although it remains deliberately
deformed and temporally damped.

### Hybrid v4.1: dual-scale local motion

V4.1 warm-starts from the v4 checkpoint and splits latent velocity into two
branches. The slow branch retains the original `0.5` step limit for stable
whole-shape motion. A new fast branch can contribute up to `2.0` per latent
cell, but is multiplied by the learned motion mask so it concentrates on
hands, wings, edges, and other changing regions. The two velocities are added
before the existing spatial scene-memory bleed.

Fine-tuning is staged to protect the working scene memory: first only the fast
head and motion mask train, then the temporal U-Net motion path, and finally
the spatial anchor gate. Timeline anchors, time addressing, and polarity stay
frozen. The epoch-zero v4 checkpoint is also retained as the fallback if
fine-tuning regresses.

```powershell
python prototype.py train-hybrid-v4 --run-dir prototype_runs/hybrid_v4_1_motion --warm-start-checkpoint prototype_runs/hybrid_v4_bleed/model_best.pt --dual-velocity --epochs 8 --minimum-rollout-steps 16 --rollout-steps 32 --truncated-backprop-steps 4 --base-channels 8 --anchors 16 --batch-size 2 --maximum-anchor-gate 0.35 --fast-head-only-epochs 2 --motion-only-epochs 6 --learning-rate 0.0001 --slow-velocity-loss-weight 0.25 --fast-velocity-loss-weight 0.25 --fast-velocity-dynamic-weight 2.0
python prototype.py rollout-ar --checkpoint prototype_runs/hybrid_v4_1_motion/model_best.pt --output-dir prototype_outputs/hybrid_v4_1_motion
```

| Post-cutoff metric | Hybrid v4 bleed | Hybrid v4.1 motion |
| --- | ---: | ---: |
| Teacher binary error | 3.62% | **3.43%** |
| Free-rollout binary error | 7.29% | **5.92%** |
| Accumulation gap | 3.66% | **2.50%** |
| Mean binary IoU | 0.805 | **0.841** |
| Peak free-rollout error | 26.76% | **25.73%** |
| Final-frame error | 14.98% | **12.31%** |

At frames 240-300, mean frame-to-frame motion reaches `0.94%`, up from
v4's `0.70%` and equal to `53%` of the target's `1.78%`. The moving-pixel
union reaches `28.4%`, or `81%` of the target's `35.2%`, while interval
rollout error falls from `7.32%` to `6.29%`. The fast branch is active rather
than decorative: its post-cutoff mean latent magnitude is `0.0213`, alongside
`0.0528` from the slow branch. V4.1 preserves the chosen soft bleed; the mean
effective rollout anchor gate remains `13.2%`.

## Outputs

- `prototype_data/manifest.json`: exact source segment and extraction metadata.
- `prototype_runs/<model>/history.json`: loss, pixel accuracy, and mean binary
  IoU for comparison.
- `prototype_outputs/<model>/probability_activations.mp4`: raw activations.
- `prototype_outputs/<model>/binary_activations.mp4`: `0/1` neuron view.
- `prototype_outputs/attention/attention_maps.mp4`: learned focus gate.
- `prototype_outputs/basic_full/binary_activations.mp4`: best tested
  full-resolution reconstruction.
- `prototype_outputs/autoregressive_drift/comparison.mp4`: synchronized
  target/control/rollout/error diagnostic.
- `prototype_outputs/autoregressive_drift/free_rollout.mp4`: uninterrupted
  autoregressive dream.
- `prototype_outputs/autoregressive_drift/error_curve.png` and
  `error_curve.csv`: frame-level accumulation measurements.
- `prototype_outputs/hybrid_memory/comparison.mp4`: hybrid target/control/
  rollout/error comparison.
- `prototype_outputs/hybrid_memory/free_rollout.mp4`: hybrid v1
  time-conditioned memory rollout.
- `prototype_outputs/hybrid_v2/comparison.mp4`: v2 synchronized diagnostic.
- `prototype_outputs/hybrid_v2/free_rollout.mp4`: polarity-canonical,
  scene-memory rollout.
- `prototype_outputs/hybrid_v3_temporal/comparison.mp4`: temporally tracked
  polarity diagnostic.
- `prototype_outputs/hybrid_v3_temporal/free_rollout.mp4`: temporal-polarity
  autoregressive dream.
- `prototype_outputs/hybrid_v4_bleed/comparison.mp4`: velocity, spatial bleed,
  and error diagnostic.
- `prototype_outputs/hybrid_v4_bleed/free_rollout.mp4`: softly bleeding v4
  autoregressive dream.
- `prototype_outputs/hybrid_v4_1_motion/comparison.mp4`: dual-scale motion,
  soft scene-memory bleed, and error diagnostic.
- `prototype_outputs/hybrid_v4_1_motion/free_rollout.mp4`: v4.1 local-motion
  autoregressive dream.

The current defaults make no style decisions beyond preserving the source
aspect ratio and using a neutral `0.5` black/white threshold. Temporal effects,
intentional distortion, inversion, blur, and other artistic choices are left
open for discussion.
