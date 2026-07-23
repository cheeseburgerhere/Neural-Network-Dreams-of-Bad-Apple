# Neural network dreams Bad Apple

Proof of concept for reconstructing **00:45–01:00** of Bad Apple from neural
network activations.

The pipeline is intentionally made of small, replaceable blocks:

1. `extract` writes the source segment to `prototype_data/source_frames`.
2. `train` learns either the plain or attention autoencoder.
3. `reconstruct` saves raw output-neuron probabilities, thresholded black/white
   activations, and (for the attention model) the learned attention maps.

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

## Outputs

- `prototype_data/manifest.json`: exact source segment and extraction metadata.
- `prototype_runs/<model>/history.json`: loss, pixel accuracy, and mean binary
  IoU for comparison.
- `prototype_outputs/<model>/probability_activations.mp4`: raw activations.
- `prototype_outputs/<model>/binary_activations.mp4`: `0/1` neuron view.
- `prototype_outputs/attention/attention_maps.mp4`: learned focus gate.
- `prototype_outputs/basic_full/binary_activations.mp4`: best tested
  full-resolution reconstruction.

The current defaults make no style decisions beyond preserving the source
aspect ratio and using a neutral `0.5` black/white threshold. Temporal effects,
intentional distortion, inversion, blur, and other artistic choices are left
open for discussion.
