from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import DEFAULT_MANIFEST, FrameDataset, _imageio_ffmpeg
from .models import build_model
from .training import resolve_device


def _save_grayscale(values: np.ndarray, path: Path) -> None:
    pixels = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _prepare_frame_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old_frame in path.glob("frame_*.png"):
        old_frame.unlink()


def frames_to_video(frame_dir: Path, fps: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _imageio_ffmpeg().get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def reconstruct(
    checkpoint_path: Path,
    frame_dir: Path,
    output_dir: Path,
    batch_size: int = 8,
    device_name: str = "auto",
    make_videos: bool = True,
    fps: float | None = None,
) -> dict:
    device = resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model, _ = build_model(
        checkpoint["model_name"], **checkpoint["model_kwargs"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    height, width = checkpoint["image_size"]
    dataset = FrameDataset(
        frame_dir,
        height=height,
        width=width,
        input_threshold=checkpoint["input_threshold"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probability_dir = output_dir / "probability_activations"
    binary_dir = output_dir / "binary_activations"
    attention_dir = output_dir / "attention_maps"
    _prepare_frame_directory(probability_dir)
    _prepare_frame_directory(binary_dir)
    if checkpoint["model_name"] == "attention":
        _prepare_frame_directory(attention_dir)

    frame_index = 0
    threshold = checkpoint["activation_threshold"]
    with torch.inference_mode():
        for targets, _ in loader:
            targets = targets.to(device)
            logits, extras = model(targets)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            attention = extras.get("attention")
            if attention is not None:
                attention = torch.nn.functional.interpolate(
                    attention,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).cpu().numpy()

            for batch_index in range(probabilities.shape[0]):
                name = f"frame_{frame_index:05d}.png"
                activation = probabilities[batch_index, 0]
                _save_grayscale(activation, probability_dir / name)
                binary = (activation >= threshold).astype(np.float32)
                _save_grayscale(binary, binary_dir / name)
                if attention is not None:
                    _save_grayscale(attention[batch_index, 0], attention_dir / name)
                frame_index += 1

    if fps is None and DEFAULT_MANIFEST.exists():
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        fps = float(manifest["frames"]["fps"])
    fps = fps or 30.0

    videos: dict[str, str] = {}
    if make_videos:
        probability_video = output_dir / "probability_activations.mp4"
        binary_video = output_dir / "binary_activations.mp4"
        frames_to_video(probability_dir, fps, probability_video)
        frames_to_video(binary_dir, fps, binary_video)
        videos = {
            "probability": str(probability_video.resolve()),
            "binary": str(binary_video.resolve()),
        }
        if checkpoint["model_name"] == "attention":
            attention_video = output_dir / "attention_maps.mp4"
            frames_to_video(attention_dir, fps, attention_video)
            videos["attention"] = str(attention_video.resolve())

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "model_name": checkpoint["model_name"],
        "frame_count": frame_index,
        "image_size": [height, width],
        "activation_threshold": threshold,
        "fps": fps,
        "videos": videos,
    }
    (output_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
