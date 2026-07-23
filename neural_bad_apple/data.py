from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_DIR = PROJECT_ROOT / "prototype_data" / "source_frames"
DEFAULT_MANIFEST = PROJECT_ROOT / "prototype_data" / "manifest.json"


def _imageio_ffmpeg():
    """Import the small vendored FFmpeg wrapper when it is available."""
    vendor_dir = PROJECT_ROOT / ".vendor"
    if vendor_dir.exists() and str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Video support is missing. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return imageio_ffmpeg


def find_source_video(project_root: Path = PROJECT_ROOT) -> Path:
    videos = sorted(project_root.glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(
            f"Expected exactly one MP4 in {project_root}, found {len(videos)}. "
            "Pass --input explicitly."
        )
    return videos[0]


def probe_video(video_path: Path) -> dict[str, Any]:
    reader = _imageio_ffmpeg().read_frames(str(video_path), pix_fmt="gray")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    return metadata


def list_frames(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("frame_*.png"))


def extract_segment(
    video_path: Path,
    output_dir: Path = DEFAULT_FRAME_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    start_seconds: float = 45.0,
    end_seconds: float = 60.0,
    fps: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Extract a reproducible source segment as lossless grayscale PNG files."""
    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    manifest_path = manifest_path.resolve()

    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")

    source_metadata = probe_video(video_path)
    output_fps = float(fps or source_metadata["fps"])
    duration = end_seconds - start_seconds
    expected_frames = round(duration * output_fps)

    existing = list_frames(output_dir) if output_dir.exists() else []
    if existing and not force:
        if len(existing) == expected_frames and manifest_path.exists():
            existing_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            existing_segment = existing_manifest.get("segment", {})
            existing_frames = existing_manifest.get("frames", {})
            same_request = (
                existing_manifest.get("source_video") == str(video_path)
                and existing_segment.get("start_seconds") == start_seconds
                and existing_segment.get("end_seconds") == end_seconds
                and existing_frames.get("fps") == output_fps
                and existing_frames.get("count") == expected_frames
            )
            if same_request:
                return existing_manifest
        raise RuntimeError(
            f"{output_dir} already contains frames from a different or partial "
            "extraction. Use --force to replace generated frames."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        for frame_path in list_frames(output_dir):
            frame_path.unlink()

    ffmpeg_exe = _imageio_ffmpeg().get_ffmpeg_exe()
    output_pattern = output_dir / "frame_%05d.png"
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-t",
        str(duration),
        "-i",
        str(video_path),
        "-an",
        "-vf",
        f"fps={output_fps}",
        "-pix_fmt",
        "gray",
        "-start_number",
        "0",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)

    frames = list_frames(output_dir)
    if len(frames) != expected_frames:
        raise RuntimeError(
            f"FFmpeg extracted {len(frames)} frames; expected {expected_frames}."
        )

    manifest = {
        "project": "Neural network dreams Bad Apple",
        "source_video": str(video_path),
        "source_metadata": source_metadata,
        "segment": {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": duration,
        },
        "frames": {
            "directory": str(output_dir),
            "pattern": "frame_%05d.png",
            "count": len(frames),
            "fps": output_fps,
            "size": source_metadata["size"],
            "color_mode": "grayscale",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class FrameDataset:
    """Loads frames and turns each source pixel into a binary training target."""

    def __init__(
        self,
        frame_dir: Path,
        height: int,
        width: int,
        input_threshold: float = 0.5,
    ) -> None:
        self.paths = list_frames(frame_dir)
        if not self.paths:
            raise RuntimeError(f"No frame_*.png files found in {frame_dir}")
        self.height = height
        self.width = width
        self.input_threshold = input_threshold

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch

        with Image.open(self.paths[index]) as image:
            image = image.convert("L")
            image = image.resize(
                (self.width, self.height), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        binary = (pixels >= self.input_threshold).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0), self.paths[index].name
