"""Post-hoc memory-only baseline; never trains or calls the temporal predictor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .autoregressive import load_autoencoder
from .data import FrameDataset, _imageio_ffmpeg
from .drift_rendering import _mean_binary_iou
from .silhouette import _batch_metrics, _load_model
from .training import resolve_device


def memory_only_latents(model, times: torch.Tensor) -> torch.Tensor:
    """Same two-nearest addressing as the full model, with no history or gates."""
    if model.anchor_count == 0:
        raise ValueError("Memory-only evaluation requires at least two anchors")
    _, weights, _ = model.address_memory(times)
    return torch.einsum("bm,mchw->bchw", weights, model.memory_tokens)


def load_reference(path: Path, count: int, fps: float) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != count:
        raise ValueError("Reference CSV and source frame counts differ")
    for index, row in enumerate(rows):
        if int(row["frame"]) != index or not math.isclose(
            float(row["seconds"]), index / fps, abs_tol=1e-6
        ):
            raise ValueError("Reference frame order or FPS differs")
        value = float(row["rollout_binary_error"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Reference contains invalid binary error")
    return rows


def summarize(rows: list[dict], warmup: int) -> dict:
    scored = rows[warmup:]
    if not scored:
        raise ValueError("No frames remain after the reference warmup")
    memory = np.array([row["memory_binary_error"] for row in scored])
    rollout = np.array([row["rollout_binary_error"] for row in scored])
    return {
        "scored_frames": len(scored),
        "memory_binary_error": float(memory.mean()),
        "rollout_binary_error": float(rollout.mean()),
        "memory_minus_rollout_error": float((memory - rollout).mean()),
        "temporal_error_reduction_fraction": float(1 - rollout.mean() / memory.mean())
        if memory.mean() else None,
        "fraction_temporal_better": float((rollout < memory - 1e-12).mean()),
        "fraction_memory_better": float((memory < rollout - 1e-12).mean()),
        "memory_mean_binary_iou": float(np.mean([
            row["memory_mean_binary_iou"] for row in scored
        ])),
        "memory_boundary_f1": float(np.mean([
            row["memory_boundary_f1"] for row in scored
        ])),
    }


def comparison_frame(target, prediction, index, fps, error) -> Image.Image:
    panel_width = 256
    panel_height = round(panel_width * target.shape[0] / target.shape[1])
    canvas = Image.new("RGB", (2 * panel_width, panel_height + 32), "black")
    for column, pixels in enumerate((target, prediction)):
        panel = Image.fromarray(pixels.astype(np.uint8) * 255).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        canvas.paste(panel, (column * panel_width, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), f"target  t={index / fps:.2f}s", fill="white")
    draw.text((panel_width + 5, 3), "memory only (no temporal model)", fill="white")
    draw.text((panel_width + 5, 17), f"pixel error={error:.4f}", fill="orange")
    return canvas


def evaluate(args) -> dict:
    if args.batch_size < 1 or args.fps <= 0:
        raise ValueError("Batch size and FPS must be positive")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "hybrid_v4_bleeding_memory":
        raise ValueError("This diagnostic supports V4 bleeding-memory checkpoints only")
    model = _load_model(checkpoint, device)
    if model.anchor_count == 0:
        raise ValueError("No anchors in checkpoint")
    autoencoder, ae_checkpoint = load_autoencoder(args.autoencoder_checkpoint, device)
    height, width = checkpoint["image_size"]
    if list(ae_checkpoint["image_size"]) != [height, width]:
        raise ValueError("Autoencoder image size differs from predictor checkpoint")
    dataset = FrameDataset(args.data_dir, height, width, checkpoint["input_threshold"])
    reference = load_reference(args.reference_csv, len(dataset), args.fps)
    warmup = int(checkpoint.get("rollout_warmup_frames", 16))
    reference_summary_path = args.reference_csv.parent / "drift_summary.json"
    if reference_summary_path.exists():
        ref = json.loads(reference_summary_path.read_text(encoding="utf-8"))
        if ref["frame_count"] != len(dataset) or ref["fps"] != args.fps:
            raise ValueError("Reference summary timeline differs")
        if ref["image_size"] != [height, width]:
            raise ValueError("Reference image size differs")
        if ref.get("memory_usage", {}).get("token_count") != model.anchor_count:
            raise ValueError("Reference anchor budget differs")
        warmup = int(ref["warmup_frames"])
    if not 0 <= warmup < len(dataset):
        raise ValueError("Invalid reference warmup")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    mean = checkpoint["latent_mean"].to(device)
    std = checkpoint["latent_standard_deviation"].to(device)
    rows = []
    video = None
    frame_index = 0
    next_progress = 0
    try:
        with torch.inference_mode():
            for targets, _ in loader:
                times = torch.arange(frame_index, frame_index + len(targets), device=device).float()
                times /= max(1, len(dataset) - 1)
                latent = memory_only_latents(model, times)
                probability = torch.sigmoid(autoencoder.decode(latent * std + mean, (height, width)))
                if checkpoint.get("canonicalize_polarity", False):
                    invert = (model.predict_polarity(times)[:, 0] >= 0)[:, None, None, None]
                    probability = torch.where(invert, 1 - probability, probability)
                prediction = probability >= checkpoint["activation_threshold"]
                metrics = _batch_metrics(prediction, targets.to(device).bool())
                predictions = prediction[:, 0].cpu().numpy()
                target_arrays = targets[:, 0].numpy().astype(bool)
                errors = metrics["pixel_error"].cpu().tolist()
                boundaries = metrics["boundary_f1"].cpu().tolist()
                distances = (times[:, None] - model.anchor_times[None]).abs().min(dim=1).values
                distances = (distances * (len(dataset) - 1) / args.fps).cpu().tolist()
                for position, (target, predicted) in enumerate(zip(target_arrays, predictions)):
                    full_error = float(reference[frame_index]["rollout_binary_error"])
                    row = {
                        "frame": frame_index, "seconds": frame_index / args.fps,
                        "memory_binary_error": errors[position],
                        "memory_mean_binary_iou": _mean_binary_iou(predicted, target),
                        "memory_boundary_f1": boundaries[position],
                        "nearest_anchor_seconds": distances[position],
                        "rollout_binary_error": full_error,
                        "memory_minus_rollout_error": errors[position] - full_error,
                    }
                    rows.append(row)
                    if not args.no_video:
                        canvas = comparison_frame(target, predicted, frame_index, args.fps, errors[position])
                        if video is None:
                            video = subprocess.Popen([
                                _imageio_ffmpeg().get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{canvas.width}x{canvas.height}",
                                "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                                str(args.output_dir / "comparison.mp4"),
                            ], stdin=subprocess.PIPE)
                        video.stdin.write(canvas.tobytes())
                    frame_index += 1
                if frame_index >= next_progress:
                    print(f"memory-only: {frame_index}/{len(dataset)} frames", flush=True)
                    next_progress = frame_index + 300
    finally:
        if video is not None:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("Comparison video encoding failed")
    with (args.output_dir / "error_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "experiment": "post-hoc memory-only removal diagnostic",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "autoencoder_checkpoint": str(args.autoencoder_checkpoint.resolve()),
        "reference_csv": str(args.reference_csv.resolve()),
        "frame_dir": str(args.data_dir.resolve()),
        "frame_count": len(dataset), "fps": args.fps,
        "anchor_count": model.anchor_count, "warmup_excluded_from_scores": warmup,
        "source_frames_fed_to_memory_model": 0,
        "post_cutoff": summarize(rows, warmup),
        "caveat": "Uses jointly trained anchors unchanged, not an independently trained memory-only model.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    score = result["post_cutoff"]
    report = [
        "# Memory-only versus full temporal rollout", "",
        "This is a post-hoc removal diagnostic, with no training or weight changes.",
        "Time selects the same two nearest learned anchors as the full model. Their weighted latent is decoded directly, using the same normalization, decoder, activation threshold and learned polarity head.",
        "No source history, temporal U-Net, predicted velocity or memory gates contribute to the memory-only output.", "",
        f"Anchor count: {model.anchor_count}. Scored frames: {score['scored_frames']} (first {warmup} excluded to match the reference).", "",
        "| Metric | Value |", "| --- | ---: |",
    ]
    report += [f"| {key} | {value:.6f} |" for key, value in score.items() if value is not None]
    report += ["", "## Limitations", "", result["caveat"],
               "A win for the temporal model shows contribution in this trained system, not that temporal models outperform all independently optimized interpolation baselines.",
               "The reference rollout is read from the supplied CSV, not rerun. Verify its source and checkpoint provenance before using this as a research result.", "",
               "## Provenance", "", "```json", json.dumps(result, indent=2), "```", ""]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, default=Path("prototype_runs/basic_full/model_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("prototype_data/full_source_frames"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-video", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
