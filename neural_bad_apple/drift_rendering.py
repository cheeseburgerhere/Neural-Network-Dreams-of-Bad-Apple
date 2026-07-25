from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .autoregressive import (
    LatentAutoregressor,
    encode_sequence,
    load_autoencoder,
    rollout_latents,
)
from .data import DEFAULT_MANIFEST, FrameDataset
from .hybrid import (
    HybridTemporalMemoryModel,
    encode_canonical_sequence,
    rollout_hybrid_latents,
)
from .hybrid_v4 import BleedingSceneMemoryModel
from .rendering import _prepare_frame_directory, frames_to_video
from .training import resolve_device


def _save_grayscale(values: np.ndarray, path: Path) -> None:
    pixels = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _error_image(target: np.ndarray, prediction: np.ndarray) -> Image.Image:
    false_positive = prediction & ~target
    false_negative = target & ~prediction
    pixels = np.zeros((*target.shape, 3), dtype=np.uint8)
    pixels[false_positive] = (255, 72, 72)
    pixels[false_negative] = (64, 190, 255)
    return Image.fromarray(pixels, mode="RGB")


def _binary_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(values.astype(np.uint8) * 255, mode="L")


def _comparison_image(
    target: np.ndarray,
    teacher: np.ndarray,
    rollout: np.ndarray,
    error: Image.Image,
    frame_index: int,
    seconds: float,
    rollout_error: float,
    accumulation_gap: float,
    panel_width: int = 256,
) -> Image.Image:
    panel_height = round(panel_width * target.shape[0] / target.shape[1])
    header_height = 34
    canvas = Image.new(
        "RGB", (panel_width * 4, panel_height + header_height), color="black"
    )
    draw = ImageDraw.Draw(canvas)
    panels = [
        _binary_image(target).convert("RGB"),
        _binary_image(teacher).convert("RGB"),
        _binary_image(rollout).convert("RGB"),
        error,
    ]
    labels = ("target", "teacher-forced", "free rollout", "error map")
    for panel_index, (panel, label) in enumerate(zip(panels, labels)):
        x = panel_index * panel_width
        panel = panel.resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        canvas.paste(panel, (x, header_height))
        draw.text((x + 5, 4), label, fill="white")

    draw.text(
        (5, 18),
        f"frame {frame_index:04d}  t={seconds:05.2f}s",
        fill=(180, 180, 180),
    )
    draw.text(
        (panel_width * 2 + 5, 18),
        f"error={rollout_error:.4f}",
        fill=(255, 170, 80),
    )
    draw.text(
        (panel_width * 3 + 5, 18),
        f"accumulation={accumulation_gap:+.4f}",
        fill=(255, 220, 90),
    )
    return canvas


def _mean_binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    scores: list[float] = []
    for value in (False, True):
        prediction_class = prediction == value
        target_class = target == value
        union = np.logical_or(prediction_class, target_class).sum()
        if union:
            intersection = np.logical_and(
                prediction_class, target_class
            ).sum()
            scores.append(float(intersection / union))
    return float(np.mean(scores))


def _teacher_forced_latents(
    model: LatentAutoregressor,
    true_latents: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    predictions = [true_latents[:1]]
    hidden = None
    with torch.inference_mode():
        for frame_index in range(len(true_latents) - 1):
            predicted, hidden = model.step(
                true_latents[frame_index : frame_index + 1].to(device),
                hidden,
            )
            predictions.append(predicted.cpu())
    return torch.cat(predictions, dim=0)


def _teacher_forced_hybrid_latents(
    model: HybridTemporalMemoryModel,
    true_latents: torch.Tensor,
    device: torch.device,
    history_length: int,
) -> torch.Tensor:
    predictions = [true_latents[:history_length]]
    with torch.inference_mode():
        for target_index in range(history_length, len(true_latents)):
            latent_history = true_latents[
                target_index - history_length : target_index
            ].unsqueeze(0).to(device)
            normalized_time = torch.tensor(
                [target_index / max(1, len(true_latents) - 1)],
                device=device,
                dtype=latent_history.dtype,
            )
            predicted, _ = model(latent_history, normalized_time)
            predictions.append(predicted.cpu())
    return torch.cat(predictions, dim=0)


def _hybrid_internal_metrics(
    model: torch.nn.Module,
    latents: torch.Tensor,
    device: torch.device,
    history_length: int,
) -> dict[str, np.ndarray]:
    values = {
        "effective_memory_gate": np.zeros(len(latents), dtype=np.float32),
        "motion_mask": np.zeros(len(latents), dtype=np.float32),
        "predicted_velocity_magnitude": np.zeros(
            len(latents), dtype=np.float32
        ),
        "slow_velocity_magnitude": np.zeros(
            len(latents), dtype=np.float32
        ),
        "fast_velocity_magnitude": np.zeros(
            len(latents), dtype=np.float32
        ),
    }
    with torch.inference_mode():
        for target_index in range(history_length, len(latents)):
            history = latents[
                target_index - history_length : target_index
            ].unsqueeze(0).to(device)
            normalized_time = torch.tensor(
                [target_index / max(1, len(latents) - 1)],
                device=device,
                dtype=history.dtype,
            )
            _, extras = model(history, normalized_time)
            if "spatial_memory_gate" in extras:
                values["effective_memory_gate"][target_index] = (
                    extras["spatial_memory_gate"].mean().item()
                )
            if "motion_mask" in extras:
                values["motion_mask"][target_index] = (
                    extras["motion_mask"].mean().item()
                )
            if "predicted_velocity" in extras:
                values["predicted_velocity_magnitude"][target_index] = (
                    extras["predicted_velocity"].abs().mean().item()
                )
            if "slow_velocity" in extras:
                values["slow_velocity_magnitude"][target_index] = (
                    extras["slow_velocity"].abs().mean().item()
                )
            if "fast_velocity" in extras:
                values["fast_velocity_magnitude"][target_index] = (
                    extras["fast_velocity"].abs().mean().item()
                )
    return values


def _draw_error_curve(
    metrics: list[dict], output_path: Path, warmup_frames: int
) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 75, 25, 55, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), color=(16, 18, 22))
    draw = ImageDraw.Draw(image)

    teacher = [row["teacher_binary_error"] for row in metrics]
    rollout = [row["rollout_binary_error"] for row in metrics]
    gap = [max(0.0, row["accumulation_gap"]) for row in metrics]
    y_max = max(max(rollout), max(teacher), 0.01) * 1.08

    for grid_index in range(5):
        fraction = grid_index / 4
        y = top + round((1.0 - fraction) * plot_height)
        draw.line((left, y, width - right, y), fill=(48, 52, 60), width=1)
        draw.text(
            (8, y - 7), f"{fraction * y_max:.3f}", fill=(175, 180, 190)
        )

    draw.line(
        (left, top, left, height - bottom), fill=(205, 210, 220), width=2
    )
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill=(205, 210, 220),
        width=2,
    )

    def points(values: list[float]) -> list[tuple[int, int]]:
        denominator = max(1, len(values) - 1)
        return [
            (
                left + round(index / denominator * plot_width),
                top + round((1.0 - value / y_max) * plot_height),
            )
            for index, value in enumerate(values)
        ]

    draw.line(points(teacher), fill=(75, 170, 255), width=3)
    draw.line(points(rollout), fill=(255, 125, 55), width=3)
    draw.line(points(gap), fill=(255, 220, 75), width=2)
    cutoff_x = left + round(
        warmup_frames / max(1, len(metrics) - 1) * plot_width
    )
    draw.line(
        (cutoff_x, top, cutoff_x, height - bottom),
        fill=(180, 100, 255),
        width=2,
    )
    draw.text(
        (cutoff_x + 4, top + 5), "source cutoff", fill=(200, 150, 255)
    )
    draw.text(
        (left, 16),
        "Autoregressive error accumulation",
        fill="white",
    )
    draw.text(
        (left + 300, 18),
        "teacher-forced",
        fill=(75, 170, 255),
    )
    draw.text(
        (left + 440, 18),
        "free rollout",
        fill=(255, 125, 55),
    )
    draw.text(
        (left + 545, 18),
        "accumulation gap",
        fill=(255, 220, 75),
    )
    draw.text(
        (width // 2 - 45, height - 28), "frame index", fill=(190, 195, 205)
    )
    image.save(output_path)


def _load_fps(fps: float | None) -> float:
    if fps is not None:
        return fps
    if DEFAULT_MANIFEST.exists():
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        return float(manifest["frames"]["fps"])
    return 30.0


def render_drift(
    checkpoint_path: Path,
    frame_dir: Path,
    output_dir: Path,
    device_name: str = "auto",
    autoencoder_checkpoint_path: Path | None = None,
    batch_size: int = 8,
    fps: float | None = None,
    make_videos: bool = True,
    warmup_frames: int | None = None,
) -> dict:
    device = resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model_type = checkpoint.get("model_type", "latent_autoregressor")
    if model_type == "latent_autoregressor":
        model = LatentAutoregressor(**checkpoint["model_kwargs"])
    elif model_type == "hybrid_temporal_memory":
        model = HybridTemporalMemoryModel(**checkpoint["model_kwargs"])
    elif model_type == "hybrid_v4_bleeding_memory":
        model = BleedingSceneMemoryModel(**checkpoint["model_kwargs"])
    else:
        raise ValueError(f"Unsupported drift model type: {model_type}")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    autoencoder_path = autoencoder_checkpoint_path or Path(
        checkpoint["autoencoder_checkpoint"]
    )
    autoencoder, autoencoder_checkpoint = load_autoencoder(
        autoencoder_path, device
    )
    canonicalize_polarity = checkpoint.get(
        "canonicalize_polarity", False
    )
    polarity_tracking_method = checkpoint.get(
        "polarity_tracking_method", "border"
    )
    polarity_switch_penalty = checkpoint.get(
        "polarity_switch_penalty", 0.05
    )
    if canonicalize_polarity:
        true_raw_latents, source_polarities = encode_canonical_sequence(
            autoencoder,
            autoencoder_checkpoint,
            frame_dir,
            device,
            batch_size=batch_size,
            polarity_tracking_method=polarity_tracking_method,
            polarity_switch_penalty=polarity_switch_penalty,
        )
    else:
        true_raw_latents = encode_sequence(
            autoencoder,
            autoencoder_checkpoint,
            frame_dir,
            device,
            batch_size=batch_size,
        )
        source_polarities = torch.zeros(
            len(true_raw_latents), dtype=torch.float32
        )
    latent_mean = checkpoint["latent_mean"].cpu()
    latent_standard_deviation = checkpoint[
        "latent_standard_deviation"
    ].cpu()
    true_latents = (
        true_raw_latents - latent_mean
    ) / latent_standard_deviation

    warmup_frames = warmup_frames or checkpoint.get(
        "rollout_warmup_frames", 1
    )
    warmup_frames = min(max(1, warmup_frames), len(true_latents) - 1)
    hybrid_model_types = {
        "hybrid_temporal_memory",
        "hybrid_v4_bleeding_memory",
    }
    if model_type in hybrid_model_types:
        warmup_frames = max(4, warmup_frames)
        teacher_latents = _teacher_forced_hybrid_latents(
            model, true_latents, device, warmup_frames
        )
        rollout_latent_sequence = rollout_hybrid_latents(
            model,
            true_latents[:warmup_frames].to(device),
            len(true_latents),
        ).cpu()
    else:
        teacher_latents = _teacher_forced_latents(
            model, true_latents, device
        )
        rollout_latent_sequence = rollout_latents(
            model,
            true_latents[:warmup_frames].to(device),
            len(true_latents),
        ).cpu()

    predicted_polarity_probabilities = torch.zeros(len(true_latents))
    if canonicalize_polarity:
        normalized_times = torch.linspace(
            0.0, 1.0, steps=len(true_latents), device=device
        )
        with torch.inference_mode():
            predicted_polarity_probabilities = torch.sigmoid(
                model.predict_polarity(normalized_times)
            )[:, 0].cpu()
    rendered_polarities = predicted_polarity_probabilities >= 0.5
    if canonicalize_polarity:
        rendered_polarities[:warmup_frames] = (
            source_polarities[:warmup_frames] >= 0.5
        )

    height, width = checkpoint["image_size"]
    dataset = FrameDataset(
        frame_dir,
        height=height,
        width=width,
        input_threshold=checkpoint["input_threshold"],
    )
    target_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    teacher_dir = output_dir / "teacher_forced"
    rollout_dir = output_dir / "free_rollout"
    error_dir = output_dir / "error_maps"
    comparison_dir = output_dir / "comparison"
    for directory in (
        teacher_dir,
        rollout_dir,
        error_dir,
        comparison_dir,
    ):
        _prepare_frame_directory(directory)

    threshold = checkpoint["activation_threshold"]
    metrics: list[dict] = []
    frame_index = 0
    frame_rate = _load_fps(fps)
    with torch.inference_mode():
        for targets, _ in target_loader:
            current_batch = len(targets)
            frame_slice = slice(frame_index, frame_index + current_batch)
            teacher_raw = (
                teacher_latents[frame_slice]
                * latent_standard_deviation
                + latent_mean
            ).to(device)
            rollout_raw = (
                rollout_latent_sequence[frame_slice]
                * latent_standard_deviation
                + latent_mean
            ).to(device)
            teacher_probability_tensor = torch.sigmoid(
                autoencoder.decode(teacher_raw, (height, width))
            ).cpu()
            rollout_probability_tensor = torch.sigmoid(
                autoencoder.decode(rollout_raw, (height, width))
            ).cpu()
            if canonicalize_polarity:
                invert = rendered_polarities[frame_slice][
                    :, None, None, None
                ]
                teacher_probability_tensor = torch.where(
                    invert,
                    1.0 - teacher_probability_tensor,
                    teacher_probability_tensor,
                )
                rollout_probability_tensor = torch.where(
                    invert,
                    1.0 - rollout_probability_tensor,
                    rollout_probability_tensor,
                )
            teacher_probabilities = teacher_probability_tensor.numpy()
            rollout_probabilities = rollout_probability_tensor.numpy()
            target_values = targets.numpy()

            for batch_index in range(current_batch):
                target = target_values[batch_index, 0] >= 0.5
                teacher_probability = teacher_probabilities[batch_index, 0]
                rollout_probability = rollout_probabilities[batch_index, 0]
                teacher_binary = teacher_probability >= threshold
                rollout_binary = rollout_probability >= threshold

                teacher_error = float(
                    np.not_equal(teacher_binary, target).mean()
                )
                rollout_error = float(
                    np.not_equal(rollout_binary, target).mean()
                )
                accumulation_gap = rollout_error - teacher_error
                latent_teacher_mse = float(
                    (
                        teacher_latents[frame_index]
                        - true_latents[frame_index]
                    )
                    .square()
                    .mean()
                )
                latent_rollout_mse = float(
                    (
                        rollout_latent_sequence[frame_index]
                        - true_latents[frame_index]
                    )
                    .square()
                    .mean()
                )
                row = {
                    "frame": frame_index,
                    "seconds": frame_index / frame_rate,
                    "teacher_mae": float(
                        np.abs(teacher_probability - target).mean()
                    ),
                    "rollout_mae": float(
                        np.abs(rollout_probability - target).mean()
                    ),
                    "teacher_binary_error": teacher_error,
                    "rollout_binary_error": rollout_error,
                    "accumulation_gap": accumulation_gap,
                    "rollout_mean_binary_iou": _mean_binary_iou(
                        rollout_binary, target
                    ),
                    "teacher_latent_mse": latent_teacher_mse,
                    "rollout_latent_mse": latent_rollout_mse,
                    "target_polarity": int(
                        source_polarities[frame_index].item()
                    ),
                    "predicted_polarity_probability": float(
                        predicted_polarity_probabilities[
                            frame_index
                        ].item()
                    ),
                }
                metrics.append(row)

                name = f"frame_{frame_index:05d}.png"
                _save_grayscale(
                    teacher_binary.astype(np.float32), teacher_dir / name
                )
                _save_grayscale(
                    rollout_binary.astype(np.float32), rollout_dir / name
                )
                error = _error_image(target, rollout_binary)
                error.save(error_dir / name)
                comparison = _comparison_image(
                    target,
                    teacher_binary,
                    rollout_binary,
                    error,
                    frame_index,
                    frame_index / frame_rate,
                    rollout_error,
                    accumulation_gap,
                )
                comparison.save(comparison_dir / name)
                frame_index += 1

    running_total = 0.0
    for row_index, row in enumerate(metrics):
        running_total += row["rollout_binary_error"]
        row["cumulative_rollout_binary_error"] = running_total / (
            row_index + 1
        )

    memory_summary = None
    if model_type in hybrid_model_types:
        normalized_times = torch.linspace(
            0.0, 1.0, steps=len(metrics), device=device
        )
        with torch.inference_mode():
            _, memory_weights, memory_gates = model.address_memory(
                normalized_times
            )
        memory_weights_array = memory_weights.cpu().numpy()
        memory_gates_array = memory_gates[:, 0].cpu().numpy()
        entropies = -np.sum(
            memory_weights_array
            * np.log(np.clip(memory_weights_array, 1e-8, 1.0)),
            axis=1,
        ) / np.log(memory_weights_array.shape[1])
        dominant_tokens = memory_weights_array.argmax(axis=1)
        for row_index, row in enumerate(metrics):
            row["memory_gate"] = float(memory_gates_array[row_index])
            row["memory_entropy"] = float(entropies[row_index])
            row["dominant_memory_token"] = int(
                dominant_tokens[row_index]
            )
            for token_index, weight in enumerate(
                memory_weights_array[row_index]
            ):
                row[f"memory_weight_{token_index:02d}"] = float(weight)
        memory_summary = {
            "token_count": int(memory_weights_array.shape[1]),
            "mean_gate": float(memory_gates_array.mean()),
            "post_cutoff_mean_gate": float(
                memory_gates_array[warmup_frames:].mean()
            ),
            "minimum_gate": float(memory_gates_array.min()),
            "maximum_gate": float(memory_gates_array.max()),
            "mean_address_entropy": float(entropies.mean()),
            "dominant_token_changes": int(
                np.count_nonzero(np.diff(dominant_tokens))
            ),
        }
        if model_type == "hybrid_v4_bleeding_memory":
            teacher_internal = _hybrid_internal_metrics(
                model,
                true_latents,
                device,
                warmup_frames,
            )
            rollout_internal = _hybrid_internal_metrics(
                model,
                rollout_latent_sequence,
                device,
                warmup_frames,
            )
            for row_index, row in enumerate(metrics):
                for name, values in teacher_internal.items():
                    row[f"teacher_{name}"] = float(values[row_index])
                for name, values in rollout_internal.items():
                    row[f"rollout_{name}"] = float(values[row_index])
            memory_summary.update(
                {
                    "post_cutoff_teacher_effective_gate": float(
                        teacher_internal["effective_memory_gate"][
                            warmup_frames:
                        ].mean()
                    ),
                    "post_cutoff_rollout_effective_gate": float(
                        rollout_internal["effective_memory_gate"][
                            warmup_frames:
                        ].mean()
                    ),
                    "post_cutoff_teacher_motion_mask": float(
                        teacher_internal["motion_mask"][
                            warmup_frames:
                        ].mean()
                    ),
                    "post_cutoff_rollout_motion_mask": float(
                        rollout_internal["motion_mask"][
                            warmup_frames:
                        ].mean()
                    ),
                    "post_cutoff_rollout_slow_velocity": float(
                        rollout_internal["slow_velocity_magnitude"][
                            warmup_frames:
                        ].mean()
                    ),
                    "post_cutoff_rollout_fast_velocity": float(
                        rollout_internal["fast_velocity_magnitude"][
                            warmup_frames:
                        ].mean()
                    ),
                }
            )

    csv_path = output_dir / "error_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    _draw_error_curve(
        metrics, output_dir / "error_curve.png", warmup_frames
    )

    post_cutoff_metrics = metrics[warmup_frames:]
    peak_row = max(
        post_cutoff_metrics,
        key=lambda row: row["rollout_binary_error"],
    )
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "model_type": model_type,
        "autoencoder_checkpoint": str(autoencoder_path.resolve()),
        "frame_count": len(metrics),
        "fps": frame_rate,
        "warmup_frames": warmup_frames,
        "source_cutoff_seconds": (warmup_frames - 1) / frame_rate,
        "image_size": [height, width],
        "canonicalize_polarity": canonicalize_polarity,
        "polarity_tracking_method": polarity_tracking_method,
        "polarity_switch_penalty": polarity_switch_penalty,
        "mean_teacher_binary_error": float(
            np.mean([row["teacher_binary_error"] for row in metrics])
        ),
        "mean_rollout_binary_error": float(
            np.mean([row["rollout_binary_error"] for row in metrics])
        ),
        "mean_accumulation_gap": float(
            np.mean([row["accumulation_gap"] for row in metrics])
        ),
        "final_rollout_binary_error": metrics[-1][
            "rollout_binary_error"
        ],
        "final_accumulation_gap": metrics[-1]["accumulation_gap"],
        "peak_error_frame": peak_row["frame"],
        "peak_error_seconds": peak_row["seconds"],
        "peak_rollout_binary_error": peak_row["rollout_binary_error"],
        "mean_rollout_iou": float(
            np.mean([row["rollout_mean_binary_iou"] for row in metrics])
        ),
        "post_cutoff_mean_teacher_binary_error": float(
            np.mean(
                [
                    row["teacher_binary_error"]
                    for row in post_cutoff_metrics
                ]
            )
        ),
        "post_cutoff_mean_rollout_binary_error": float(
            np.mean(
                [
                    row["rollout_binary_error"]
                    for row in post_cutoff_metrics
                ]
            )
        ),
        "post_cutoff_mean_accumulation_gap": float(
            np.mean(
                [row["accumulation_gap"] for row in post_cutoff_metrics]
            )
        ),
        "post_cutoff_mean_rollout_iou": float(
            np.mean(
                [
                    row["rollout_mean_binary_iou"]
                    for row in post_cutoff_metrics
                ]
            )
        ),
        "error_curve_csv": str(csv_path.resolve()),
        "error_curve_image": str(
            (output_dir / "error_curve.png").resolve()
        ),
        "error_map_legend": {
            "red": "false positive: dreamed white where target is black",
            "cyan": "false negative: missed a white target pixel",
        },
    }
    if memory_summary is not None:
        summary["memory_usage"] = memory_summary
    if canonicalize_polarity:
        predicted_polarities = (
            predicted_polarity_probabilities >= 0.5
        )
        summary["polarity_accuracy"] = float(
            (
                predicted_polarities
                == (source_polarities >= 0.5)
            )
            .float()
            .mean()
        )
        summary["target_polarity_switches"] = int(
            (source_polarities[1:] != source_polarities[:-1]).sum().item()
        )
        summary["predicted_polarity_switches"] = int(
            (
                predicted_polarities[1:]
                != predicted_polarities[:-1]
            )
            .sum()
            .item()
        )

    if make_videos:
        videos = {
            "teacher_forced": output_dir / "teacher_forced.mp4",
            "free_rollout": output_dir / "free_rollout.mp4",
            "error_maps": output_dir / "error_maps.mp4",
            "comparison": output_dir / "comparison.mp4",
        }
        frames_to_video(teacher_dir, frame_rate, videos["teacher_forced"])
        frames_to_video(rollout_dir, frame_rate, videos["free_rollout"])
        frames_to_video(error_dir, frame_rate, videos["error_maps"])
        frames_to_video(comparison_dir, frame_rate, videos["comparison"])
        summary["videos"] = {
            name: str(path.resolve()) for name, path in videos.items()
        }

    (output_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
