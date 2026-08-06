from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .autoregressive import load_autoencoder
from .data import FrameDataset
from .hybrid import encode_canonical_sequence
from .hybrid_v4 import BleedingSceneMemoryModel
from .training import resolve_device


@dataclass(frozen=True)
class SilhouetteVariant:
    name: str
    fast_scale: float = 1.0
    moving_bleed: float = 0.0
    disagreement_recovery: float = 0.0


VARIANTS = {
    "baseline": SilhouetteVariant("baseline"),
    "memory-only": SilhouetteVariant("memory-only"),
    "fast-1.5": SilhouetteVariant("fast-1.5", fast_scale=1.5),
    "fast-2.0": SilhouetteVariant("fast-2.0", fast_scale=2.0),
    "moving-0.5": SilhouetteVariant("moving-0.5", moving_bleed=0.5),
    "moving-1.0": SilhouetteVariant("moving-1.0", moving_bleed=1.0),
    "recovery-0.25": SilhouetteVariant(
        "recovery-0.25", disagreement_recovery=0.25
    ),
    "recovery-0.50": SilhouetteVariant(
        "recovery-0.50", disagreement_recovery=0.5
    ),
    "fast-1.5-moving-0.5": SilhouetteVariant(
        "fast-1.5-moving-0.5", fast_scale=1.5, moving_bleed=0.5
    ),
}


@dataclass
class SilhouetteDiagnosticConfig:
    checkpoint: Path
    frame_dir: Path
    output_dir: Path
    cache_path: Path
    variant: str = "baseline"
    sample_stride: int = 15
    focus_start_seconds: float = 53.0
    focus_end_seconds: float = 55.0
    fps: float = 30.0
    batch_size: int = 16
    device: str = "auto"


def _load_model(
    checkpoint: dict, device: torch.device
) -> BleedingSceneMemoryModel:
    model = BleedingSceneMemoryModel(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_or_create_cache(
    config: SilhouetteDiagnosticConfig,
    checkpoint: dict,
    autoencoder: torch.nn.Module,
    autoencoder_checkpoint: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.cache_path.exists():
        cache = torch.load(
            config.cache_path, map_location="cpu", weights_only=False
        )
        if cache.get("frame_count") != len(
            FrameDataset(
                config.frame_dir,
                *autoencoder_checkpoint["image_size"],
                input_threshold=checkpoint["input_threshold"],
            )
        ):
            raise ValueError("silhouette cache frame count is stale")
        return (
            cache["normalized_latents"].float(),
            cache["polarities"].float(),
        )

    raw_latents, polarities = encode_canonical_sequence(
        autoencoder,
        autoencoder_checkpoint,
        config.frame_dir,
        device,
        batch_size=config.batch_size,
        polarity_tracking_method=checkpoint["polarity_tracking_method"],
        polarity_switch_penalty=checkpoint["polarity_switch_penalty"],
    )
    normalized_latents = (
        raw_latents - checkpoint["latent_mean"].cpu()
    ) / checkpoint["latent_standard_deviation"].cpu()
    config.cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint": str(config.checkpoint.resolve()),
            "frame_dir": str(config.frame_dir.resolve()),
            "frame_count": len(normalized_latents),
            "normalized_latents": normalized_latents.half(),
            "polarities": polarities.to(torch.uint8),
        },
        config.cache_path,
    )
    return normalized_latents, polarities


def _variant_prediction(
    model: BleedingSceneMemoryModel,
    history: torch.Tensor,
    normalized_time: torch.Tensor,
    variant: SilhouetteVariant,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction, extras = model(history, normalized_time)
    if variant == VARIANTS["baseline"]:
        return prediction, extras

    motion_candidate = (
        history[:, -1]
        + extras["slow_velocity"]
        + variant.fast_scale * extras["fast_velocity"]
    )
    gate = extras["spatial_memory_gate"]
    remaining_gate = (model.maximum_transition_gate - gate).clamp_min(0.0)
    if variant.moving_bleed:
        gate = gate + (
            variant.moving_bleed
            * extras["motion_mask"]
            * remaining_gate
        )
        remaining_gate = (
            model.maximum_transition_gate - gate
        ).clamp_min(0.0)
    if variant.disagreement_recovery:
        recovery_signal = torch.sigmoid(
            (extras["anchor_disagreement"] - 0.35) / 0.10
        )
        gate = gate + (
            variant.disagreement_recovery
            * recovery_signal
            * remaining_gate
        )
    prediction = motion_candidate + gate * (
        extras["memory_candidate"] - motion_candidate
    )
    extras = {**extras, "diagnostic_gate": gate}
    return prediction, extras


def _sample_indices(config: SilhouetteDiagnosticConfig, count: int) -> list[int]:
    warmup = 16
    sampled = set(range(warmup, count, config.sample_stride))
    focus_start = max(warmup, round(config.focus_start_seconds * config.fps))
    focus_end = min(count, round(config.focus_end_seconds * config.fps) + 1)
    sampled.update(range(focus_start, focus_end))
    return sorted(sampled)


def _rollout_samples(
    model: BleedingSceneMemoryModel,
    latents: torch.Tensor,
    sample_indices: list[int],
    variant: SilhouetteVariant,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if variant.name == "memory-only":
        times = torch.tensor(
            [index / max(1, len(latents) - 1) for index in sample_indices],
            device=device,
        )
        with torch.inference_mode():
            _, weights, _ = model.address_memory(times)
            samples = torch.einsum(
                "bm,mchw->bchw", weights, model.memory_tokens
            )
        return samples.cpu(), {"mean_gate": 1.0}

    wanted = set(sample_indices)
    samples: list[torch.Tensor] = []
    gate_total = 0.0
    motion_total = 0.0
    fast_total = 0.0
    cut_total = 0.0
    count = 0
    history = latents[:16].unsqueeze(0).to(device)
    with torch.inference_mode():
        for target_index in range(16, len(latents)):
            normalized_time = torch.tensor(
                [target_index / max(1, len(latents) - 1)],
                device=device,
                dtype=history.dtype,
            )
            prediction, extras = _variant_prediction(
                model, history, normalized_time, variant
            )
            history = torch.cat(
                (history[:, 1:], prediction.unsqueeze(1)), dim=1
            )
            gate = extras.get(
                "diagnostic_gate", extras["spatial_memory_gate"]
            )
            gate_total += gate.mean().item()
            motion_total += extras["motion_mask"].mean().item()
            fast_total += extras["fast_velocity"].abs().mean().item()
            cut_total += extras["cut_gate"].mean().item()
            count += 1
            if target_index in wanted:
                samples.append(prediction.cpu())
    return torch.cat(samples, dim=0), {
        "mean_gate": gate_total / count,
        "mean_motion_mask": motion_total / count,
        "mean_fast_velocity": fast_total / count,
        "mean_cut_gate": cut_total / count,
    }


def _binary_edges(values: torch.Tensor) -> torch.Tensor:
    floats = values.float()
    maximum = F.max_pool2d(floats, kernel_size=3, stride=1, padding=1)
    minimum = -F.max_pool2d(
        -floats, kernel_size=3, stride=1, padding=1
    )
    return maximum != minimum


def _batch_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    pixel_error = (predictions != targets).float().mean(dim=(1, 2, 3))
    prediction_edges = _binary_edges(predictions)
    target_edges = _binary_edges(targets)
    prediction_tolerance = F.max_pool2d(
        prediction_edges.float(), kernel_size=5, stride=1, padding=2
    ).bool()
    target_tolerance = F.max_pool2d(
        target_edges.float(), kernel_size=5, stride=1, padding=2
    ).bool()
    precision = (
        (prediction_edges & target_tolerance)
        .float()
        .sum(dim=(1, 2, 3))
        / prediction_edges.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    )
    recall = (
        (target_edges & prediction_tolerance)
        .float()
        .sum(dim=(1, 2, 3))
        / target_edges.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    )
    boundary_f1 = 2.0 * precision * recall / (
        precision + recall
    ).clamp_min(1e-6)
    area_bias = predictions.float().mean(dim=(1, 2, 3)) - targets.float().mean(
        dim=(1, 2, 3)
    )
    return {
        "pixel_error": pixel_error,
        "boundary_f1": boundary_f1,
        "area_bias": area_bias,
    }


def _decode_and_measure(
    autoencoder: torch.nn.Module,
    checkpoint: dict,
    sampled_latents: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, list[float]]:
    results = {"pixel_error": [], "boundary_f1": [], "area_bias": []}
    mean = checkpoint["latent_mean"].cpu()
    standard_deviation = checkpoint["latent_standard_deviation"].cpu()
    height, width = checkpoint["image_size"]
    threshold = checkpoint["activation_threshold"]
    with torch.inference_mode():
        for start in range(0, len(sampled_latents), batch_size):
            end = start + batch_size
            raw_latents = (
                sampled_latents[start:end] * standard_deviation + mean
            ).to(device)
            probabilities = torch.sigmoid(
                autoencoder.decode(raw_latents, (height, width))
            )
            predictions = probabilities >= threshold
            metrics = _batch_metrics(
                predictions, targets[start:end].to(device)
            )
            for name, values in metrics.items():
                results[name].extend(values.cpu().tolist())
    return results


def _summarize(
    values: dict[str, list[float]], positions: list[int]
) -> dict[str, float]:
    indices = torch.tensor(positions, dtype=torch.long)
    summary: dict[str, float] = {}
    for name, items in values.items():
        selected = torch.tensor(items)[indices]
        summary[f"mean_{name}"] = float(selected.mean().item())
    return summary


def _write_report(output_dir: Path) -> None:
    rows = []
    for result_path in sorted(output_dir.glob("*.json")):
        if result_path.name in {
            "summary.json",
            "target_mismatch.json",
            "oracle_state_correction.json",
        }:
            continue
        rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    if not rows:
        return
    rows.sort(key=lambda row: row["sample"]["mean_pixel_error"])
    summary = {"variants": rows, "best_tested_variant": rows[0]["variant"]}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# V4.2 rollout silhouette diagnostics",
        "",
        "**Status:** Greedy diagnosis complete",
        "",
        "## Main finding",
        "",
        "Teacher-good/rollout-bad behavior comes from a training-target "
        "mismatch. During free-running burn-in and rollout, the model state "
        "is its own imperfect prediction, but velocity supervision remains "
        "`target - true_previous`. Correct recovery requires "
        "`target - predicted_previous`.",
        "",
        "The latent loss asks the model to remove accumulated state error. "
        "The velocity losses simultaneously ask it to reproduce only true "
        "scene motion. Teacher forcing satisfies both because predicted and "
        "true previous states coincide. Free rollout does not.",
        "",
        "Inference-only gain and bleed changes were rejected:",
        "",
        "- Fast scaling amplifies directionally wrong velocity.",
        "- Stronger memory bleed pulls toward interpolated anchors and blurs "
        "local contours.",
        "- Memory-only output confirms anchors are recovery references, not "
        "finished frames.",
        "",
        "## Greedy ablation",
        "",
        "| Variant | Sample error | Sample boundary F1 | "
        "53-55s error | 53-55s boundary F1 | Mean gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {sample_error:.4f} | {sample_f1:.4f} | "
            "{focus_error:.4f} | {focus_f1:.4f} | {gate:.4f} |".format(
                variant=row["variant"],
                sample_error=row["sample"]["mean_pixel_error"],
                sample_f1=row["sample"]["mean_boundary_f1"],
                focus_error=row["focus"]["mean_pixel_error"],
                focus_f1=row["focus"]["mean_boundary_f1"],
                gate=row["internal"]["mean_gate"],
            )
        )
    mismatch_path = output_dir / "target_mismatch.json"
    if mismatch_path.exists():
        mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## 53-55 second target mismatch",
                "",
                (
                    "- True scene-velocity RMS: "
                    f"{mismatch['true_velocity_rms']:.4f}."
                ),
                (
                    "- Required state-relative recovery RMS: "
                    f"{mismatch['required_recovery_velocity_rms']:.4f} "
                    f"({mismatch['recovery_to_true_ratio']:.2f}x larger)."
                ),
                (
                    "- Previous rollout-state MSE: "
                    f"{mismatch['previous_state_mse']:.4f}."
                ),
                (
                    "- Predicted velocity MSE versus training target: "
                    f"{mismatch['predicted_vs_true_velocity_mse']:.4f}."
                ),
                (
                    "- Predicted velocity MSE versus required recovery: "
                    f"{mismatch['predicted_vs_recovery_velocity_mse']:.4f}."
                ),
            ]
        )
    oracle_path = output_dir / "oracle_state_correction.json"
    if oracle_path.exists():
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Causal oracle test",
                "",
                "For diagnosis only, subtracting a known fraction of previous "
                "state error from each next prediction isolates the missing "
                "recovery behavior.",
                "",
                "| Correction fraction | Pixel error | Boundary F1 | "
                "Latent MSE |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in oracle["results"]:
            lines.append(
                "| {fraction:.2f} | {pixel_error:.4f} | "
                "{boundary_f1:.4f} | {latent_mse:.4f} |".format(**row)
            )
        lines.extend(
            [
                "",
                "This oracle uses true previous state and is not deployable. "
                "Its monotonic improvement establishes causality and defines "
                "the next training change.",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended implementation",
            "",
            "Fine-tune motion path with state-relative velocity supervision:",
            "",
            "`recovery_velocity = target - latent_history[:, -1]`",
            "",
            "Keep original true-scene velocity as a smaller auxiliary term so "
            "local motion character remains. Freeze polarity spline and scene "
            "memory. Train on self-generated burn-ins, then validate teacher "
            "precision and full rollout together.",
            "",
            "## Definitions",
            "",
            "- Sample metrics: every configured stride plus every 53-55s frame.",
            "- Boundary F1: silhouette edges matched within two pixels.",
            "- Memory-only: two nearest learned time anchors, no autoregression.",
            "- Moving bleed: restores anchor correction where motion mask "
            "normally suppresses it.",
            "- Recovery: adds anchor correction when latent disagreement grows.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def diagnose_silhouette(config: SilhouetteDiagnosticConfig) -> Path:
    if config.variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {config.variant}; choose from {sorted(VARIANTS)}"
        )
    if config.sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    if config.focus_end_seconds < config.focus_start_seconds:
        raise ValueError("focus end must not precede focus start")

    device = resolve_device(config.device)
    checkpoint = torch.load(
        config.checkpoint, map_location="cpu", weights_only=False
    )
    model = _load_model(checkpoint, device)
    autoencoder, autoencoder_checkpoint = load_autoencoder(
        Path(checkpoint["autoencoder_checkpoint"]), device
    )
    latents, polarities = _load_or_create_cache(
        config,
        checkpoint,
        autoencoder,
        autoencoder_checkpoint,
        device,
    )
    sample_indices = _sample_indices(config, len(latents))
    variant = VARIANTS[config.variant]
    sampled_latents, internal = _rollout_samples(
        model, latents, sample_indices, variant, device
    )

    dataset = FrameDataset(
        config.frame_dir,
        *checkpoint["image_size"],
        input_threshold=checkpoint["input_threshold"],
    )
    targets = torch.stack([dataset[index][0] for index in sample_indices])
    sampled_polarities = polarities[sample_indices, None, None, None] > 0.5
    canonical_targets = torch.where(
        sampled_polarities, 1.0 - targets, targets
    ).bool()
    measurements = _decode_and_measure(
        autoencoder,
        checkpoint,
        sampled_latents,
        canonical_targets,
        device,
        config.batch_size,
    )
    focus_positions = [
        position
        for position, frame_index in enumerate(sample_indices)
        if config.focus_start_seconds
        <= frame_index / config.fps
        <= config.focus_end_seconds
    ]
    all_positions = list(range(len(sample_indices)))
    result = {
        "variant": variant.name,
        "parameters": asdict(variant),
        "checkpoint": str(config.checkpoint.resolve()),
        "cache": str(config.cache_path.resolve()),
        "frame_count": len(latents),
        "sample_count": len(sample_indices),
        "sample_stride": config.sample_stride,
        "focus_seconds": [
            config.focus_start_seconds,
            config.focus_end_seconds,
        ],
        "sample": _summarize(measurements, all_positions),
        "focus": _summarize(measurements, focus_positions),
        "internal": internal,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.output_dir / f"{variant.name}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(config.output_dir)
    print(json.dumps(result, indent=2))
    return result_path
