from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .hybrid_v4 import BleedingSceneMemoryModel
from .training import resolve_device


@dataclass
class PolarityFixConfig:
    checkpoint: Path
    target_csv: Path
    run_dir: Path
    knot_counts: tuple[int, ...] = (16, 24, 32, 48, 64, 96)
    steps: int = 1500
    learning_rate: float = 0.1
    smoothness_weight: float = 1e-4
    device: str = "auto"


def interpolate_polarity_logits(
    knot_logits: torch.Tensor, normalized_times: torch.Tensor
) -> torch.Tensor:
    positions = normalized_times.clamp(0.0, 1.0) * (
        len(knot_logits) - 1
    )
    left = positions.floor().long()
    right = (left + 1).clamp(max=len(knot_logits) - 1)
    fractions = positions - left
    return (
        knot_logits[left] * (1.0 - fractions)
        + knot_logits[right] * fractions
    )


def load_polarity_targets(path: Path) -> torch.Tensor:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or "target_polarity" not in rows[0]:
        raise ValueError(
            "target CSV must contain at least one target_polarity row"
        )
    return torch.tensor(
        [float(row["target_polarity"]) for row in rows],
        dtype=torch.float32,
    )


def _switch_count(values: torch.Tensor) -> int:
    return int((values[1:] != values[:-1]).sum().item())


def _fit_candidate(
    targets: torch.Tensor,
    knot_count: int,
    steps: int,
    learning_rate: float,
    smoothness_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    normalized_times = torch.linspace(
        0.0, 1.0, steps=len(targets), device=device
    )
    device_targets = targets.to(device)
    knot_logits = torch.nn.Parameter(torch.zeros(knot_count, device=device))
    optimizer = torch.optim.Adam([knot_logits], lr=learning_rate)
    loss = torch.zeros((), device=device)
    binary_loss = torch.zeros((), device=device)
    smoothness = torch.zeros((), device=device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = interpolate_polarity_logits(
            knot_logits, normalized_times
        )
        binary_loss = F.binary_cross_entropy_with_logits(
            logits, device_targets
        )
        second_difference = (
            knot_logits[2:]
            - 2.0 * knot_logits[1:-1]
            + knot_logits[:-2]
        )
        smoothness = second_difference.square().mean()
        loss = binary_loss + smoothness_weight * smoothness
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        logits = interpolate_polarity_logits(
            knot_logits, normalized_times
        )
        predictions = logits >= 0.0
        target_binary = device_targets >= 0.5
        mismatches = predictions != target_binary
        metrics: dict[str, float | int] = {
            "knot_count": knot_count,
            "steps": steps,
            "loss": float(loss.item()),
            "binary_cross_entropy": float(binary_loss.item()),
            "smoothness": float(smoothness.item()),
            "accuracy": float((~mismatches).float().mean().item()),
            "mismatch_frames": int(mismatches.sum().item()),
            "predicted_switches": _switch_count(predictions),
            "target_switches": _switch_count(target_binary),
        }
    return knot_logits.detach().cpu(), metrics


def _write_report(
    config: PolarityFixConfig,
    checkpoint_path: Path,
    candidates: list[dict[str, float | int]],
    selected: dict[str, float | int],
) -> None:
    lines = [
        "# Hybrid V4.2 polarity fix",
        "",
        "**Status:** Complete",
        "",
        "## Result",
        "",
        (
            f"- Selected {selected['knot_count']}-knot normalized-time "
            "linear spline."
        ),
        f"- Accuracy: {100.0 * float(selected['accuracy']):.4f}%.",
        (
            f"- Switches: {selected['predicted_switches']} predicted / "
            f"{selected['target_switches']} target."
        ),
        f"- Mismatch frames: {selected['mismatch_frames']}.",
        "- Temporal U-Net, latent heads, scene memory, and gates unchanged.",
        "",
        "## Candidate sweep",
        "",
        "| Knots | Accuracy | Mismatches | Switches | BCE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidates:
        lines.append(
            "| {knot_count} | {accuracy:.6f} | {mismatch_frames} | "
            "{predicted_switches} | {binary_cross_entropy:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Checkpoint: `{checkpoint_path.resolve()}`",
            f"- Metrics: `{(config.run_dir / 'calibration.json').resolve()}`",
            f"- Source checkpoint: `{config.checkpoint.resolve()}`",
            f"- Polarity targets: `{config.target_csv.resolve()}`",
            "",
            "## Reproduction",
            "",
            "```powershell",
            (
                "python prototype.py fix-polarity "
                f"--checkpoint \"{config.checkpoint}\" "
                f"--target-csv \"{config.target_csv}\" "
                f"--run-dir \"{config.run_dir}\""
            ),
            "```",
            "",
        ]
    )
    (config.run_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def fix_checkpoint_polarity(config: PolarityFixConfig) -> Path:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if not config.knot_counts:
        raise ValueError("at least one knot count is required")
    if any(count < 2 for count in config.knot_counts):
        raise ValueError("each knot count must be at least two")

    device = resolve_device(config.device)
    targets = load_polarity_targets(config.target_csv)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, float | int]] = []
    fitted: dict[int, torch.Tensor] = {}
    for knot_count in config.knot_counts:
        logits, metrics = _fit_candidate(
            targets=targets,
            knot_count=knot_count,
            steps=config.steps,
            learning_rate=config.learning_rate,
            smoothness_weight=config.smoothness_weight,
            device=device,
        )
        fitted[knot_count] = logits
        candidates.append(metrics)
        print(
            f"polarity knots {knot_count:03d} | "
            f"accuracy {float(metrics['accuracy']):.6f} | "
            f"switches {metrics['predicted_switches']} | "
            f"mismatches {metrics['mismatch_frames']}"
        )

    target_switches = int(candidates[0]["target_switches"])
    selected = max(
        candidates,
        key=lambda row: (
            int(row["predicted_switches"]) == target_switches,
            float(row["accuracy"]),
            -int(row["knot_count"]),
        ),
    )
    selected_count = int(selected["knot_count"])

    checkpoint = torch.load(
        config.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_type") != "hybrid_v4_bleeding_memory":
        raise ValueError("polarity fix requires a hybrid v4 checkpoint")
    model_kwargs = dict(checkpoint["model_kwargs"])
    model_kwargs["polarity_knot_count"] = selected_count
    model = BleedingSceneMemoryModel(**model_kwargs)
    missing, unexpected = model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if missing != ["polarity_spline_logits"] or unexpected:
        raise ValueError(
            f"incompatible checkpoint: missing={missing}, "
            f"unexpected={unexpected}"
        )
    with torch.no_grad():
        model.polarity_spline_logits.copy_(fitted[selected_count])

    checkpoint["model_kwargs"] = model_kwargs
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["architecture_version"] = "v4.2-polarity-fix"
    checkpoint["polarity_calibration"] = {
        "method": "normalized_time_linear_spline",
        **selected,
    }
    checkpoint_path = config.run_dir / "model_best.pt"
    torch.save(checkpoint, checkpoint_path)

    payload = {
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint.resolve()),
            "target_csv": str(config.target_csv.resolve()),
            "run_dir": str(config.run_dir.resolve()),
            "knot_counts": list(config.knot_counts),
            "resolved_device": str(device),
        },
        "selected": selected,
        "candidates": candidates,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (config.run_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_report(config, checkpoint_path, candidates, selected)
    return checkpoint_path
