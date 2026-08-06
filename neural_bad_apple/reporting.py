from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_training_report(
    run_dir: Path,
    *,
    title: str,
    status: str,
    architecture: Iterable[str],
    config: dict[str, Any],
    command: str | None = None,
    history: list[dict[str, Any]] | None = None,
    checkpoint: Path | None = None,
    notes: Iterable[str] = (),
) -> Path:
    """Write a human-readable experiment log beside the checkpoints."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"**Status:** {status}",
        "",
        "## Purpose and architecture",
        "",
    ]
    lines.extend(f"- {item}" for item in architecture)
    if command:
        lines.extend(
            [
                "",
                "## Reproduction command",
                "",
                "```powershell",
                command,
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(_json_ready(config), indent=2),
            "```",
        ]
    )
    if checkpoint is not None:
        lines.extend(
            [
                "",
                "## Checkpoint",
                "",
                f"`{checkpoint.resolve()}`",
            ]
        )
    if history:
        lines.extend(
            [
                "",
                "## Training history",
                "",
                "| Epoch | Stage | Burn-in | Rollout | Train loss | "
                "Rollout MSE | Peak MSE | Seconds |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in history:
            lines.append(
                "| {epoch} | {stage} | {burn_in} | {rollout} | "
                "{loss} | {mse} | {peak} | {seconds} |".format(
                    epoch=row.get("epoch", "-"),
                    stage=row.get("training_stage", "-"),
                    burn_in=row.get("mean_burn_in_steps", 0),
                    rollout=row.get("active_rollout_steps", "-"),
                    loss=_metric(row.get("training_loss", "-")),
                    mse=_metric(row.get("rollout_mse", "-")),
                    peak=_metric(row.get("peak_frame_mse", "-")),
                    seconds=_metric(row.get("seconds", "-")),
                )
            )
    notes = list(notes)
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_render_report(
    output_dir: Path,
    *,
    title: str,
    checkpoint: Path,
    summary: dict[str, Any],
    notes: Iterable[str] = (),
) -> Path:
    """Write a compact explanation beside rendered videos and diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = (
        "frame_count",
        "fps",
        "post_cutoff_mean_teacher_binary_error",
        "post_cutoff_mean_rollout_binary_error",
        "post_cutoff_mean_accumulation_gap",
        "post_cutoff_mean_rollout_iou",
        "peak_error_frame",
        "peak_error_seconds",
        "peak_rollout_binary_error",
        "final_rollout_binary_error",
    )
    lines = [
        f"# {title}",
        "",
        f"**Checkpoint:** `{checkpoint.resolve()}`",
        "",
        "## What is in this folder",
        "",
        "- `error_curve.csv` and `error_curve.png`: frame-level drift metrics.",
    ]
    if "videos" in summary:
        lines.extend(
            [
                "- `comparison.mp4`: target, teacher-forced prediction, free "
                "rollout, and error map.",
                "- `free_rollout.mp4`: uninterrupted autoregressive dream.",
                "- `teacher_forced.mp4`: one-step control using true history.",
                "- `error_maps.mp4`: false positives in red and misses in "
                "cyan.",
            ]
        )
    else:
        lines.append("- MP4 generation was disabled for this metrics-only run.")
    lines.extend(
        [
            "",
            "## Headline metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for name in metric_names:
        if name in summary:
            lines.append(f"| `{name}` | {_metric(summary[name])} |")
    memory = summary.get("memory_usage")
    if memory:
        lines.extend(
            [
                "",
                "## Memory and motion diagnostics",
                "",
                "```json",
                json.dumps(_json_ready(memory), indent=2),
                "```",
            ]
        )
    notes = list(notes)
    if notes:
        lines.extend(["", "## Interpretation notes", ""])
        lines.extend(f"- {note}" for note in notes)
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
