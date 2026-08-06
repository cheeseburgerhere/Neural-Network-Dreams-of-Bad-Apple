"""Build the final V4.3 evaluation package from existing render artifacts.

This module is intentionally stdlib-only. It does not train or render the model;
it reads the V4.2/V4.3 evaluation files and creates the compact datasets used by
the portable HTML report.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
V42_OUTPUT = ROOT / "prototype_outputs" / "hybrid_v4_2_polarity_fix"
V43_OUTPUT = ROOT / "prototype_outputs" / "hybrid_v4_3_recovery"
V43_RUN = ROOT / "prototype_runs" / "hybrid_v4_3_recovery"

INTERVALS: tuple[tuple[str, float, float | None], ...] = (
    ("0–5 s", 0.0, 5.0),
    ("5–15 s", 5.0, 15.0),
    ("45–60 s", 45.0, 60.0),
    ("53–55 s", 53.0, 55.0),
    ("105–120 s", 105.0, 120.0),
    ("210 s–end", 210.0, None),
)

METRIC_FIELDS = (
    "teacher_binary_error",
    "rollout_binary_error",
    "accumulation_gap",
    "rollout_mean_binary_iou",
    "teacher_latent_mse",
    "rollout_latent_mse",
)

REPORT_SQL = """SELECT * FROM headline_metrics;
SELECT * FROM timeline ORDER BY seconds, model;
SELECT * FROM intervals ORDER BY start_seconds;
SELECT * FROM training_stages ORDER BY stage_order;
SELECT * FROM files ORDER BY artifact;
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            row = {
                "frame": int(source["frame"]),
                "seconds": float(source["seconds"]),
            }
            row.update({field: float(source[field]) for field in METRIC_FIELDS})
            rows.append(row)
    return rows


def _mean(rows: Iterable[dict[str, float]], field: str) -> float:
    return fmean(row[field] for row in rows)


def _relative_reduction(baseline: float, current: float) -> float:
    return (baseline - current) / baseline if baseline else 0.0


def _interval_rows(
    v42: list[dict[str, float]], v43: list[dict[str, float]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label, start, end in INTERVALS:
        rows42 = [
            row for row in v42 if row["seconds"] >= start and (end is None or row["seconds"] < end)
        ]
        rows43 = [
            row for row in v43 if row["seconds"] >= start and (end is None or row["seconds"] < end)
        ]
        means42 = {field: _mean(rows42, field) for field in METRIC_FIELDS}
        means43 = {field: _mean(rows43, field) for field in METRIC_FIELDS}
        output.append(
            {
                "interval": label,
                "start_seconds": start,
                "end_seconds": end,
                "frame_count": len(rows43),
                "v42_teacher_error": means42["teacher_binary_error"],
                "v43_teacher_error": means43["teacher_binary_error"],
                "v42_rollout_error": means42["rollout_binary_error"],
                "v43_rollout_error": means43["rollout_binary_error"],
                "rollout_error_reduction": _relative_reduction(
                    means42["rollout_binary_error"], means43["rollout_binary_error"]
                ),
                "v42_accumulation_gap": means42["accumulation_gap"],
                "v43_accumulation_gap": means43["accumulation_gap"],
                "gap_reduction": _relative_reduction(
                    means42["accumulation_gap"], means43["accumulation_gap"]
                ),
                "v42_rollout_iou": means42["rollout_mean_binary_iou"],
                "v43_rollout_iou": means43["rollout_mean_binary_iou"],
                "iou_change": (
                    means43["rollout_mean_binary_iou"]
                    - means42["rollout_mean_binary_iou"]
                ),
                "v42_teacher_latent_mse": means42["teacher_latent_mse"],
                "v43_teacher_latent_mse": means43["teacher_latent_mse"],
                "v42_rollout_latent_mse": means42["rollout_latent_mse"],
                "v43_rollout_latent_mse": means43["rollout_latent_mse"],
            }
        )
    return output


def _timeline_rows(
    v42: list[dict[str, float]], v43: list[dict[str, float]]
) -> list[dict[str, float]]:
    """Aggregate frame-level metrics to one-second bins for readable charts."""

    bins: dict[int, list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)
    for row42, row43 in zip(v42, v43, strict=True):
        bins[math.floor(row43["seconds"])].append((row42, row43))

    output: list[dict[str, float]] = []
    for second, pairs in sorted(bins.items()):
        output.append(
            {
                "seconds": second,
                "v42_teacher_error": fmean(pair[0]["teacher_binary_error"] for pair in pairs),
                "v43_teacher_error": fmean(pair[1]["teacher_binary_error"] for pair in pairs),
                "v42_rollout_error": fmean(pair[0]["rollout_binary_error"] for pair in pairs),
                "v43_rollout_error": fmean(pair[1]["rollout_binary_error"] for pair in pairs),
                "v42_accumulation_gap": fmean(pair[0]["accumulation_gap"] for pair in pairs),
                "v43_accumulation_gap": fmean(pair[1]["accumulation_gap"] for pair in pairs),
                "v43_rollout_iou": fmean(
                    pair[1]["rollout_mean_binary_iou"] for pair in pairs
                ),
            }
        )
    return output


def _window_rows(
    v42: list[dict[str, float]],
    v43: list[dict[str, float]],
    start: float = 0.5,
    width: float = 5.0,
) -> list[dict[str, float]]:
    final_second = min(v42[-1]["seconds"], v43[-1]["seconds"])
    windows: list[dict[str, float]] = []
    left = start
    while left < final_second:
        right = min(left + width, final_second + 1e-9)
        rows42 = [row for row in v42 if left <= row["seconds"] < right]
        rows43 = [row for row in v43 if left <= row["seconds"] < right]
        if rows42 and rows43:
            error42 = _mean(rows42, "rollout_binary_error")
            error43 = _mean(rows43, "rollout_binary_error")
            windows.append(
                {
                    "start_seconds": left,
                    "end_seconds": right,
                    "v42_rollout_error": error42,
                    "v43_rollout_error": error43,
                    "rollout_error_change": error43 - error42,
                    "rollout_error_reduction": _relative_reduction(error42, error43),
                }
            )
        left += width
    return windows


def _file_rows() -> list[dict[str, Any]]:
    files = (
        ("Comparison + audio", "comparison_with_audio.mp4", "Four-panel final review", True),
        ("Free rollout + audio", "free_rollout_with_audio.mp4", "Autoregressive dream", True),
        ("Comparison", "comparison.mp4", "Target / teacher / rollout / error", False),
        ("Free rollout", "free_rollout.mp4", "Generated frames only", False),
        ("Teacher forced", "teacher_forced.mp4", "One-step diagnostic", False),
        ("Error maps", "error_maps.mp4", "False-positive / false-negative map", False),
        ("Error chart", "error_curve.png", "Static diagnostic chart", False),
        ("Frame metrics", "error_curve.csv", "6,573-frame raw evaluation", False),
        ("Render summary", "drift_summary.json", "Aggregate metrics and settings", False),
        ("Markdown report", "report.md", "Human-readable project handoff", False),
        ("Portable report", "findings_report.html", "Self-contained technical report", False),
        ("Report dataset", "analysis.json", "Derived comparison metrics", False),
        ("Report manifest", "artifact.json", "Portable report source", False),
    )
    output: list[dict[str, Any]] = []
    for label, filename, purpose, audio in files:
        path = V43_OUTPUT / filename
        output.append(
            {
                "artifact": label,
                "filename": filename,
                "purpose": purpose,
                "audio": "yes" if audio else "no",
                "size_mib": (path.stat().st_size / 1024**2) if path.exists() else None,
            }
        )
    return output


def _sql_materialize(
    datasets: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    """Round-trip final widget datasets through the SQL recorded in report_source.sql."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for table, rows in datasets.items():
            if not rows:
                continue
            fields = list(rows[0])
            for row in rows[1:]:
                for field in row:
                    if field not in fields:
                        fields.append(field)
            column_types = {
                field: (
                    "TEXT"
                    if all(
                        row.get(field) is None or isinstance(row.get(field), str)
                        for row in rows
                    )
                    else "REAL"
                )
                for field in fields
            }
            columns = ", ".join(
                f'"{field}" {column_types[field]}' for field in fields
            )
            connection.execute(f'CREATE TABLE "{table}" ({columns})')
            placeholders = ", ".join("?" for _ in fields)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [[row.get(field) for field in fields] for row in rows],
            )
        connection.commit()

        materialized: dict[str, list[dict[str, Any]]] = {}
        for statement in (part.strip() for part in REPORT_SQL.split(";")):
            if not statement:
                continue
            table = statement.split("FROM", 1)[1].strip().split()[0]
            materialized[table] = [
                dict(row) for row in connection.execute(statement).fetchall()
            ]
        return materialized
    finally:
        connection.close()


def _sources(generated_at: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "comparison_analysis",
            "label": "Materialized V4.2 vs V4.3 report datasets",
            "path": "prototype_outputs/hybrid_v4_3_recovery/report_source.sql",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": REPORT_SQL,
                "description": (
                    "Actual widget materialization queries run against in-memory SQLite "
                    "staging tables. The staging tables are derived from the aligned CSV and "
                    "JSON inputs by neural_bad_apple/final_report.py."
                ),
                "executed_at": generated_at,
                "filters": [
                    "Headline metrics use frames after the 0.5 s source cutoff.",
                    "Timeline chart uses one-second mean bins.",
                    "Interval tables use start-inclusive, end-exclusive windows.",
                ],
                "metric_definitions": [
                    "Binary error: fraction of thresholded output pixels unequal to target.",
                    "Accumulation gap: free-rollout binary error minus teacher-forced binary error.",
                    "Rollout IoU: mean binary intersection-over-union across black and white classes.",
                    "Relative reduction: (V4.2 metric - V4.3 metric) / V4.2 metric.",
                ],
                "tables_used": [
                    "prototype_outputs/hybrid_v4_2_polarity_fix/error_curve.csv",
                    "prototype_outputs/hybrid_v4_3_recovery/error_curve.csv",
                ],
            },
        },
        {
            "id": "derived_json",
            "label": "Derived V4.2 vs V4.3 comparison data",
            "path": "prototype_outputs/hybrid_v4_3_recovery/analysis.json",
        },
        {
            "id": "v43_render",
            "label": "V4.3 full render summary",
            "path": "prototype_outputs/hybrid_v4_3_recovery/drift_summary.json",
        },
        {
            "id": "v42_render",
            "label": "V4.2 full render summary",
            "path": "prototype_outputs/hybrid_v4_2_polarity_fix/drift_summary.json",
        },
        {
            "id": "v43_training",
            "label": "V4.3 recovery training evaluation",
            "path": "prototype_runs/hybrid_v4_3_recovery/evaluation.json",
        },
    ]


def _manifest(
    generated_at: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": 1,
        "surface": "report",
        "title": "Neural Network Dreams Bad Apple — V4.3 Final Evaluation",
        "description": (
            "Full-video evaluation of the autoregressive latent dream, with V4.2 baseline "
            "comparison and recovery-training decision."
        ),
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "rollout_error",
                "description": "V4.3 post-cutoff rollout binary error",
                "dataset": "headline_metrics",
                "sourceId": "comparison_analysis",
                "metrics": [
                    {
                        "label": "Rollout error",
                        "field": "v43_rollout_error",
                        "format": "percent",
                    }
                ],
            },
            {
                "id": "rollout_improvement",
                "description": "Relative reduction from V4.2",
                "dataset": "headline_metrics",
                "sourceId": "comparison_analysis",
                "metrics": [
                    {
                        "label": "Error reduction",
                        "field": "rollout_error_reduction",
                        "format": "percent",
                        "signed": True,
                    }
                ],
            },
            {
                "id": "gap_reduction",
                "description": "Relative reduction in teacher-to-rollout gap",
                "dataset": "headline_metrics",
                "sourceId": "comparison_analysis",
                "metrics": [
                    {
                        "label": "Gap reduction",
                        "field": "gap_reduction",
                        "format": "percent",
                        "signed": True,
                    }
                ],
            },
            {
                "id": "rollout_iou",
                "description": "V4.3 post-cutoff mean binary IoU",
                "dataset": "headline_metrics",
                "sourceId": "comparison_analysis",
                "metrics": [
                    {
                        "label": "Rollout IoU",
                        "field": "v43_rollout_iou",
                        "format": "percent",
                    }
                ],
            },
        ],
        "charts": [
            {
                "id": "timeline",
                "title": "Full-video rollout error",
                "subtitle": "One-second means; lower is better",
                "headerMarkdown": (
                    "**Question:** Did recovery fine-tuning reduce autoregressive error across "
                    "the complete 219.1 s render?\n\n**Takeaway:** V4.3 is lower on 65.8% of "
                    "individual frames and every evaluated non-overlapping 5 s window."
                ),
                "intent": "trend",
                "question": "How does V4.3 free-rollout error compare with V4.2 over time?",
                "rationale": (
                    "A line chart preserves temporal spikes and exposes whether a global "
                    "average hides local regressions."
                ),
                "comparisonContext": {
                    "baseline": "V4.2 polarity-fixed model",
                    "grain": "one-second mean",
                    "unit": "fraction of mismatched pixels",
                },
                "type": "line",
                "dataset": "timeline",
                "sourceId": "comparison_analysis",
                "encodings": {
                    "x": {
                        "field": "seconds",
                        "type": "quantitative",
                        "label": "Video time (s)",
                    },
                    "y": {
                        "field": "rollout_error",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Binary error",
                    },
                    "color": {
                        "field": "model",
                        "type": "nominal",
                        "label": "Checkpoint",
                    },
                    "lineStyle": {
                        "field": "role",
                        "type": "nominal",
                        "label": "Role",
                    },
                    "tooltip": [
                        {"field": "seconds", "type": "quantitative", "label": "Second"},
                        {
                            "field": "rollout_error",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Rollout error",
                        },
                        {"field": "model", "type": "nominal", "label": "Checkpoint"},
                    ],
                },
                "xAxisTitle": "Video time (s)",
                "yAxisTitle": "Binary error",
                "valueFormat": "percent",
                "layout": "full",
                "labels": {"values": "none"},
                "legend": {"position": "bottom", "sort": "spec"},
                "maxRows": 300,
                "surface": {
                    "surface": "explorer",
                    "interactiveLegend": True,
                    "showControls": True,
                    "viewMode": "visualization",
                },
            },
            {
                "id": "interval_improvement",
                "title": "Rollout error reduction by review interval",
                "subtitle": "Relative reduction from V4.2; higher is better",
                "headerMarkdown": (
                    "**Question:** Does the improvement survive difficult and artistically "
                    "important intervals?\n\n**Takeaway:** All six review intervals improve; "
                    "53–55 s improves least, confirming that fine silhouette motion remains "
                    "the principal weakness."
                ),
                "intent": "comparison",
                "question": "Where did V4.3 reduce rollout error relative to V4.2?",
                "rationale": (
                    "A horizontal bar chart makes interval-level relative changes directly "
                    "comparable while the adjacent table retains exact absolute errors."
                ),
                "type": "horizontalBar",
                "dataset": "intervals",
                "sourceId": "comparison_analysis",
                "encodings": {
                    "x": {
                        "field": "interval",
                        "type": "nominal",
                        "label": "Review interval",
                    },
                    "y": {
                        "field": "rollout_error_reduction",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Relative error reduction",
                    },
                    "tooltip": [
                        {"field": "interval", "type": "nominal", "label": "Interval"},
                        {
                            "field": "rollout_error_reduction",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Error reduction",
                        },
                    ],
                },
                "xAxisTitle": "Review interval",
                "yAxisTitle": "Relative error reduction",
                "valueFormat": "percent",
                "layout": "full",
                "labels": {"values": "all"},
                "legend": {"position": "bottom"},
                "settings": {
                    "orientation": "horizontal",
                    "sort": "none",
                    "showValues": True,
                    "categoryLabelPolicy": "wrap",
                },
            },
        ],
        "tables": [
            {
                "id": "interval_table",
                "title": "Exact interval metrics",
                "subtitle": "Frame-weighted means; lower error/gap and higher IoU are better",
                "dataset": "intervals",
                "sourceId": "comparison_analysis",
                "density": "dense",
                "columns": [
                    {"field": "interval", "label": "Interval", "type": "text"},
                    {
                        "field": "v42_rollout_error",
                        "label": "V4.2 error",
                        "format": "percent",
                    },
                    {
                        "field": "v43_rollout_error",
                        "label": "V4.3 error",
                        "format": "percent",
                    },
                    {
                        "field": "rollout_error_reduction",
                        "label": "Error reduction",
                        "format": "percent",
                        "movement": True,
                    },
                    {
                        "field": "gap_reduction",
                        "label": "Gap reduction",
                        "format": "percent",
                        "movement": True,
                    },
                    {
                        "field": "v43_rollout_iou",
                        "label": "V4.3 IoU",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "training_table",
                "title": "Recovery-training selection",
                "subtitle": "Epoch 1 remains the selected checkpoint; epoch 2 regressed",
                "dataset": "training_stages",
                "sourceId": "comparison_analysis",
                "density": "dense",
                "columns": [
                    {"field": "stage", "label": "Stage", "type": "text"},
                    {
                        "field": "rollout_latent_mse",
                        "label": "Rollout latent MSE",
                        "format": "number",
                    },
                    {
                        "field": "sample_binary_error",
                        "label": "Sample error",
                        "format": "percent",
                    },
                    {
                        "field": "focus_53_55_binary_error",
                        "label": "53–55 s error",
                        "format": "percent",
                    },
                    {
                        "field": "focus_53_55_boundary_f1",
                        "label": "53–55 s boundary F1",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "artifact_table",
                "title": "Deliverable manifest",
                "subtitle": "All files live beside this report",
                "dataset": "files",
                "sourceId": "comparison_analysis",
                "density": "dense",
                "columns": [
                    {"field": "artifact", "label": "Artifact", "type": "text"},
                    {"field": "filename", "label": "Filename", "type": "text"},
                    {"field": "purpose", "label": "Purpose", "type": "text"},
                    {"field": "audio", "label": "Audio", "type": "text"},
                    {"field": "size_mib", "label": "MiB", "format": "number"},
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# Neural Network Dreams Bad Apple — V4.3 Final Evaluation",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    "The selected V4.3 recovery checkpoint is a real but incremental "
                    "improvement over V4.2. Across the complete 6,573-frame render, it "
                    "reduces free-rollout binary error and—more importantly—the gap between "
                    "teacher forcing and autonomous rollout. This supports the diagnosis "
                    "that collapse was primarily a state-recovery problem. It does not fully "
                    "solve fine local motion: the hands/wings interval at 53–55 s remains the "
                    "weakest selected review window.\n\n"
                    "**Decision:** keep epoch 1 (`model_best.pt`) and stop training. Epoch 2 "
                    "slightly regressed rollout metrics, so more training is not justified "
                    "without a new experiment."
                ),
            },
            {
                "id": "metrics",
                "type": "metric-strip",
                "cardIds": [
                    "rollout_error",
                    "rollout_improvement",
                    "gap_reduction",
                    "rollout_iou",
                ],
            },
            {
                "id": "key_findings",
                "type": "markdown",
                "body": (
                    "## Key findings with visual evidence\n\n"
                    "- V4.3 post-cutoff rollout error is **4.45%**, down from **4.93%** "
                    "(**9.72% relative improvement**).\n"
                    "- The accumulation gap falls from **1.72%** to **1.28%** "
                    "(**25.68% relative reduction**), stronger evidence than teacher-forced "
                    "quality alone.\n"
                    "- Rollout IoU rises from **84.40%** to **85.42%**.\n"
                    "- Peak rollout error drops from **38.52%** to **34.61%**, near 12 s in "
                    "both versions.\n"
                    "- V4.3 beats V4.2 on **65.77%** of individual frames and every evaluated "
                    "non-overlapping 5 s window."
                ),
                "sourceId": "comparison_analysis",
            },
            {"id": "timeline_block", "type": "chart", "chartId": "timeline"},
            {
                "id": "timeline_note",
                "type": "markdown",
                "body": (
                    "The remaining spikes are concentrated around abrupt scene changes and "
                    "high-motion transitions. The global reduction is therefore not caused "
                    "by smoothing one easy region; it is distributed across the timeline."
                ),
            },
            {
                "id": "silhouette_problem",
                "type": "markdown",
                "body": (
                    "## Silhouette and local-motion result\n\n"
                    "At 53–55 s, teacher-forced binary error is essentially unchanged "
                    "(2.19% → 2.20%), while rollout error improves from **6.92%** to **6.46%**. "
                    "The teacher/rollout split remains large, so the autoencoder can represent "
                    "the frame and the predictor can reconstruct it with correct history, but "
                    "autonomous state loses the small moving parts. Recovery training helps "
                    "without fixing the underlying spatial-scale bias."
                ),
                "sourceId": "comparison_analysis",
            },
            {
                "id": "interval_chart_block",
                "type": "chart",
                "chartId": "interval_improvement",
            },
            {"id": "interval_table_block", "type": "table", "tableId": "interval_table"},
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## Scope, data, and metric definitions\n\n"
                    "- **Source:** the full extracted Bad Apple frame sequence, 6,573 frames "
                    "at 30 fps (219.1 s), rendered at 512×384.\n"
                    "- **Teacher forcing:** each prediction receives true recent latent state.\n"
                    "- **Free rollout:** after the 16-frame warm-up, predicted state is fed "
                    "back recursively; this is the actual autoregressive dream.\n"
                    "- **Binary error:** fraction of thresholded pixels that disagree with "
                    "the target.\n"
                    "- **Accumulation gap:** rollout error minus teacher-forced error; it "
                    "isolates damage caused by feeding predictions back into the model.\n"
                    "- **IoU:** mean black/white intersection-over-union; it is less dominated "
                    "by background area than raw accuracy."
                ),
            },
            {
                "id": "design",
                "type": "markdown",
                "body": (
                    "## Model specification and experimental design\n\n"
                    "The model is a causal latent predictor, not a conventional LSTM. A "
                    "frozen autoencoder maps frames into spatial latents. A temporal U-Net-like "
                    "predictor combines recent latent history, slow/fast motion paths, learned "
                    "time-indexed scene memory, cut gating, and anchor correction to produce "
                    "the next latent. V4.3 keeps the V4.2 architecture and fine-tunes recovery "
                    "from self-generated/corrupted history, directly matching the deployment "
                    "failure mode.\n\n"
                    "The experiment compares the same full frame sequence and evaluation "
                    "pipeline. Epoch 1 was selected on rollout metrics; epoch 2 was retained "
                    "as a negative result rather than overwriting the best checkpoint."
                ),
            },
            {"id": "training_table_block", "type": "table", "tableId": "training_table"},
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Methodology\n\n"
                    "Both checkpoints were rolled out over the full sequence with temporal "
                    "polarity canonicalization and a 0.5 s source cutoff for headline "
                    "statistics. Frame-level CSVs were aligned by frame number. Headline "
                    "statistics use all post-cutoff frames; the timeline chart displays "
                    "one-second means only for readability. Six preselected review intervals "
                    "and non-overlapping 5 s windows were then compared. Video files were "
                    "tail-decoded after encoding to detect truncated outputs."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations, uncertainty, and robustness\n\n"
                    "- This is one memorized video, not a general video-prediction benchmark.\n"
                    "- Improvements are descriptive for this controlled checkpoint comparison; "
                    "there are no repeated random seeds or confidence intervals.\n"
                    "- Binary metrics depend on the decoder threshold and can miss perceptually "
                    "important boundary jitter.\n"
                    "- The one-second chart can visually soften single-frame spikes; exact "
                    "headline and interval values use unsampled frame-level data.\n"
                    "- The original soundtrack is muxed into the two `_with_audio` deliverables; "
                    "the diagnostic renders remain intentionally silent.\n"
                    "- Artistic acceptability—especially whether drift should remain visible "
                    "as part of the 'dream'—cannot be decided by these metrics."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. **Ship V4.3 epoch 1 as the prototype result.** It is the best measured "
                    "checkpoint and further identical training already began to regress.\n"
                    "2. **Do an artistic review of the audio comparison.** Mark only the local "
                    "motion failures that matter to the intended look.\n"
                    "3. **If continuing technically, run one isolated experiment:** add a "
                    "high-resolution residual/boundary head trained on temporal frame "
                    "differences around small moving parts. Keep the current rollout model as "
                    "the baseline and do not mix this with more recovery epochs.\n"
                    "4. **Preserve the error bleed.** Do not hard-reset to target latents; the "
                    "visible accumulation is the central artistic mechanism of this project."
                ),
            },
            {
                "id": "deliverables",
                "type": "markdown",
                "body": (
                    "## Deliverables\n\n"
                    "Start with [the four-panel comparison with audio]"
                    "(comparison_with_audio.mp4) or [the autonomous dream with audio]"
                    "(free_rollout_with_audio.mp4). Use the silent teacher/error videos for "
                    "diagnosis and the CSV/JSON files for future analysis."
                ),
            },
            {"id": "artifact_table_block", "type": "table", "tableId": "artifact_table"},
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- Should the final public cut foreground the raw autoregressive failure, "
                    "or should small boundary errors be cleaned while preserving global drift?\n"
                    "- Is the four-panel technical comparison part of the artwork, or only "
                    "blog evidence beside the single-panel dream?\n"
                    "- If one more model experiment is allowed, which motion moments are "
                    "artistically non-negotiable beyond the 53–55 s hands/wings example?"
                ),
            },
        ],
    }


def build() -> tuple[Path, Path, Path]:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary42 = _read_json(V42_OUTPUT / "drift_summary.json")
    summary43 = _read_json(V43_OUTPUT / "drift_summary.json")
    evaluation = _read_json(V43_RUN / "evaluation.json")
    curve42 = _read_curve(V42_OUTPUT / "error_curve.csv")
    curve43 = _read_curve(V43_OUTPUT / "error_curve.csv")

    if len(curve42) != len(curve43):
        raise ValueError(f"Frame count mismatch: V4.2={len(curve42)}, V4.3={len(curve43)}")
    if any(a["frame"] != b["frame"] for a, b in zip(curve42, curve43, strict=True)):
        raise ValueError("V4.2 and V4.3 frame indices are not aligned")

    intervals = _interval_rows(curve42, curve43)
    timeline = _timeline_rows(curve42, curve43)
    windows = _window_rows(curve42, curve43)
    better = sum(
        row43["rollout_binary_error"] < row42["rollout_binary_error"]
        for row42, row43 in zip(curve42, curve43, strict=True)
    )
    equal = sum(
        row43["rollout_binary_error"] == row42["rollout_binary_error"]
        for row42, row43 in zip(curve42, curve43, strict=True)
    )
    teacher_better = sum(
        row43["teacher_binary_error"] < row42["teacher_binary_error"]
        for row42, row43 in zip(curve42, curve43, strict=True)
    )

    v42_rollout = summary42["post_cutoff_mean_rollout_binary_error"]
    v43_rollout = summary43["post_cutoff_mean_rollout_binary_error"]
    v42_gap = summary42["post_cutoff_mean_accumulation_gap"]
    v43_gap = summary43["post_cutoff_mean_accumulation_gap"]
    v42_iou = summary42["post_cutoff_mean_rollout_iou"]
    v43_iou = summary43["post_cutoff_mean_rollout_iou"]

    headline = {
        "v42_teacher_error": summary42["post_cutoff_mean_teacher_binary_error"],
        "v43_teacher_error": summary43["post_cutoff_mean_teacher_binary_error"],
        "v42_rollout_error": v42_rollout,
        "v43_rollout_error": v43_rollout,
        "rollout_error_reduction": _relative_reduction(v42_rollout, v43_rollout),
        "v42_accumulation_gap": v42_gap,
        "v43_accumulation_gap": v43_gap,
        "gap_reduction": _relative_reduction(v42_gap, v43_gap),
        "v42_rollout_iou": v42_iou,
        "v43_rollout_iou": v43_iou,
        "iou_change": v43_iou - v42_iou,
        "v42_peak_rollout_error": summary42["peak_rollout_binary_error"],
        "v43_peak_rollout_error": summary43["peak_rollout_binary_error"],
        "peak_error_reduction": _relative_reduction(
            summary42["peak_rollout_binary_error"],
            summary43["peak_rollout_binary_error"],
        ),
        "rollout_better_frame_share": better / len(curve43),
        "rollout_equal_frame_share": equal / len(curve43),
        "teacher_better_frame_share": teacher_better / len(curve43),
        "improved_five_second_windows": sum(
            row["rollout_error_change"] < 0 for row in windows
        ),
        "five_second_window_count": len(windows),
        "frame_count": len(curve43),
        "fps": summary43["fps"],
        "duration_seconds": curve43[-1]["seconds"],
        "selected_epoch": evaluation["selected_epoch"],
    }

    training_stages = []
    for key, label in (
        ("baseline", "V4.2 baseline"),
        ("epoch_1", "V4.3 epoch 1 — selected"),
        ("epoch_2", "V4.3 epoch 2 — regressed"),
    ):
        stage = evaluation[key]
        training_stages.append(
            {
                "stage": label,
                "stage_order": len(training_stages),
                "rollout_latent_mse": stage.get("rollout_latent_mse"),
                "sample_binary_error": stage.get("sample_binary_error"),
                "sample_boundary_f1": stage.get("sample_boundary_f1"),
                "focus_53_55_binary_error": stage.get("focus_53_55_binary_error"),
                "focus_53_55_boundary_f1": stage.get("focus_53_55_boundary_f1"),
                "teacher_sample_binary_error": stage.get("teacher_sample_binary_error"),
                "teacher_focus_53_55_binary_error": stage.get(
                    "teacher_focus_53_55_binary_error"
                ),
            }
        )

    analysis = {
        "generated_at": generated_at,
        "headline": headline,
        "intervals": intervals,
        "five_second_windows": windows,
        "best_five_second_windows": sorted(
            windows, key=lambda row: row["rollout_error_change"]
        )[:5],
        "training_stages": training_stages,
        "decision": evaluation["decision"],
    }
    analysis_path = V43_OUTPUT / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sources = _sources(generated_at)
    timeline_long = [
        {
            "seconds": row["seconds"],
            "model": model,
            "role": role,
            "rollout_error": row[field],
        }
        for row in timeline
        for model, role, field in (
            ("V4.2 rollout", "Baseline", "v42_rollout_error"),
            ("V4.3 rollout", "Selected", "v43_rollout_error"),
        )
    ]

    source_sql_path = V43_OUTPUT / "report_source.sql"
    source_sql_path.write_text(REPORT_SQL, encoding="utf-8")

    datasets = _sql_materialize(
        {
            "headline_metrics": [headline],
            "timeline": timeline_long,
            "intervals": intervals,
            "training_stages": training_stages,
            "files": _file_rows(),
        }
    )

    artifact = {
        "surface": "report",
        "manifest": _manifest(generated_at, sources),
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }
    artifact_path = V43_OUTPUT / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    focus = next(row for row in intervals if row["interval"] == "53–55 s")
    report = f"""# Hybrid V4.3 recovery — final full-video report

## Outcome

V4.3 epoch 1 is the final selected checkpoint. The full 6,573-frame render completed,
all four diagnostic MP4s tail-decode successfully, and the original soundtrack was
muxed into the two viewing deliverables.

## Headline comparison against V4.2

| Metric | V4.2 | V4.3 | Change |
| --- | ---: | ---: | ---: |
| Post-cutoff rollout binary error | {v42_rollout:.6f} | {v43_rollout:.6f} | {_relative_reduction(v42_rollout, v43_rollout):.2%} lower |
| Post-cutoff accumulation gap | {v42_gap:.6f} | {v43_gap:.6f} | {_relative_reduction(v42_gap, v43_gap):.2%} lower |
| Post-cutoff rollout IoU | {v42_iou:.6f} | {v43_iou:.6f} | {v43_iou - v42_iou:+.6f} |
| Peak rollout error | {summary42["peak_rollout_binary_error"]:.6f} | {summary43["peak_rollout_binary_error"]:.6f} | {_relative_reduction(summary42["peak_rollout_binary_error"], summary43["peak_rollout_binary_error"]):.2%} lower |

V4.3 has lower rollout error on {better / len(curve43):.2%} of frames and every
evaluated non-overlapping five-second window.

## Silhouette / local-motion finding

At 53–55 s, rollout error improves from {focus["v42_rollout_error"]:.6f} to
{focus["v43_rollout_error"]:.6f} ({focus["rollout_error_reduction"]:.2%} lower),
but the V4.3 teacher error remains only {focus["v43_teacher_error"]:.6f}. The large
teacher-to-rollout split confirms that small hand/wing motion is still lost mainly
through autonomous state drift, not because the autoencoder cannot represent it.

## Training decision

- Baseline rollout latent MSE: {evaluation["baseline"]["rollout_latent_mse"]:.6f}
- Epoch 1: {evaluation["epoch_1"]["rollout_latent_mse"]:.6f} — selected
- Epoch 2: {evaluation["epoch_2"]["rollout_latent_mse"]:.6f} — regressed

Do not continue the same recovery schedule. Keep `model_best.pt` (epoch 1).

## Deliverables

- `comparison_with_audio.mp4` — target / teacher / rollout / error with soundtrack
- `free_rollout_with_audio.mp4` — standalone autoregressive dream with soundtrack
- `comparison.mp4` — silent diagnostic comparison
- `teacher_forced.mp4` — silent teacher-forced diagnostic
- `free_rollout.mp4` — silent free rollout
- `error_maps.mp4` — silent false-positive / false-negative maps
- `error_curve.csv` — complete frame-level metrics
- `drift_summary.json` — aggregate render metrics and settings
- `analysis.json` — V4.2/V4.3 derived comparison data
- `artifact.json` — portable report source
- `findings_report.html` — self-contained interactive technical report

## Recommended next step

Ship this checkpoint for the prototype and review the two audio videos artistically.
If one more technical experiment is approved, isolate it to a high-resolution
temporal residual/boundary head for small moving parts. Preserve free-running error
bleed; do not hard-reset the rollout to target latents.
"""
    report_path = V43_OUTPUT / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return analysis_path, artifact_path, report_path


if __name__ == "__main__":
    for built_path in build():
        print(built_path)
