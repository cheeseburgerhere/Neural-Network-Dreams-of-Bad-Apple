"""Inference-only blog assets. Original checkpoints and training code stay unchanged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from .autoregressive import load_autoencoder
from .data import FrameDataset, _imageio_ffmpeg
from .drift_rendering import _mean_binary_iou
from .hybrid import encode_canonical_sequence
from .memory_baseline import memory_only_latents
from .silhouette import _load_model

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "prototype_outputs/blog_work"
OUT = ROOT / "blog_assets"
DATA = ROOT / "prototype_data/full_source_frames"
AE = ROOT / "prototype_runs/basic_full/model_best.pt"
FPS = 30
COUNT = 6573
WARMUP = 16


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_polarity(probability, polarity):
    return torch.where(polarity[:, None, None, None], 1 - probability, probability)


def pixel_metrics(prediction, target):
    return float(np.not_equal(prediction, target).mean()), _mean_binary_iou(prediction, target)


def summarize(rows, warmup=WARMUP):
    scored = rows[warmup:]
    if not scored:
        raise ValueError("No scored frames after warmup")
    keys = [k for k in rows[0] if k.endswith(("_error", "_iou", "_gate"))]
    result = {k: float(np.mean([r[k] for r in scored])) for k in keys}
    errors = np.array([r["rollout_error"] for r in scored])
    result.update(scored_frames=len(scored), rollout_p95=float(np.quantile(errors, .95)),
                  polarity_accuracy=float(np.mean([r["polarity_correct"] for r in scored])),
                  accumulation_gap=result["rollout_error"] - result["teacher_error"])
    if "memory_error" in result:
        result["full_error_reduction_fraction"] = 1 - result["rollout_error"] / result["memory_error"]
        result["full_wins_fraction"] = float(np.mean([r["rollout_error"] < r["memory_error"] for r in scored]))
    return result


def font(size=20):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


class VideoWriter:
    def __init__(self, path, size):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([
            _imageio_ffmpeg().get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
            "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2",
            "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("Video encoder failed")


def panel(pixels, label, detail="", width=512):
    image = Image.fromarray(pixels.astype(np.uint8) * 255).convert("RGB")
    image = image.resize((width, width * 3 // 4), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, image.height + 58), "#111820")
    canvas.paste(image, (0, 58))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 5), label, fill="white", font=font(20))
    draw.text((12, 32), detail, fill="#bac4cd", font=font(16))
    return canvas


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    dataset = FrameDataset(DATA, 384, 512, .5)
    if len(dataset) != COUNT:
        raise ValueError(f"Expected {COUNT} source frames; got {len(dataset)}")
    checkpoint_hash = sha256(AE)
    cache_path = ROOT / "prototype_data/cache/blog_canonical.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["autoencoder_sha256"] != checkpoint_hash or len(cached["raw_latents"]) != COUNT:
            raise ValueError("Canonical cache provenance mismatch")
        return cached
    print("Encoding the source once; no model optimization", flush=True)
    autoencoder, metadata = load_autoencoder(AE, torch.device("cuda"))
    raw, polarities = encode_canonical_sequence(autoencoder, metadata, DATA, torch.device("cuda"), 8,
                                               polarity_tracking_method="temporal", polarity_switch_penalty=.05)
    result = {"raw_latents": raw, "polarities": polarities.bool(), "autoencoder_sha256": checkpoint_hash,
              "frame_count": COUNT, "fps": FPS, "image_size": metadata["image_size"],
              "input_threshold": metadata["input_threshold"], "polarity_tracking_method": "temporal",
              "polarity_switch_penalty": .05}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    return result


def checkpoint_path(budget):
    corrected = ROOT / "prototype_runs" / f"anchors_{budget:03d}_polarity" / "model_best.pt"
    if budget != 220 or corrected.is_file():
        return corrected
    return ROOT / "prototype_runs/anchors_220/model_best.pt"


def render(budget, cache, polarity_donor=None):
    path = checkpoint_path(budget)
    suffix = "_shared_polarity" if polarity_donor else (
        "_polarity" if budget == 220 and path.parent.name == "anchors_220_polarity" else ""
    )
    tag = f"anchors_{budget:03d}" + suffix
    folder = WORK / tag
    identity = {"checkpoint_sha256": sha256(path), "autoencoder_sha256": cache["autoencoder_sha256"],
                "polarity_donor_sha256": sha256(polarity_donor) if polarity_donor else None}
    summary_path = folder / "summary.json"
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if any(result[k] != v for k, v in identity.items()):
            raise ValueError("Completed export uses different weights; choose a different output folder")
        print(f"Reusing verified completed inference: {tag}", flush=True)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("*.mp4")):
        raise FileExistsError(f"Partial render exists at {folder}; preserve or move it before retrying")
    device = torch.device("cuda")
    cp = torch.load(path, map_location="cpu", weights_only=False)
    assert cp["image_size"] == cache["image_size"] and cp["input_threshold"] == cache["input_threshold"]
    assert cp["rollout_warmup_frames"] == WARMUP and cp["canonicalize_polarity"]
    assert cp["polarity_tracking_method"] == "temporal" and cp["polarity_switch_penalty"] == .05
    model = _load_model(cp, device)
    donor_model = None
    if polarity_donor:
        donor_cp = torch.load(polarity_donor, map_location="cpu", weights_only=False)
        donor_model = _load_model(donor_cp, device)
        assert donor_model.polarity_spline_logits is not None
    autoencoder, _ = load_autoencoder(AE, device)
    truth = (cache["raw_latents"] - cp["latent_mean"]) / cp["latent_standard_deviation"]
    prediction = torch.empty_like(truth)
    teacher = torch.empty_like(truth)
    prediction[:WARMUP] = truth[:WARMUP]
    teacher[:WARMUP] = truth[:WARMUP]
    gates = np.zeros(COUNT)
    started = time.monotonic()
    with torch.inference_mode():
        history = truth[:WARMUP].unsqueeze(0).to(device)
        for i in range(WARMUP, COUNT):
            timestamp = torch.tensor([i / (COUNT - 1)], device=device)
            next_latent, extra = model(history, timestamp)
            prediction[i] = next_latent[0].cpu()
            gates[i] = extra["spatial_memory_gate"].mean().item()
            history = torch.cat((history[:, 1:], next_latent[:, None]), dim=1)
            if i % 600 == 0:
                print(f"{tag}: rollout {i}/{COUNT}, {time.monotonic()-started:.0f}s", flush=True)
        for start in range(WARMUP, COUNT, 8):
            end = min(COUNT, start + 8)
            histories = torch.stack([truth[i-WARMUP:i] for i in range(start, end)]).to(device)
            times = torch.arange(start, end, device=device).float() / (COUNT - 1)
            teacher[start:end] = model(histories, times)[0].cpu()
            if start % 600 == WARMUP:
                print(f"{tag}: teacher {start}/{COUNT}", flush=True)
        times = torch.arange(COUNT, device=device).float() / (COUNT - 1)
        native_polarities = model.predict_polarity(times)[:, 0].cpu() >= 0
        predicted_polarities = (donor_model or model).predict_polarity(times)[:, 0].cpu() >= 0
    restore = predicted_polarities.clone()
    restore[:WARMUP] = cache["polarities"][:WARMUP]
    dataset = FrameDataset(DATA, *cp["image_size"], cp["input_threshold"])
    loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
    writer = VideoWriter(folder / "comparison.mp4", (1024, 884))
    rollout_writer = VideoWriter(folder / "rollout.mp4", (512, 384))
    memory_writer = VideoWriter(folder / "memory.mp4", (512, 384)) if budget else None
    source_writer = VideoWriter(WORK / "source.mp4", (512, 384)) if not (WORK / "source.mp4").exists() else None
    rows = []
    mean, std = cp["latent_mean"].to(device), cp["latent_standard_deviation"].to(device)
    try:
        with torch.inference_mode():
            for start, (targets, _) in zip(range(0, COUNT, 8), loader):
                stop = start + len(targets)
                sl = slice(start, stop)
                latents = [teacher[sl].to(device), prediction[sl].to(device)]
                times = torch.arange(start, stop, device=device).float() / (COUNT - 1)
                if budget:
                    latents.append(memory_only_latents(model, times))
                probs = torch.sigmoid(autoencoder.decode(torch.cat(latents) * std + mean, cp["image_size"]))
                pieces = probs.split(len(targets))
                decoded = [restore_polarity(p, restore[sl].to(device)) >= cp["activation_threshold"] for p in pieces[:2]]
                if budget:
                    # Memory receives no source frames, including during the excluded warmup.
                    decoded.append(restore_polarity(pieces[2], predicted_polarities[sl].to(device)) >= cp["activation_threshold"])
                arrays = [p[:, 0].cpu().numpy() for p in decoded]
                target_arrays = targets[:, 0].numpy().astype(bool)
                distances = ((times[:, None] - model.anchor_times[None]).abs().min(1).values.cpu().numpy()
                             * (COUNT - 1) / FPS) if budget else np.full(len(targets), -1.)
                for j, target in enumerate(target_arrays):
                    i = start + j
                    row = {"frame": i, "seconds": i / FPS, "target_polarity": int(cache["polarities"][i]),
                           "predicted_polarity": int(restore[i]), "polarity_correct": int(restore[i] == cache["polarities"][i]),
                           "effective_gate": float(gates[i]), "nearest_anchor_seconds": float(distances[j])}
                    for kind, pixels in zip(("teacher", "rollout", "memory"), arrays):
                        row[kind + "_error"], row[kind + "_iou"] = pixel_metrics(pixels[j], target)
                    # Oracle-polarity shape diagnostic is a metric only, never a rendered correction.
                    row["oracle_polarity_rollout_error"] = row["rollout_error"] if row["polarity_correct"] else 1 - row["rollout_error"]
                    rows.append(row)
                    canvas = Image.new("RGB", (1024, 884), "#111820")
                    stamp = f"t = {i/FPS:06.2f}s | frame {i}"
                    p = [panel(target, "Source", stamp), panel(arrays[0][j], "Teacher-forced", "True history supplied every step"),
                         panel(arrays[1][j], f"Full rollout | {budget} anchors", "Source warmup" if i < WARMUP else "Own predictions only since frame 16")]
                    p.append(panel(arrays[2][j], "Memory only", "Time + same anchors + decoder; no history") if budget else
                             panel(np.zeros_like(target), "No memory baseline", "Zero anchors: memory-only is undefined"))
                    for tile, xy in zip(p, ((0, 0), (512, 0), (0, 442), (512, 442))):
                        canvas.paste(tile, xy)
                    writer.write(canvas)
                    rollout_writer.write(Image.fromarray(arrays[1][j].astype(np.uint8)*255).convert("RGB"))
                    if memory_writer:
                        memory_writer.write(Image.fromarray(arrays[2][j].astype(np.uint8)*255).convert("RGB"))
                    if source_writer:
                        source_writer.write(Image.fromarray(target.astype(np.uint8)*255).convert("RGB"))
                if start % 600 == 0:
                    print(f"{tag}: decode/video {stop}/{COUNT}", flush=True)
    finally:
        for stream in (writer, rollout_writer, memory_writer, source_writer):
            if stream:
                stream.close()
    write_csv(folder / "metrics.csv", rows)
    groups = {}
    for name, parameter in model.named_parameters():
        key = "anchors" if name == "memory_tokens" else "temporal_unet" if name.split('.')[0] in {
            "encoder_high", "encoder_middle", "bottleneck", "decoder_middle", "decoder_high"} else "time_features" if name.startswith(("time_encoder", "time_to_bottleneck")) else "heads_and_gates"
        groups[key] = groups.get(key, 0) + parameter.numel()
    config = json.loads((ROOT / f"prototype_runs/anchors_{budget:03d}/config.json").read_text())
    result = {**identity, "checkpoint": str(path.relative_to(ROOT)), "anchor_count": budget, "tag": tag,
              "checkpoint_epoch": cp["epoch"], "image_size": cp["image_size"], "fps": FPS, "frame_count": COUNT,
              "warmup_frames": WARMUP, "native_polarity_accuracy": float((native_polarities == cache["polarities"]).float().mean()),
              "polarity_handling": "existing trained spline reused" if polarity_donor else (
                  "separately calibrated 96-knot spline" if cp["model_kwargs"].get("polarity_knot_count") == 96 else "checkpoint native"),
              "polarity_donor": str(polarity_donor.relative_to(ROOT)) if polarity_donor else None,
              "training": {k: config.get(k) for k in ("batch_size", "learning_rate", "epochs", "seed", "burn_in_steps", "rollout_steps", "freeze_memory_epochs")},
              "parameter_groups": groups, "parameters": sum(p.numel() for p in model.parameters()),
              "autoencoder_parameters": sum(p.numel() for p in autoencoder.parameters()),
              "anchor_frames": cp["anchor_initialization_indices"], "metrics": summarize(rows),
              "elapsed_seconds": time.monotonic()-started, "torch_version": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "evaluation": "fresh full-resolution inference; videos are H.264 previews"}
    write_json(summary_path, result)
    (folder / "report.md").write_text("# Blog inference export\n\nNo training or checkpoint changes. All 6,573 frames; scores exclude the first 16.\n\n"
        + "Memory-only uses jointly trained anchors, not an independently trained baseline. Teacher uses true history; rollout does not after warmup.\n\n"
        + "Oracle-polarity error is a separately labelled shape diagnostic using target polarity, not an autonomous score.\n\n```json\n"
        + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"tag": tag, **result["metrics"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", nargs="+", type=int, default=[220, 32, 55, 110, 16, 0])
    parser.add_argument("--polarity-donor", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache = prepare()
    for budget in args.anchors:
        render(budget, cache, args.polarity_donor.resolve() if args.polarity_donor else None)


if __name__ == "__main__":
    main()


