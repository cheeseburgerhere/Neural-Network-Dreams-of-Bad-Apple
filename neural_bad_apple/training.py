from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import FrameDataset
from .models import build_model


@dataclass
class TrainingConfig:
    frame_dir: Path
    run_dir: Path
    model_name: str = "basic"
    height: int = 192
    width: int = 256
    input_threshold: float = 0.5
    activation_threshold: float = 0.5
    base_channels: int = 16
    latent_channels: int = 64
    epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 1e-3
    validation_every: int = 10
    seed: int = 7
    device: str = "auto"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_dataset(dataset: FrameDataset, validation_every: int):
    if validation_every < 2:
        raise ValueError("validation_every must be at least 2")
    validation_indices = [
        index for index in range(len(dataset)) if index % validation_every == 0
    ]
    training_indices = [
        index for index in range(len(dataset)) if index % validation_every != 0
    ]
    if not training_indices or not validation_indices:
        raise RuntimeError("Dataset is too small for the requested validation split")
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def _mean_binary_iou(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> float:
    predictions = probabilities >= threshold
    truth = targets >= 0.5
    scores: list[torch.Tensor] = []
    for value in (False, True):
        prediction_class = predictions == value
        truth_class = truth == value
        intersection = (prediction_class & truth_class).sum()
        union = (prediction_class | truth_class).sum()
        if union:
            scores.append(intersection.float() / union.float())
    return torch.stack(scores).mean().item()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    activation_threshold: float,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    accuracy_total = 0.0
    iou_total = 0.0
    batches = 0
    with torch.inference_mode():
        for targets, _ in loader:
            targets = targets.to(device)
            logits, _ = model(targets)
            loss_total += loss_function(logits, targets).item()
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= activation_threshold
            accuracy_total += (predictions == (targets >= 0.5)).float().mean().item()
            iou_total += _mean_binary_iou(
                probabilities, targets, activation_threshold
            )
            batches += 1
    return {
        "loss": loss_total / batches,
        "pixel_accuracy": accuracy_total / batches,
        "mean_binary_iou": iou_total / batches,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    model_kwargs: dict,
    config: TrainingConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "image_size": [config.height, config.width],
            "input_threshold": config.input_threshold,
            "activation_threshold": config.activation_threshold,
            "epoch": epoch,
            "metrics": metrics,
            "state_dict": model.state_dict(),
        },
        path,
    )


def train(config: TrainingConfig) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    dataset = FrameDataset(
        config.frame_dir,
        height=config.height,
        width=config.width,
        input_threshold=config.input_threshold,
    )
    training_set, validation_set = _split_dataset(
        dataset, config.validation_every
    )
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    training_loader = DataLoader(training_set, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)

    model, model_kwargs = build_model(
        config.model_name,
        base_channels=config.base_channels,
        latent_channels=config.latent_channels,
    )
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    serializable_config = asdict(config)
    serializable_config["frame_dir"] = str(config.frame_dir.resolve())
    serializable_config["run_dir"] = str(config.run_dir.resolve())
    serializable_config["resolved_device"] = str(device)
    serializable_config["parameter_count"] = parameter_count
    (config.run_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )

    print(
        f"Training {config.model_name} on {device}: "
        f"{len(training_set)} train / {len(validation_set)} validation frames, "
        f"{parameter_count:,} parameters"
    )
    history: list[dict] = []
    best_loss = float("inf")
    best_checkpoint = config.run_dir / "model_best.pt"

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        training_loss = 0.0
        for targets, _ in training_loader:
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(targets)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        metrics = _evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            config.activation_threshold,
        )
        row = {
            "epoch": epoch,
            "training_loss": training_loss / len(training_loader),
            "validation_loss": metrics["loss"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "mean_binary_iou": metrics["mean_binary_iou"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | train {row['training_loss']:.4f} | "
            f"val {row['validation_loss']:.4f} | "
            f"IoU {row['mean_binary_iou']:.4f} | {row['seconds']:.1f}s"
        )

        _save_checkpoint(
            config.run_dir / "model_last.pt",
            model,
            config.model_name,
            model_kwargs,
            config,
            epoch,
            metrics,
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            _save_checkpoint(
                best_checkpoint,
                model,
                config.model_name,
                model_kwargs,
                config,
                epoch,
                metrics,
            )

        (config.run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    return best_checkpoint
