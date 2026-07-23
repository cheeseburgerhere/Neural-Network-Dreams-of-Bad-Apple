from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import FrameDataset
from .models import PixelActivationAutoencoder, build_model
from .training import resolve_device


class ConvGRUCell(nn.Module):
    """Minimal convolutional GRU that preserves the latent spatial grid."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        combined_channels = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            combined_channels, hidden_channels * 2, kernel_size=3, padding=1
        )
        self.candidate = nn.Conv2d(
            combined_channels, hidden_channels, kernel_size=3, padding=1
        )

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor | None
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                inputs.shape[0],
                self.hidden_channels,
                inputs.shape[2],
                inputs.shape[3],
                device=inputs.device,
                dtype=inputs.dtype,
            )
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class LatentAutoregressor(nn.Module):
    """Predicts the next normalized latent from the previous predicted latent."""

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int = 64,
        max_residual_step: float = 0.5,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.max_residual_step = max_residual_step
        self.recurrent = ConvGRUCell(latent_channels, hidden_channels)
        self.delta_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def step(
        self,
        current_latent: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self.recurrent(current_latent, hidden)
        residual = self.max_residual_step * torch.tanh(
            self.delta_head(next_hidden)
        )
        return current_latent + residual, next_hidden


class LatentWindowDataset(Dataset):
    def __init__(self, latents: torch.Tensor, sequence_length: int) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if len(latents) <= sequence_length:
            raise ValueError("sequence_length must be shorter than the video")
        self.latents = latents
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.latents) - self.sequence_length

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.latents[index : index + self.sequence_length + 1]


@dataclass
class AutoregressiveTrainingConfig:
    autoencoder_checkpoint: Path
    frame_dir: Path
    run_dir: Path
    hidden_channels: int = 64
    max_residual_step: float = 0.5
    sequence_length: int = 16
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-3
    rollout_loss_weight: float = 0.1
    rollout_warmup_frames: int = 16
    seed: int = 7
    device: str = "auto"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_autoencoder(
    checkpoint_path: Path, device: torch.device
) -> tuple[PixelActivationAutoencoder, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model, _ = build_model(
        checkpoint["model_name"], **checkpoint["model_kwargs"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def encode_sequence(
    autoencoder: PixelActivationAutoencoder,
    autoencoder_checkpoint: dict,
    frame_dir: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    height, width = autoencoder_checkpoint["image_size"]
    dataset = FrameDataset(
        frame_dir,
        height=height,
        width=width,
        input_threshold=autoencoder_checkpoint["input_threshold"],
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    encoded: list[torch.Tensor] = []
    with torch.inference_mode():
        for frames, _ in loader:
            latent, _ = autoencoder.encode(frames.to(device))
            encoded.append(latent.cpu())
    return torch.cat(encoded, dim=0)


def normalize_latents(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    standard_deviation = latents.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-5)
    return (latents - mean) / standard_deviation, mean, standard_deviation


def _teacher_forced_loss(
    model: LatentAutoregressor,
    sequences: torch.Tensor,
    loss_function: nn.Module,
) -> torch.Tensor:
    hidden = None
    loss = torch.zeros((), device=sequences.device)
    for step_index in range(sequences.shape[1] - 1):
        predicted, hidden = model.step(sequences[:, step_index], hidden)
        loss = loss + loss_function(predicted, sequences[:, step_index + 1])
    return loss / (sequences.shape[1] - 1)


def _free_rollout_loss(
    model: LatentAutoregressor,
    sequences: torch.Tensor,
    loss_function: nn.Module,
) -> torch.Tensor:
    hidden = None
    current = sequences[:, 0]
    loss = torch.zeros((), device=sequences.device)
    for step_index in range(1, sequences.shape[1]):
        current, hidden = model.step(current, hidden)
        loss = loss + loss_function(current, sequences[:, step_index])
    return loss / (sequences.shape[1] - 1)


def rollout_latents(
    model: LatentAutoregressor,
    seed_latents: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    if len(seed_latents) < 1:
        raise ValueError("At least one seed latent is required")
    if len(seed_latents) > frame_count:
        raise ValueError("Seed sequence cannot be longer than the rollout")

    predictions = [seed_latents]
    hidden = None
    current = seed_latents[-1:]
    with torch.inference_mode():
        for seed_index in range(len(seed_latents) - 1):
            _, hidden = model.step(
                seed_latents[seed_index : seed_index + 1], hidden
            )
        for _ in range(len(seed_latents), frame_count):
            current, hidden = model.step(current, hidden)
            predictions.append(current)
    return torch.cat(predictions, dim=0)


def _rollout_metrics(
    model: LatentAutoregressor,
    latents: torch.Tensor,
    device: torch.device,
    warmup_frames: int,
) -> dict[str, float]:
    model.eval()
    warmup_frames = min(max(1, warmup_frames), len(latents) - 1)
    predicted = rollout_latents(
        model,
        latents[:warmup_frames].to(device),
        frame_count=len(latents),
    ).cpu()
    per_frame = (predicted - latents).square().mean(dim=(1, 2, 3))
    return {
        "rollout_mse": per_frame.mean().item(),
        "final_frame_mse": per_frame[-1].item(),
        "peak_frame_mse": per_frame.max().item(),
    }


def _save_checkpoint(
    path: Path,
    model: LatentAutoregressor,
    config: AutoregressiveTrainingConfig,
    autoencoder_checkpoint: dict,
    latent_mean: torch.Tensor,
    latent_standard_deviation: torch.Tensor,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_type": "latent_autoregressor",
            "model_kwargs": {
                "latent_channels": model.latent_channels,
                "hidden_channels": model.hidden_channels,
                "max_residual_step": model.max_residual_step,
            },
            "autoencoder_checkpoint": str(
                config.autoencoder_checkpoint.resolve()
            ),
            "image_size": autoencoder_checkpoint["image_size"],
            "input_threshold": autoencoder_checkpoint["input_threshold"],
            "activation_threshold": autoencoder_checkpoint[
                "activation_threshold"
            ],
            "rollout_warmup_frames": config.rollout_warmup_frames,
            "latent_mean": latent_mean,
            "latent_standard_deviation": latent_standard_deviation,
            "epoch": epoch,
            "metrics": metrics,
            "state_dict": model.state_dict(),
        },
        path,
    )


def train_autoregressor(config: AutoregressiveTrainingConfig) -> Path:
    _set_seed(config.seed)
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    autoencoder, autoencoder_checkpoint = load_autoencoder(
        config.autoencoder_checkpoint, device
    )
    print("Encoding the source sequence with the frozen autoencoder...")
    raw_latents = encode_sequence(
        autoencoder,
        autoencoder_checkpoint,
        config.frame_dir,
        device,
        batch_size=config.batch_size,
    )
    latents, latent_mean, latent_standard_deviation = normalize_latents(
        raw_latents
    )
    latent_channels = latents.shape[1]
    dataset = LatentWindowDataset(latents, config.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = LatentAutoregressor(
        latent_channels=latent_channels,
        hidden_channels=config.hidden_channels,
        max_residual_step=config.max_residual_step,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()

    serializable_config = asdict(config)
    for key in ("autoencoder_checkpoint", "frame_dir", "run_dir"):
        serializable_config[key] = str(Path(serializable_config[key]).resolve())
    serializable_config["resolved_device"] = str(device)
    serializable_config["latent_shape"] = list(latents.shape[1:])
    serializable_config["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    (config.run_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )

    print(
        f"Training latent autoregressor on {device}: {len(latents)} frames, "
        f"{len(dataset)} windows, {serializable_config['parameter_count']:,} "
        "parameters"
    )
    history: list[dict] = []
    best_rollout_mse = float("inf")
    best_checkpoint = config.run_dir / "model_best.pt"

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        training_loss = 0.0
        teacher_loss_total = 0.0
        rollout_loss_total = 0.0
        for sequences in loader:
            sequences = sequences.to(device)
            optimizer.zero_grad(set_to_none=True)
            teacher_loss = _teacher_forced_loss(
                model, sequences, loss_function
            )
            rollout_loss = _free_rollout_loss(
                model, sequences, loss_function
            )
            loss = (
                teacher_loss
                + config.rollout_loss_weight * rollout_loss
            )
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            teacher_loss_total += teacher_loss.item()
            rollout_loss_total += rollout_loss.item()

        metrics = _rollout_metrics(
            model, latents, device, config.rollout_warmup_frames
        )
        row = {
            "epoch": epoch,
            "training_loss": training_loss / len(loader),
            "teacher_forced_loss": teacher_loss_total / len(loader),
            "window_rollout_loss": rollout_loss_total / len(loader),
            **metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | teacher {row['teacher_forced_loss']:.5f} | "
            f"window rollout {row['window_rollout_loss']:.5f} | "
            f"full rollout {row['rollout_mse']:.5f} | "
            f"final {row['final_frame_mse']:.5f} | {row['seconds']:.1f}s"
        )

        _save_checkpoint(
            config.run_dir / "model_last.pt",
            model,
            config,
            autoencoder_checkpoint,
            latent_mean,
            latent_standard_deviation,
            epoch,
            metrics,
        )
        if metrics["rollout_mse"] < best_rollout_mse:
            best_rollout_mse = metrics["rollout_mse"]
            _save_checkpoint(
                best_checkpoint,
                model,
                config,
                autoencoder_checkpoint,
                latent_mean,
                latent_standard_deviation,
                epoch,
                metrics,
            )

        (config.run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    return best_checkpoint
