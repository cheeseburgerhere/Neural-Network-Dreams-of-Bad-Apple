from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .autoregressive import (
    encode_sequence,
    load_autoencoder,
    normalize_latents,
)
from .data import FrameDataset
from .training import resolve_device


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class TemporalBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv3d(
                input_channels, output_channels, kernel_size=3, padding=1
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                output_channels, output_channels, kernel_size=3, padding=1
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class HybridTemporalMemoryModel(nn.Module):
    """Causal temporal U-Net fused with time-addressed learned memories."""

    def __init__(
        self,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        base_channels: int = 16,
        memory_token_count: int = 8,
        fourier_frequencies: int = 6,
        max_residual_step: float = 0.5,
        memory_temperature: float = 1.0,
        use_polarity_head: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.base_channels = base_channels
        self.memory_token_count = memory_token_count
        self.fourier_frequencies = fourier_frequencies
        self.max_residual_step = max_residual_step
        self.memory_temperature = memory_temperature
        self.use_polarity_head = use_polarity_head

        self.encoder_high = TemporalBlock(latent_channels, base_channels)
        self.encoder_middle = TemporalBlock(base_channels, base_channels * 2)
        self.bottleneck = TemporalBlock(base_channels * 2, base_channels * 4)
        self.decoder_middle = TemporalBlock(
            base_channels * 6, base_channels * 2
        )
        self.decoder_high = TemporalBlock(
            base_channels * 3, base_channels
        )
        self.motion_head = nn.Conv2d(
            base_channels, latent_channels, kernel_size=3, padding=1
        )

        time_feature_count = 1 + fourier_frequencies * 2
        time_hidden = max(64, base_channels * 4)
        self.time_encoder = nn.Sequential(
            nn.Linear(time_feature_count, time_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(time_hidden, time_hidden),
            nn.SiLU(inplace=True),
        )
        self.time_to_bottleneck = nn.Linear(
            time_hidden, base_channels * 4
        )
        self.time_to_memory_logits = nn.Linear(
            time_hidden, memory_token_count
        )
        self.time_to_memory_gate = nn.Linear(time_hidden, 1)
        nn.init.constant_(self.time_to_memory_gate.bias, -2.0)
        self.time_to_polarity = (
            nn.Linear(time_hidden, 1) if use_polarity_head else None
        )

        self.memory_tokens = nn.Parameter(
            torch.zeros(
                memory_token_count,
                latent_channels,
                latent_height,
                latent_width,
            )
        )
        nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)

    def _time_features(self, normalized_time: torch.Tensor) -> torch.Tensor:
        normalized_time = normalized_time.reshape(-1, 1)
        features = [normalized_time]
        for frequency_index in range(self.fourier_frequencies):
            frequency = 2.0**frequency_index
            angle = normalized_time * (2.0 * math.pi * frequency)
            features.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(features, dim=1)

    def initialize_memory(
        self,
        normalized_latents: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        if indices is None:
            indices = torch.linspace(
                0,
                len(normalized_latents) - 1,
                steps=self.memory_token_count,
            ).round().long()
        if len(indices) != self.memory_token_count:
            raise ValueError("One initialization index is required per memory")
        with torch.no_grad():
            self.memory_tokens.copy_(
                normalized_latents[indices.to(normalized_latents.device)]
            )

    def address_memory(
        self, normalized_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_state = self.time_encoder(
            self._time_features(normalized_time)
        )
        memory_weights = torch.softmax(
            self.time_to_memory_logits(time_state) / self.memory_temperature,
            dim=1,
        )
        memory_gate = torch.sigmoid(
            self.time_to_memory_gate(time_state)
        )
        return time_state, memory_weights, memory_gate

    def predict_polarity(self, normalized_time: torch.Tensor) -> torch.Tensor:
        if self.time_to_polarity is None:
            return torch.full(
                (normalized_time.numel(), 1),
                -20.0,
                device=normalized_time.device,
                dtype=normalized_time.dtype,
            )
        time_state = self.time_encoder(
            self._time_features(normalized_time)
        )
        return self.time_to_polarity(time_state)

    def forward(
        self,
        latent_history: torch.Tensor,
        normalized_time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if latent_history.ndim != 5:
            raise ValueError("latent_history must have shape [B, T, C, H, W]")
        if latent_history.shape[2:] != (
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        ):
            raise ValueError("latent_history has the wrong latent grid shape")

        time_state, memory_weights, memory_gate = self.address_memory(
            normalized_time
        )
        features = latent_history.permute(0, 2, 1, 3, 4)
        high = self.encoder_high(features)
        middle_input = F.avg_pool3d(
            high, kernel_size=(2, 1, 1), stride=(2, 1, 1)
        )
        middle = self.encoder_middle(middle_input)
        bottleneck_input = F.avg_pool3d(
            middle, kernel_size=(2, 1, 1), stride=(2, 1, 1)
        )
        bottleneck = self.bottleneck(bottleneck_input)
        time_bias = self.time_to_bottleneck(time_state)[
            :, :, None, None, None
        ]
        bottleneck = bottleneck + time_bias

        up_middle = F.interpolate(
            bottleneck,
            size=middle.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        up_middle = self.decoder_middle(
            torch.cat((up_middle, middle), dim=1)
        )
        up_high = F.interpolate(
            up_middle,
            size=high.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        up_high = self.decoder_high(torch.cat((up_high, high), dim=1))

        motion_delta = self.motion_head(up_high[:, :, -1])
        current_latent = latent_history[:, -1]
        motion_candidate = current_latent + self.max_residual_step * torch.tanh(
            motion_delta
        )

        memory_candidate = torch.einsum(
            "bm,mchw->bchw", memory_weights, self.memory_tokens
        )
        memory_gate = memory_gate[:, :, None, None]
        next_latent = (
            (1.0 - memory_gate) * motion_candidate
            + memory_gate * memory_candidate
        )
        polarity_logits = (
            self.time_to_polarity(time_state)
            if self.time_to_polarity is not None
            else torch.full_like(memory_gate[:, :, 0, 0], -20.0)
        )
        return next_latent, {
            "memory_weights": memory_weights,
            "memory_gate": memory_gate[:, :, 0, 0],
            "motion_candidate": motion_candidate,
            "memory_candidate": memory_candidate,
            "polarity_logits": polarity_logits,
        }


class HybridWindowDataset(Dataset):
    def __init__(
        self,
        latents: torch.Tensor,
        history_length: int,
        rollout_steps: int,
        polarities: torch.Tensor | None = None,
    ) -> None:
        if history_length < 4:
            raise ValueError("history_length must be at least 4")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        if len(latents) < history_length + rollout_steps:
            raise ValueError("Video is too short for this temporal window")
        self.latents = latents
        self.history_length = history_length
        self.rollout_steps = rollout_steps
        self.polarities = (
            polarities
            if polarities is not None
            else torch.zeros(len(latents), dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.latents) - self.history_length - self.rollout_steps + 1

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = self.history_length + self.rollout_steps
        sequence = self.latents[index : index + length]
        target_indices = torch.arange(
            index + self.history_length,
            index + length,
            dtype=torch.float32,
        )
        times = target_indices / max(1, len(self.latents) - 1)
        target_polarities = self.polarities[
            index + self.history_length : index + length
        ]
        return sequence, times, target_polarities


@dataclass
class HybridTrainingConfig:
    autoencoder_checkpoint: Path
    frame_dir: Path
    run_dir: Path
    history_length: int = 16
    minimum_rollout_steps: int = 4
    rollout_steps: int = 16
    base_channels: int = 16
    memory_token_count: int = 8
    fourier_frequencies: int = 6
    max_residual_step: float = 0.5
    memory_temperature: float = 0.5
    memory_entropy_weight: float = 1e-3
    polarity_loss_weight: float = 0.2
    canonicalize_polarity: bool = True
    polarity_tracking_method: str = "temporal"
    polarity_switch_penalty: float = 0.05
    scene_cut_minimum_distance: int = 15
    memory_initialization_indices: list[int] = field(default_factory=list)
    latent_noise_standard_deviation: float = 0.03
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 5e-4
    seed: int = 7
    device: str = "auto"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_frame_polarity(frames: torch.Tensor) -> torch.Tensor:
    """Return 1 for white-background frames and 0 for black-background frames."""
    top = frames[:, :, 0, :].flatten(1)
    bottom = frames[:, :, -1, :].flatten(1)
    left = frames[:, :, 1:-1, 0].flatten(1)
    right = frames[:, :, 1:-1, -1].flatten(1)
    border = torch.cat((top, bottom, left, right), dim=1)
    return (border.mean(dim=1) >= 0.5).float()


def track_frame_polarity(
    frames: torch.Tensor,
    switch_penalty: float = 0.05,
    initial_polarity: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Choose the temporally smoother orientation of each binary frame.

    The border detector only anchors the first frame. Later frames switch
    orientation when comparing against the inverse is clearly smoother than
    keeping the current orientation. The penalty supplies hysteresis around
    ambiguous cuts.
    """
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("frames must be a non-empty TCHW tensor")
    if switch_penalty < 0:
        raise ValueError("switch_penalty must be non-negative")

    polarities = torch.empty(
        len(frames), device=frames.device, dtype=frames.dtype
    )
    if initial_polarity is None:
        polarities[0] = detect_frame_polarity(frames[:1])[0]
    else:
        polarities[0] = torch.as_tensor(
            initial_polarity, device=frames.device, dtype=frames.dtype
        )

    for frame_index in range(1, len(frames)):
        previous = frames[frame_index - 1]
        current = frames[frame_index]
        same_orientation_error = (current - previous).abs().mean()
        inverted_orientation_error = (1.0 - current - previous).abs().mean()
        should_switch = (
            inverted_orientation_error + switch_penalty
            < same_orientation_error
        )
        polarities[frame_index] = torch.where(
            should_switch,
            1.0 - polarities[frame_index - 1],
            polarities[frame_index - 1],
        )
    return polarities


def encode_canonical_sequence(
    autoencoder: nn.Module,
    autoencoder_checkpoint: dict,
    frame_dir: Path,
    device: torch.device,
    batch_size: int,
    polarity_tracking_method: str = "temporal",
    polarity_switch_penalty: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    if polarity_tracking_method not in {"border", "temporal"}:
        raise ValueError(
            "polarity_tracking_method must be 'border' or 'temporal'"
        )
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
    polarities: list[torch.Tensor] = []
    previous_tracking_frame: torch.Tensor | None = None
    previous_polarity: torch.Tensor | None = None
    with torch.inference_mode():
        for frames, _ in loader:
            frames = frames.to(device)
            if polarity_tracking_method == "border":
                polarity = detect_frame_polarity(frames)
            else:
                tracking_frames = F.interpolate(
                    frames,
                    size=(min(height, 48), min(width, 64)),
                    mode="area",
                ).cpu()
                if previous_tracking_frame is None:
                    initial_polarity = detect_frame_polarity(frames[:1])[
                        0
                    ].cpu()
                    polarity = track_frame_polarity(
                        tracking_frames,
                        switch_penalty=polarity_switch_penalty,
                        initial_polarity=initial_polarity,
                    )
                else:
                    tracking_with_context = torch.cat(
                        (previous_tracking_frame, tracking_frames), dim=0
                    )
                    polarity = track_frame_polarity(
                        tracking_with_context,
                        switch_penalty=polarity_switch_penalty,
                        initial_polarity=previous_polarity,
                    )[1:]
                previous_tracking_frame = tracking_frames[-1:].clone()
                previous_polarity = polarity[-1].clone()
                polarity = polarity.to(device)
            canonical_frames = torch.where(
                polarity[:, None, None, None] > 0.5,
                1.0 - frames,
                frames,
            )
            latent, _ = autoencoder.encode(canonical_frames)
            encoded.append(latent.cpu())
            polarities.append(polarity.cpu())
    return torch.cat(encoded, dim=0), torch.cat(polarities, dim=0)


def select_scene_memory_indices(
    normalized_latents: torch.Tensor,
    memory_token_count: int,
    minimum_distance: int,
) -> torch.Tensor:
    """Select high-change frames while preventing multiple tokens per cut."""
    changes = (
        normalized_latents[1:] - normalized_latents[:-1]
    ).square().mean(dim=(1, 2, 3))
    ranked = (torch.argsort(changes, descending=True) + 1).tolist()
    selected = [0]
    for candidate in ranked:
        if all(
            abs(candidate - existing) >= minimum_distance
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) == memory_token_count:
            break

    if len(selected) < memory_token_count:
        uniform = torch.linspace(
            0,
            len(normalized_latents) - 1,
            steps=memory_token_count,
        ).round().long().tolist()
        for candidate in uniform:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == memory_token_count:
                break
    return torch.tensor(sorted(selected), dtype=torch.long)


def rollout_hybrid_latents(
    model: HybridTemporalMemoryModel,
    seed_latents: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    history_length = len(seed_latents)
    if history_length < 4:
        raise ValueError("At least four context latents are required")
    if history_length >= frame_count:
        raise ValueError("Context must be shorter than the output sequence")

    predictions = [seed_latents]
    history = seed_latents.unsqueeze(0)
    with torch.inference_mode():
        for target_index in range(history_length, frame_count):
            normalized_time = torch.tensor(
                [target_index / max(1, frame_count - 1)],
                device=history.device,
                dtype=history.dtype,
            )
            predicted, _ = model(history, normalized_time)
            predictions.append(predicted)
            history = torch.cat(
                (history[:, 1:], predicted.unsqueeze(1)), dim=1
            )
    return torch.cat(predictions, dim=0)


def _rollout_metrics(
    model: HybridTemporalMemoryModel,
    latents: torch.Tensor,
    polarities: torch.Tensor,
    device: torch.device,
    history_length: int,
) -> dict[str, float]:
    model.eval()
    prediction = rollout_hybrid_latents(
        model,
        latents[:history_length].to(device),
        frame_count=len(latents),
    ).cpu()
    per_frame = (prediction - latents).square().mean(dim=(1, 2, 3))
    post_cutoff = per_frame[history_length:]
    normalized_times = torch.linspace(0.0, 1.0, steps=len(latents), device=device)
    with torch.inference_mode():
        polarity_predictions = (
            torch.sigmoid(model.predict_polarity(normalized_times))[:, 0]
            >= 0.5
        ).cpu()
    polarity_accuracy = (
        polarity_predictions == (polarities >= 0.5)
    ).float().mean()
    return {
        "rollout_mse": post_cutoff.mean().item(),
        "final_frame_mse": per_frame[-1].item(),
        "peak_frame_mse": post_cutoff.max().item(),
        "polarity_accuracy": polarity_accuracy.item(),
    }


def _save_checkpoint(
    path: Path,
    model: HybridTemporalMemoryModel,
    config: HybridTrainingConfig,
    autoencoder_checkpoint: dict,
    latent_mean: torch.Tensor,
    latent_standard_deviation: torch.Tensor,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_type": "hybrid_temporal_memory",
            "model_kwargs": {
                "latent_channels": model.latent_channels,
                "latent_height": model.latent_height,
                "latent_width": model.latent_width,
                "base_channels": model.base_channels,
                "memory_token_count": model.memory_token_count,
                "fourier_frequencies": model.fourier_frequencies,
                "max_residual_step": model.max_residual_step,
                "memory_temperature": model.memory_temperature,
                "use_polarity_head": model.use_polarity_head,
            },
            "autoencoder_checkpoint": str(
                config.autoencoder_checkpoint.resolve()
            ),
            "image_size": autoencoder_checkpoint["image_size"],
            "input_threshold": autoencoder_checkpoint["input_threshold"],
            "activation_threshold": autoencoder_checkpoint[
                "activation_threshold"
            ],
            "rollout_warmup_frames": config.history_length,
            "canonicalize_polarity": config.canonicalize_polarity,
            "polarity_tracking_method": config.polarity_tracking_method,
            "polarity_switch_penalty": config.polarity_switch_penalty,
            "memory_initialization_indices": config.memory_initialization_indices,
            "latent_mean": latent_mean,
            "latent_standard_deviation": latent_standard_deviation,
            "epoch": epoch,
            "metrics": metrics,
            "state_dict": model.state_dict(),
        },
        path,
    )


def train_hybrid(config: HybridTrainingConfig) -> Path:
    if config.minimum_rollout_steps < 1:
        raise ValueError("minimum_rollout_steps must be positive")
    if config.rollout_steps < config.minimum_rollout_steps:
        raise ValueError(
            "rollout_steps must be at least minimum_rollout_steps"
        )
    if config.memory_temperature <= 0:
        raise ValueError("memory_temperature must be positive")
    if config.polarity_tracking_method not in {"border", "temporal"}:
        raise ValueError(
            "polarity_tracking_method must be 'border' or 'temporal'"
        )
    if config.polarity_switch_penalty < 0:
        raise ValueError("polarity_switch_penalty must be non-negative")
    _set_seed(config.seed)
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    autoencoder, autoencoder_checkpoint = load_autoencoder(
        config.autoencoder_checkpoint, device
    )
    print("Encoding the source sequence with the frozen autoencoder...")
    if config.canonicalize_polarity:
        raw_latents, polarities = encode_canonical_sequence(
            autoencoder,
            autoencoder_checkpoint,
            config.frame_dir,
            device,
            batch_size=config.batch_size,
            polarity_tracking_method=config.polarity_tracking_method,
            polarity_switch_penalty=config.polarity_switch_penalty,
        )
    else:
        raw_latents = encode_sequence(
            autoencoder,
            autoencoder_checkpoint,
            config.frame_dir,
            device,
            batch_size=config.batch_size,
        )
        polarities = torch.zeros(len(raw_latents), dtype=torch.float32)
    polarity_switches = int(
        (polarities[1:] != polarities[:-1]).sum().item()
    )
    print(
        f"Polarity path: {config.polarity_tracking_method}, "
        f"{polarity_switches} switches"
    )
    latents, latent_mean, latent_standard_deviation = normalize_latents(
        raw_latents
    )
    dataset = HybridWindowDataset(
        latents,
        history_length=config.history_length,
        rollout_steps=config.rollout_steps,
        polarities=polarities,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = HybridTemporalMemoryModel(
        latent_channels=latents.shape[1],
        latent_height=latents.shape[2],
        latent_width=latents.shape[3],
        base_channels=config.base_channels,
        memory_token_count=config.memory_token_count,
        fourier_frequencies=config.fourier_frequencies,
        max_residual_step=config.max_residual_step,
        memory_temperature=config.memory_temperature,
        use_polarity_head=config.canonicalize_polarity,
    ).to(device)
    memory_indices = select_scene_memory_indices(
        latents,
        config.memory_token_count,
        config.scene_cut_minimum_distance,
    )
    config.memory_initialization_indices = memory_indices.tolist()
    model.initialize_memory(latents.to(device), memory_indices)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    polarity_loss_function = nn.BCEWithLogitsLoss()

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
        f"Training hybrid temporal memory on {device}: {len(latents)} frames, "
        f"{len(dataset)} windows, {serializable_config['parameter_count']:,} "
        "parameters"
    )
    print(f"Scene-memory frames: {config.memory_initialization_indices}")

    history_rows: list[dict] = []
    best_rollout_mse = float("inf")
    best_checkpoint = config.run_dir / "model_best.pt"
    for epoch in range(1, config.epochs + 1):
        curriculum_progress = (
            (epoch - 1) / max(1, config.epochs - 1)
        )
        active_rollout_steps = round(
            config.minimum_rollout_steps
            + curriculum_progress
            * (config.rollout_steps - config.minimum_rollout_steps)
        )
        started = time.perf_counter()
        model.train()
        training_loss = 0.0
        polarity_loss_total = 0.0
        gate_total = 0.0
        entropy_total = 0.0
        batches = 0
        for sequences, times, target_polarities in loader:
            sequences = sequences.to(device)
            times = times.to(device)
            target_polarities = target_polarities.to(device)
            latent_history = sequences[:, : config.history_length]
            if config.latent_noise_standard_deviation:
                latent_history = latent_history + torch.randn_like(
                    latent_history
                ) * config.latent_noise_standard_deviation

            optimizer.zero_grad(set_to_none=True)
            latent_loss = torch.zeros((), device=device)
            polarity_loss = torch.zeros((), device=device)
            memory_entropy = torch.zeros((), device=device)
            gate_mean = torch.zeros((), device=device)
            for rollout_index in range(active_rollout_steps):
                predicted, extras = model(
                    latent_history, times[:, rollout_index]
                )
                target = sequences[
                    :, config.history_length + rollout_index
                ]
                latent_loss = latent_loss + loss_function(predicted, target)
                polarity_loss = polarity_loss + polarity_loss_function(
                    extras["polarity_logits"][:, 0],
                    target_polarities[:, rollout_index],
                )
                weights = extras["memory_weights"]
                memory_entropy = memory_entropy + (
                    -(
                        weights
                        * torch.log(weights.clamp_min(1e-8))
                    )
                    .sum(dim=1)
                    .mean()
                    / math.log(config.memory_token_count)
                )
                gate_mean = gate_mean + extras["memory_gate"].mean()
                latent_history = torch.cat(
                    (
                        latent_history[:, 1:],
                        predicted.detach().unsqueeze(1),
                    ),
                    dim=1,
                )
            latent_loss = latent_loss / active_rollout_steps
            polarity_loss = polarity_loss / active_rollout_steps
            memory_entropy = memory_entropy / active_rollout_steps
            gate_mean = gate_mean / active_rollout_steps
            loss = (
                latent_loss
                + config.polarity_loss_weight * polarity_loss
                + config.memory_entropy_weight * memory_entropy
            )
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            polarity_loss_total += polarity_loss.item()
            gate_total += gate_mean.item()
            entropy_total += memory_entropy.item()
            batches += 1

        metrics = _rollout_metrics(
            model,
            latents,
            polarities,
            device,
            config.history_length,
        )
        row = {
            "epoch": epoch,
            "active_rollout_steps": active_rollout_steps,
            "training_loss": training_loss / batches,
            "polarity_loss": polarity_loss_total / batches,
            "mean_memory_gate": gate_total / batches,
            "mean_memory_entropy": entropy_total / batches,
            **metrics,
            "seconds": time.perf_counter() - started,
        }
        history_rows.append(row)
        print(
            f"epoch {epoch:02d} | steps {active_rollout_steps:02d} | "
            f"train {row['training_loss']:.5f} | "
            f"gate {row['mean_memory_gate']:.3f} | "
            f"polarity {row['polarity_accuracy']:.3f} | "
            f"rollout {row['rollout_mse']:.5f} | "
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
            json.dumps(history_rows, indent=2), encoding="utf-8"
        )

    return best_checkpoint
