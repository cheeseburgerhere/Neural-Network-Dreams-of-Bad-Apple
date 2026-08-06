from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .autoregressive import load_autoencoder, normalize_latents
from .hybrid import (
    HybridWindowDataset,
    TemporalBlock,
    _rollout_metrics,
    _set_seed,
    encode_canonical_sequence,
)
from .reporting import write_training_report
from .training import resolve_device


class BleedingSceneMemoryModel(nn.Module):
    """Velocity predictor with softly bleeding, time-local scene anchors."""

    def __init__(
        self,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        base_channels: int = 8,
        anchor_count: int = 16,
        fourier_frequencies: int = 6,
        max_velocity_step: float = 0.5,
        use_dual_velocity: bool = False,
        max_fast_velocity_step: float = 2.0,
        anchor_temperature: float = 0.03,
        anchor_temperature_mode: str = "fixed",
        anchor_temperature_ratio: float = 0.45,
        maximum_anchor_gate: float = 0.35,
        maximum_transition_gate: float | None = None,
        use_cut_gate: bool = False,
        time_basis: str = "normalized",
        timeline_seconds: float = 1.0,
        time_fourier_base_frequency: float = 1.0,
        use_polarity_head: bool = True,
        polarity_knot_count: int = 0,
    ) -> None:
        super().__init__()
        if anchor_temperature_mode not in {"fixed", "spacing"}:
            raise ValueError(
                "anchor_temperature_mode must be 'fixed' or 'spacing'"
            )
        if time_basis not in {"normalized", "seconds"}:
            raise ValueError(
                "time_basis must be 'normalized' or 'seconds'"
            )
        if timeline_seconds <= 0:
            raise ValueError("timeline_seconds must be positive")
        if time_fourier_base_frequency <= 0:
            raise ValueError(
                "time_fourier_base_frequency must be positive"
            )
        if anchor_temperature_ratio <= 0:
            raise ValueError("anchor_temperature_ratio must be positive")
        if polarity_knot_count not in {0} and polarity_knot_count < 2:
            raise ValueError("polarity_knot_count must be zero or at least two")
        if maximum_transition_gate is None:
            maximum_transition_gate = maximum_anchor_gate
        if not maximum_anchor_gate <= maximum_transition_gate <= 1.0:
            raise ValueError(
                "maximum_transition_gate must be between "
                "maximum_anchor_gate and one"
            )
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.base_channels = base_channels
        self.anchor_count = anchor_count
        self.fourier_frequencies = fourier_frequencies
        self.max_velocity_step = max_velocity_step
        self.use_dual_velocity = use_dual_velocity
        self.max_fast_velocity_step = max_fast_velocity_step
        self.anchor_temperature = anchor_temperature
        self.anchor_temperature_mode = anchor_temperature_mode
        self.anchor_temperature_ratio = anchor_temperature_ratio
        self.maximum_anchor_gate = maximum_anchor_gate
        self.maximum_transition_gate = maximum_transition_gate
        self.use_cut_gate = use_cut_gate
        self.time_basis = time_basis
        self.timeline_seconds = timeline_seconds
        self.time_fourier_base_frequency = (
            time_fourier_base_frequency
        )
        self.use_polarity_head = use_polarity_head
        self.polarity_knot_count = polarity_knot_count

        self.encoder_high = TemporalBlock(
            latent_channels * 2, base_channels
        )
        self.encoder_middle = TemporalBlock(
            base_channels, base_channels * 2
        )
        self.bottleneck = TemporalBlock(
            base_channels * 2, base_channels * 4
        )
        self.decoder_middle = TemporalBlock(
            base_channels * 6, base_channels * 2
        )
        self.decoder_high = TemporalBlock(
            base_channels * 3, base_channels
        )
        self.velocity_head = nn.Conv2d(
            base_channels, latent_channels, kernel_size=3, padding=1
        )
        self.fast_velocity_head = (
            nn.Conv2d(
                base_channels,
                latent_channels,
                kernel_size=3,
                padding=1,
            )
            if use_dual_velocity
            else None
        )
        if self.fast_velocity_head is not None:
            nn.init.zeros_(self.fast_velocity_head.weight)
            nn.init.zeros_(self.fast_velocity_head.bias)
        self.motion_mask_head = nn.Conv2d(
            base_channels, 1, kernel_size=3, padding=1
        )
        self.spatial_anchor_gate_head = nn.Conv2d(
            base_channels, 1, kernel_size=3, padding=1
        )
        self.cut_gate_head = (
            nn.Conv2d(
                base_channels + 1, 1, kernel_size=3, padding=1
            )
            if use_cut_gate
            else None
        )
        nn.init.constant_(self.motion_mask_head.bias, -2.0)
        nn.init.constant_(self.spatial_anchor_gate_head.bias, -1.0)
        if self.cut_gate_head is not None:
            nn.init.zeros_(self.cut_gate_head.weight)
            nn.init.constant_(self.cut_gate_head.bias, -4.0)

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
        self.time_to_anchor_gate = nn.Linear(time_hidden, 1)
        nn.init.constant_(self.time_to_anchor_gate.bias, -1.0)
        self.time_to_polarity = (
            nn.Linear(time_hidden, 1) if use_polarity_head else None
        )
        self.polarity_spline_logits = (
            nn.Parameter(torch.zeros(polarity_knot_count))
            if use_polarity_head and polarity_knot_count
            else None
        )

        self.memory_tokens = nn.Parameter(
            torch.zeros(
                anchor_count,
                latent_channels,
                latent_height,
                latent_width,
            )
        )
        nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)
        self.register_buffer(
            "anchor_times", torch.linspace(0.0, 1.0, steps=anchor_count)
        )
        self.register_buffer(
            "anchor_reference",
            torch.zeros_like(self.memory_tokens),
        )

    def _time_features(self, normalized_time: torch.Tensor) -> torch.Tensor:
        normalized_time = normalized_time.reshape(-1, 1)
        features = [normalized_time]
        phase_time = (
            normalized_time * self.timeline_seconds
            if self.time_basis == "seconds"
            else normalized_time
        )
        for frequency_index in range(self.fourier_frequencies):
            frequency = (
                self.time_fourier_base_frequency
                * 2.0**frequency_index
            )
            angle = phase_time * (2.0 * math.pi * frequency)
            features.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(features, dim=1)

    def initialize_memory(
        self,
        normalized_latents: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        if len(indices) != self.anchor_count:
            raise ValueError("One initialization index is required per anchor")
        indices = indices.to(normalized_latents.device)
        values = normalized_latents[indices]
        with torch.no_grad():
            self.memory_tokens.copy_(values)
            self.anchor_reference.copy_(values)
            self.anchor_times.copy_(
                indices.float() / max(1, len(normalized_latents) - 1)
            )
        if self.anchor_temperature_mode == "spacing":
            spacing = torch.diff(self.anchor_times).median().item()
            self.anchor_temperature = max(
                1e-6, self.anchor_temperature_ratio * spacing
            )

    def address_memory(
        self, normalized_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_state = self.time_encoder(
            self._time_features(normalized_time)
        )
        distances = (
            normalized_time.reshape(-1, 1)
            - self.anchor_times.reshape(1, -1)
        ).abs()
        selected_count = min(2, self.anchor_count)
        selected_distances, selected_indices = torch.topk(
            distances, k=selected_count, dim=1, largest=False
        )
        selected_weights = torch.softmax(
            -selected_distances / self.anchor_temperature, dim=1
        )
        memory_weights = torch.zeros_like(distances)
        memory_weights.scatter_(1, selected_indices, selected_weights)
        maximum_gate = self.maximum_anchor_gate * torch.sigmoid(
            self.time_to_anchor_gate(time_state)
        )
        return time_state, memory_weights, maximum_gate

    def predict_polarity(self, normalized_time: torch.Tensor) -> torch.Tensor:
        if self.polarity_spline_logits is not None:
            times = normalized_time.reshape(-1).clamp(0.0, 1.0)
            positions = times * (self.polarity_knot_count - 1)
            left = positions.floor().long()
            right = (left + 1).clamp(max=self.polarity_knot_count - 1)
            fractions = positions - left
            logits = (
                self.polarity_spline_logits[left] * (1.0 - fractions)
                + self.polarity_spline_logits[right] * fractions
            )
            return logits[:, None]
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

        velocity_history = torch.cat(
            (
                torch.zeros_like(latent_history[:, :1]),
                latent_history[:, 1:] - latent_history[:, :-1],
            ),
            dim=1,
        )
        state_and_velocity = torch.cat(
            (latent_history, velocity_history), dim=2
        )
        features = state_and_velocity.permute(0, 2, 1, 3, 4)

        time_state, memory_weights, maximum_gate = self.address_memory(
            normalized_time
        )
        high = self.encoder_high(features)
        middle = self.encoder_middle(
            F.avg_pool3d(
                high, kernel_size=(2, 1, 1), stride=(2, 1, 1)
            )
        )
        bottleneck = self.bottleneck(
            F.avg_pool3d(
                middle, kernel_size=(2, 1, 1), stride=(2, 1, 1)
            )
        )
        bottleneck = bottleneck + self.time_to_bottleneck(time_state)[
            :, :, None, None, None
        ]

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
        up_high = self.decoder_high(
            torch.cat((up_high, high), dim=1)
        )
        final_features = up_high[:, :, -1]

        motion_mask = torch.sigmoid(
            self.motion_mask_head(final_features)
        )
        slow_velocity = self.max_velocity_step * torch.tanh(
            self.velocity_head(final_features)
        )
        if self.fast_velocity_head is None:
            fast_velocity = torch.zeros_like(slow_velocity)
        else:
            fast_velocity = (
                self.max_fast_velocity_step
                * motion_mask
                * torch.tanh(self.fast_velocity_head(final_features))
            )
        predicted_velocity = slow_velocity + fast_velocity
        motion_candidate = latent_history[:, -1] + predicted_velocity
        memory_candidate = torch.einsum(
            "bm,mchw->bchw", memory_weights, self.memory_tokens
        )
        spatial_gate = torch.sigmoid(
            self.spatial_anchor_gate_head(final_features)
            + self.time_to_anchor_gate(time_state)[:, :, None, None]
        )
        base_effective_gate = (
            self.maximum_anchor_gate
            * spatial_gate
            * (1.0 - motion_mask)
        )
        anchor_disagreement = (
            memory_candidate - motion_candidate
        ).abs().mean(dim=1, keepdim=True)
        if self.cut_gate_head is None:
            cut_gate = torch.zeros_like(base_effective_gate)
            effective_gate = base_effective_gate
        else:
            cut_gate = torch.sigmoid(
                self.cut_gate_head(
                    torch.cat((final_features, anchor_disagreement), dim=1)
                )
            )
            transition_strength = cut_gate * (
                1.0 - 0.5 * motion_mask
            )
            effective_gate = base_effective_gate + transition_strength * (
                self.maximum_transition_gate - base_effective_gate
            )
        next_latent = motion_candidate + effective_gate * (
            memory_candidate - motion_candidate
        )
        polarity_logits = self.predict_polarity(normalized_time)
        return next_latent, {
            "memory_weights": memory_weights,
            "memory_gate": effective_gate.mean(dim=(2, 3)),
            "maximum_memory_gate": maximum_gate,
            "spatial_memory_gate": effective_gate,
            "base_memory_gate": base_effective_gate,
            "cut_gate": cut_gate,
            "anchor_disagreement": anchor_disagreement,
            "motion_mask": motion_mask,
            "slow_velocity": slow_velocity,
            "fast_velocity": fast_velocity,
            "predicted_velocity": predicted_velocity,
            "motion_candidate": motion_candidate,
            "memory_candidate": memory_candidate,
            "polarity_logits": polarity_logits,
        }


def select_covered_anchor_indices(
    normalized_latents: torch.Tensor,
    anchor_count: int,
    minimum_distance: int,
) -> torch.Tensor:
    """Mix uniform coverage with high-change anchors."""
    if anchor_count < 2:
        raise ValueError("anchor_count must be at least two")
    if anchor_count > len(normalized_latents):
        raise ValueError("anchor_count cannot exceed the frame count")

    uniform_count = max(2, anchor_count // 2)
    selected = set(
        torch.linspace(
            0, len(normalized_latents) - 1, steps=uniform_count
        )
        .round()
        .long()
        .tolist()
    )
    changes = (
        normalized_latents[1:] - normalized_latents[:-1]
    ).square().mean(dim=(1, 2, 3))
    ranked = (torch.argsort(changes, descending=True) + 1).tolist()
    for candidate in ranked:
        if all(
            abs(candidate - existing) >= minimum_distance
            for existing in selected
        ):
            selected.add(candidate)
        if len(selected) == anchor_count:
            break

    if len(selected) < anchor_count:
        for candidate in ranked:
            selected.add(candidate)
            if len(selected) == anchor_count:
                break
    return torch.tensor(sorted(selected), dtype=torch.long)


@dataclass
class HybridV4TrainingConfig:
    autoencoder_checkpoint: Path
    frame_dir: Path
    run_dir: Path
    history_length: int = 16
    minimum_rollout_steps: int = 4
    rollout_steps: int = 32
    truncated_backprop_steps: int = 4
    base_channels: int = 8
    anchor_count: int = 16
    anchor_temperature: float = 0.03
    anchor_temperature_mode: str = "fixed"
    anchor_temperature_ratio: float = 0.45
    maximum_anchor_gate: float = 0.35
    maximum_transition_gate: float = 0.35
    anchor_minimum_distance: int = 8
    fourier_frequencies: int = 6
    time_basis: str = "normalized"
    timeline_seconds: float | None = None
    frames_per_second: float = 30.0
    time_fourier_base_frequency: float = 1.0
    max_velocity_step: float = 0.5
    use_dual_velocity: bool = False
    use_cut_gate: bool = False
    max_fast_velocity_step: float = 2.0
    velocity_loss_weight: float = 0.5
    slow_velocity_loss_weight: float = 0.5
    fast_velocity_loss_weight: float = 1.0
    fast_velocity_dynamic_weight: float = 4.0
    dynamic_loss_weight: float = 0.5
    motion_mask_loss_weight: float = 0.05
    cut_gate_loss_weight: float = 0.0
    anchor_loss_weight: float = 0.01
    polarity_loss_weight: float = 0.2
    polarity_calibration_steps: int = 500
    polarity_calibration_learning_rate: float = 0.03
    canonicalize_polarity: bool = True
    polarity_tracking_method: str = "temporal"
    polarity_switch_penalty: float = 0.05
    latent_noise_standard_deviation: float = 0.03
    epochs: int = 12
    batch_size: int = 2
    learning_rate: float = 5e-4
    warm_start_checkpoint: Path | None = None
    fast_head_only_epochs: int = 0
    motion_only_epochs: int = 0
    minimum_burn_in_steps: int = 0
    burn_in_steps: int = 0
    freeze_memory_epochs: int = 0
    architecture_version: str = "v4.1"
    reproduction_command: str | None = None
    seed: int = 7
    device: str = "auto"
    anchor_initialization_indices: list[int] = field(default_factory=list)


def _save_checkpoint(
    path: Path,
    model: BleedingSceneMemoryModel,
    config: HybridV4TrainingConfig,
    autoencoder_checkpoint: dict,
    latent_mean: torch.Tensor,
    latent_standard_deviation: torch.Tensor,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_type": "hybrid_v4_bleeding_memory",
            "model_kwargs": {
                "latent_channels": model.latent_channels,
                "latent_height": model.latent_height,
                "latent_width": model.latent_width,
                "base_channels": model.base_channels,
                "anchor_count": model.anchor_count,
                "fourier_frequencies": model.fourier_frequencies,
                "max_velocity_step": model.max_velocity_step,
                "use_dual_velocity": model.use_dual_velocity,
                "max_fast_velocity_step": model.max_fast_velocity_step,
                "anchor_temperature": model.anchor_temperature,
                "anchor_temperature_mode": (
                    model.anchor_temperature_mode
                ),
                "anchor_temperature_ratio": (
                    model.anchor_temperature_ratio
                ),
                "maximum_anchor_gate": model.maximum_anchor_gate,
                "maximum_transition_gate": (
                    model.maximum_transition_gate
                ),
                "use_cut_gate": model.use_cut_gate,
                "time_basis": model.time_basis,
                "timeline_seconds": model.timeline_seconds,
                "time_fourier_base_frequency": (
                    model.time_fourier_base_frequency
                ),
                "use_polarity_head": model.use_polarity_head,
                "polarity_knot_count": model.polarity_knot_count,
            },
            "architecture_version": config.architecture_version,
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
            "anchor_initialization_indices": (
                config.anchor_initialization_indices
            ),
            "latent_mean": latent_mean,
            "latent_standard_deviation": latent_standard_deviation,
            "epoch": epoch,
            "metrics": metrics,
            "state_dict": model.state_dict(),
        },
        path,
    )


def _calibrate_polarity_head(
    model: BleedingSceneMemoryModel,
    polarities: torch.Tensor,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    if model.time_to_polarity is None or steps == 0:
        return {"steps": 0, "accuracy": 1.0, "loss": 0.0}

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.time_to_polarity.parameters():
        parameter.requires_grad_(True)
    normalized_times = torch.linspace(
        0.0, 1.0, steps=len(polarities), device=device
    )
    with torch.no_grad():
        time_states = model.time_encoder(
            model._time_features(normalized_times)
        )
    targets = polarities.to(device)
    optimizer = torch.optim.Adam(
        model.time_to_polarity.parameters(), lr=learning_rate
    )
    loss = torch.zeros((), device=device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model.time_to_polarity(time_states)[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = model.time_to_polarity(time_states)[:, 0] >= 0.0
        accuracy = (predictions == (targets >= 0.5)).float().mean()
    return {
        "steps": steps,
        "accuracy": accuracy.item(),
        "loss": loss.item(),
    }


def _configure_motion_finetuning(
    model: BleedingSceneMemoryModel,
    fast_head_only: bool,
    train_spatial_gate: bool,
) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if fast_head_only:
        if model.fast_velocity_head is None:
            raise ValueError("fast-head stage requires dual velocity")
        motion_modules = [
            model.fast_velocity_head,
            model.motion_mask_head,
        ]
    else:
        motion_modules = [
            model.encoder_high,
            model.encoder_middle,
            model.bottleneck,
            model.decoder_middle,
            model.decoder_high,
            model.velocity_head,
            model.motion_mask_head,
        ]
        if model.fast_velocity_head is not None:
            motion_modules.append(model.fast_velocity_head)
    if train_spatial_gate:
        motion_modules.append(model.spatial_anchor_gate_head)
        if model.cut_gate_head is not None:
            motion_modules.append(model.cut_gate_head)
    for module in motion_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _architecture_notes(
    config: HybridV4TrainingConfig,
    model: BleedingSceneMemoryModel,
) -> list[str]:
    notes = [
        "A causal temporal U-Net predicts slow and fast latent velocity "
        "from the previous latent window.",
        f"{model.anchor_count} time-local scene memories bleed into the "
        f"motion candidate with a normal cap of "
        f"{model.maximum_anchor_gate:.2f}.",
        f"Time features use the `{model.time_basis}` basis across "
        f"{model.timeline_seconds:.2f} seconds.",
        f"Anchor temperature is `{model.anchor_temperature_mode}` and "
        f"resolved to {model.anchor_temperature:.6f}.",
    ]
    if model.use_cut_gate:
        notes.append(
            "A learned cut gate can temporarily raise scene correction "
            f"to {model.maximum_transition_gate:.2f}; its disagreement "
            "signal naturally decays as the rollout approaches memory."
        )
    if config.burn_in_steps:
        notes.append(
            "Each supervised window starts after a randomly sampled "
            f"{config.minimum_burn_in_steps}-{config.burn_in_steps} frame "
            "free-running burn-in with no gradient retained through it."
        )
    if config.freeze_memory_epochs:
        notes.append(
            f"Scene-memory tensors remain frozen for the first "
            f"{config.freeze_memory_epochs} epochs."
        )
    return notes


def train_hybrid_v4(config: HybridV4TrainingConfig) -> Path:
    if config.rollout_steps < config.minimum_rollout_steps:
        raise ValueError(
            "rollout_steps must be at least minimum_rollout_steps"
        )
    if config.truncated_backprop_steps < 1:
        raise ValueError("truncated_backprop_steps must be positive")
    if config.anchor_temperature <= 0:
        raise ValueError("anchor_temperature must be positive")
    if config.anchor_temperature_mode not in {"fixed", "spacing"}:
        raise ValueError(
            "anchor_temperature_mode must be 'fixed' or 'spacing'"
        )
    if config.anchor_temperature_ratio <= 0:
        raise ValueError("anchor_temperature_ratio must be positive")
    if not 0.0 <= config.maximum_anchor_gate <= 1.0:
        raise ValueError("maximum_anchor_gate must be between zero and one")
    if not (
        config.maximum_anchor_gate
        <= config.maximum_transition_gate
        <= 1.0
    ):
        raise ValueError(
            "maximum_transition_gate must be between "
            "maximum_anchor_gate and one"
        )
    if config.time_basis not in {"normalized", "seconds"}:
        raise ValueError("time_basis must be 'normalized' or 'seconds'")
    if config.frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    if config.time_fourier_base_frequency <= 0:
        raise ValueError(
            "time_fourier_base_frequency must be positive"
        )
    if config.polarity_calibration_steps < 0:
        raise ValueError("polarity_calibration_steps must be non-negative")
    if config.polarity_calibration_learning_rate <= 0:
        raise ValueError(
            "polarity_calibration_learning_rate must be positive"
        )
    if config.max_fast_velocity_step <= 0:
        raise ValueError("max_fast_velocity_step must be positive")
    if config.motion_only_epochs < 0:
        raise ValueError("motion_only_epochs must be non-negative")
    if not 0 <= config.minimum_burn_in_steps <= config.burn_in_steps:
        raise ValueError(
            "minimum_burn_in_steps must be between zero and burn_in_steps"
        )
    if config.freeze_memory_epochs < 0:
        raise ValueError("freeze_memory_epochs must be non-negative")
    if not 0 <= config.fast_head_only_epochs <= config.motion_only_epochs:
        raise ValueError(
            "fast_head_only_epochs must be between zero and "
            "motion_only_epochs"
        )
    if config.warm_start_checkpoint is not None:
        if not config.use_dual_velocity:
            raise ValueError(
                "warm-start motion fine-tuning requires dual velocity"
            )
        if not config.warm_start_checkpoint.exists():
            raise FileNotFoundError(config.warm_start_checkpoint)

    _set_seed(config.seed)
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    autoencoder, autoencoder_checkpoint = load_autoencoder(
        config.autoencoder_checkpoint, device
    )
    print("Encoding the temporally canonical source sequence...")
    raw_latents, polarities = encode_canonical_sequence(
        autoencoder,
        autoencoder_checkpoint,
        config.frame_dir,
        device,
        batch_size=config.batch_size,
        polarity_tracking_method=config.polarity_tracking_method,
        polarity_switch_penalty=config.polarity_switch_penalty,
    )
    latents, latent_mean, latent_standard_deviation = normalize_latents(
        raw_latents
    )
    if config.timeline_seconds is None:
        config.timeline_seconds = (
            (len(latents) - 1) / config.frames_per_second
        )
    global_changes = torch.zeros(len(latents))
    global_changes[1:] = (
        latents[1:] - latents[:-1]
    ).square().mean(dim=(1, 2, 3))
    cut_floor = torch.quantile(global_changes[1:], 0.50).item()
    cut_ceiling = torch.quantile(global_changes[1:], 0.95).item()
    cut_scale = max(1e-6, cut_ceiling - cut_floor)
    dataset = HybridWindowDataset(
        latents,
        history_length=config.history_length,
        rollout_steps=config.burn_in_steps + config.rollout_steps,
        polarities=polarities,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = BleedingSceneMemoryModel(
        latent_channels=latents.shape[1],
        latent_height=latents.shape[2],
        latent_width=latents.shape[3],
        base_channels=config.base_channels,
        anchor_count=config.anchor_count,
        fourier_frequencies=config.fourier_frequencies,
        max_velocity_step=config.max_velocity_step,
        use_dual_velocity=config.use_dual_velocity,
        max_fast_velocity_step=config.max_fast_velocity_step,
        anchor_temperature=config.anchor_temperature,
        anchor_temperature_mode=config.anchor_temperature_mode,
        anchor_temperature_ratio=config.anchor_temperature_ratio,
        maximum_anchor_gate=config.maximum_anchor_gate,
        maximum_transition_gate=config.maximum_transition_gate,
        use_cut_gate=config.use_cut_gate,
        time_basis=config.time_basis,
        timeline_seconds=config.timeline_seconds,
        time_fourier_base_frequency=(
            config.time_fourier_base_frequency
        ),
        use_polarity_head=config.canonicalize_polarity,
    ).to(device)
    anchor_indices = select_covered_anchor_indices(
        latents,
        anchor_count=config.anchor_count,
        minimum_distance=config.anchor_minimum_distance,
    )
    config.anchor_initialization_indices = anchor_indices.tolist()
    model.initialize_memory(latents.to(device), anchor_indices)
    config.anchor_temperature = model.anchor_temperature
    if config.warm_start_checkpoint is not None:
        warm_state = torch.load(
            config.warm_start_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if warm_state.get("model_type") != "hybrid_v4_bleeding_memory":
            raise ValueError("warm-start checkpoint must be a v4 model")
        missing, unexpected = model.load_state_dict(
            warm_state["state_dict"], strict=False
        )
        allowed_missing = {
            "fast_velocity_head.weight",
            "fast_velocity_head.bias",
        }
        if model.cut_gate_head is not None:
            allowed_missing.update(
                {
                    "cut_gate_head.weight",
                    "cut_gate_head.bias",
                }
            )
        allowed_missing.intersection_update(set(missing))
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                "incompatible warm-start state: "
                f"missing={missing}, unexpected={unexpected}"
            )
        warm_anchor_indices = warm_state.get(
            "anchor_initialization_indices",
            config.anchor_initialization_indices,
        )
        config.anchor_initialization_indices = list(warm_anchor_indices)
        print(
            f"Warm-started motion path from "
            f"{config.warm_start_checkpoint.resolve()}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    polarity_loss_function = nn.BCEWithLogitsLoss()
    mask_loss_function = nn.BCELoss()

    serializable_config = asdict(config)
    for key in ("autoencoder_checkpoint", "frame_dir", "run_dir"):
        serializable_config[key] = str(Path(serializable_config[key]).resolve())
    if config.warm_start_checkpoint is not None:
        serializable_config["warm_start_checkpoint"] = str(
            config.warm_start_checkpoint.resolve()
        )
    serializable_config["resolved_device"] = str(device)
    serializable_config["latent_shape"] = list(latents.shape[1:])
    serializable_config["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    (config.run_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )
    report_title = (
        "Neural Network Dreams Bad Apple — "
        f"Hybrid {config.architecture_version}"
    )
    architecture_notes = _architecture_notes(config, model)
    write_training_report(
        config.run_dir,
        title=report_title,
        status="Training initialized",
        architecture=architecture_notes,
        config=serializable_config,
        command=config.reproduction_command,
        notes=(
            "This report is updated after every epoch so an interrupted "
            "experiment still has a readable history.",
        ),
    )
    print(
        f"Training bleeding-memory {config.architecture_version} on "
        f"{device}: {len(latents)} frames, "
        f"{len(dataset)} windows, {serializable_config['parameter_count']:,} "
        "parameters"
    )
    print(f"Scene anchors: {config.anchor_initialization_indices}")

    history_rows: list[dict] = []
    best_rollout_mse = float("inf")
    best_checkpoint = config.run_dir / "model_best.pt"
    if config.warm_start_checkpoint is not None:
        baseline_metrics = _rollout_metrics(
            model,
            latents,
            polarities,
            device,
            config.history_length,
        )
        best_rollout_mse = baseline_metrics["rollout_mse"]
        _save_checkpoint(
            best_checkpoint,
            model,
            config,
            autoencoder_checkpoint,
            latent_mean,
            latent_standard_deviation,
            epoch=0,
            metrics=baseline_metrics,
        )
        print(
            f"Warm-start baseline rollout: {best_rollout_mse:.5f}"
        )
    for epoch in range(1, config.epochs + 1):
        curriculum_progress = (epoch - 1) / max(1, config.epochs - 1)
        active_rollout_steps = round(
            config.minimum_rollout_steps
            + curriculum_progress
            * (config.rollout_steps - config.minimum_rollout_steps)
        )
        started = time.perf_counter()
        training_stage = "full"
        trainable_parameter_count = serializable_config["parameter_count"]
        if config.warm_start_checkpoint is not None:
            fast_head_only = epoch <= config.fast_head_only_epochs
            train_spatial_gate = epoch > config.motion_only_epochs
            if fast_head_only:
                training_stage = "fast-head-only"
            elif train_spatial_gate:
                training_stage = "motion+gate"
            else:
                training_stage = "motion-only"
            trainable_parameter_count = _configure_motion_finetuning(
                model,
                fast_head_only=fast_head_only,
                train_spatial_gate=train_spatial_gate,
            )
        else:
            freeze_memory = epoch <= config.freeze_memory_epochs
            model.memory_tokens.requires_grad_(not freeze_memory)
            if freeze_memory:
                training_stage = "memory-frozen"
            trainable_parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        model.train()
        component_totals = {
            "loss": 0.0,
            "latent": 0.0,
            "velocity": 0.0,
            "slow_velocity": 0.0,
            "fast_velocity": 0.0,
            "dynamic": 0.0,
            "mask": 0.0,
            "cut_gate": 0.0,
            "polarity": 0.0,
            "anchor": 0.0,
            "gate": 0.0,
        }
        batches = 0
        burn_in_total = 0
        for sequences, times, target_polarities in loader:
            sequences = sequences.to(device)
            times = times.to(device)
            target_polarities = target_polarities.to(device)
            active_burn_in_steps = random.randint(
                config.minimum_burn_in_steps,
                config.burn_in_steps,
            )
            latent_history = sequences[:, : config.history_length]
            if config.latent_noise_standard_deviation:
                latent_history = latent_history + torch.randn_like(
                    latent_history
                ) * config.latent_noise_standard_deviation
            with torch.no_grad():
                for burn_in_index in range(active_burn_in_steps):
                    predicted, _ = model(
                        latent_history,
                        times[:, burn_in_index],
                    )
                    latent_history = torch.cat(
                        (
                            latent_history[:, 1:],
                            predicted.unsqueeze(1),
                        ),
                        dim=1,
                    )
            latent_history = latent_history.detach()
            burn_in_total += active_burn_in_steps

            optimizer.zero_grad(set_to_none=True)
            block_loss: torch.Tensor | None = None
            batch_components = {
                key: 0.0 for key in component_totals if key != "anchor"
            }
            for rollout_index in range(active_rollout_steps):
                predicted, extras = model(
                    latent_history,
                    times[:, active_burn_in_steps + rollout_index],
                )
                target_index = (
                    config.history_length
                    + active_burn_in_steps
                    + rollout_index
                )
                target = sequences[:, target_index]
                target_velocity = (
                    target - sequences[:, target_index - 1]
                )

                latent_loss = F.mse_loss(predicted, target)
                motion_strength = target_velocity.abs().mean(
                    dim=1, keepdim=True
                )
                relative_motion = motion_strength / (
                    motion_strength.mean(dim=(2, 3), keepdim=True) + 1e-6
                )
                dynamic_weights = 1.0 + relative_motion.clamp(max=4.0)
                if model.use_dual_velocity:
                    slow_target = target_velocity.clamp(
                        min=-model.max_velocity_step,
                        max=model.max_velocity_step,
                    )
                    fast_target = target_velocity - slow_target
                    slow_velocity_loss = F.smooth_l1_loss(
                        extras["slow_velocity"],
                        slow_target,
                        beta=0.25,
                    )
                    fast_error = F.smooth_l1_loss(
                        extras["fast_velocity"],
                        fast_target,
                        beta=0.25,
                        reduction="none",
                    )
                    fast_strength = fast_target.abs().mean(
                        dim=1, keepdim=True
                    )
                    relative_fast_motion = fast_strength / (
                        fast_strength.mean(
                            dim=(2, 3), keepdim=True
                        )
                        + 1e-6
                    )
                    fast_weights = 1.0 + (
                        config.fast_velocity_dynamic_weight
                        * relative_fast_motion.clamp(max=4.0)
                    )
                    fast_velocity_loss = (
                        fast_weights * fast_error
                    ).mean()
                    velocity_loss = (
                        slow_velocity_loss + fast_velocity_loss
                    )
                else:
                    slow_velocity_loss = F.mse_loss(
                        extras["predicted_velocity"], target_velocity
                    )
                    fast_velocity_loss = torch.zeros(
                        (), device=device
                    )
                    velocity_loss = slow_velocity_loss
                dynamic_loss = (
                    dynamic_weights * (predicted - target).square()
                ).mean()
                target_motion_mask = (relative_motion / 2.0).clamp(
                    min=0.0, max=1.0
                )
                mask_loss = mask_loss_function(
                    extras["motion_mask"], target_motion_mask
                )
                if model.cut_gate_head is None:
                    cut_gate_loss = torch.zeros((), device=device)
                else:
                    target_cut_strength = (
                        target_velocity.square().mean(dim=(1, 2, 3))
                        - cut_floor
                    ) / cut_scale
                    target_cut_strength = target_cut_strength.clamp(
                        min=0.0, max=1.0
                    )
                    predicted_cut_strength = extras["cut_gate"].mean(
                        dim=(1, 2, 3)
                    )
                    cut_gate_loss = F.smooth_l1_loss(
                        predicted_cut_strength,
                        target_cut_strength,
                        beta=0.2,
                    )
                polarity_loss = polarity_loss_function(
                    extras["polarity_logits"][:, 0],
                    target_polarities[
                        :, active_burn_in_steps + rollout_index
                    ],
                )
                if model.use_dual_velocity:
                    velocity_objective = (
                        config.slow_velocity_loss_weight
                        * slow_velocity_loss
                        + config.fast_velocity_loss_weight
                        * fast_velocity_loss
                    )
                else:
                    velocity_objective = (
                        config.velocity_loss_weight * velocity_loss
                    )
                step_loss = (
                    latent_loss
                    + velocity_objective
                    + config.dynamic_loss_weight * dynamic_loss
                    + config.motion_mask_loss_weight * mask_loss
                    + config.cut_gate_loss_weight * cut_gate_loss
                    + config.polarity_loss_weight * polarity_loss
                )
                scaled_step_loss = step_loss / active_rollout_steps
                block_loss = (
                    scaled_step_loss
                    if block_loss is None
                    else block_loss + scaled_step_loss
                )

                batch_components["loss"] += step_loss.item()
                batch_components["latent"] += latent_loss.item()
                batch_components["velocity"] += velocity_loss.item()
                batch_components["slow_velocity"] += (
                    slow_velocity_loss.item()
                )
                batch_components["fast_velocity"] += (
                    fast_velocity_loss.item()
                )
                batch_components["dynamic"] += dynamic_loss.item()
                batch_components["mask"] += mask_loss.item()
                batch_components["cut_gate"] += cut_gate_loss.item()
                batch_components["polarity"] += polarity_loss.item()
                batch_components["gate"] += extras[
                    "memory_gate"
                ].mean().item()

                latent_history = torch.cat(
                    (
                        latent_history[:, 1:],
                        predicted.unsqueeze(1),
                    ),
                    dim=1,
                )
                block_finished = (
                    (rollout_index + 1) % config.truncated_backprop_steps == 0
                    or rollout_index + 1 == active_rollout_steps
                )
                if block_finished:
                    if block_loss is None:
                        raise RuntimeError("empty truncated backprop block")
                    block_loss.backward()
                    block_loss = None
                    latent_history = latent_history.detach()

            anchor_loss = F.mse_loss(
                model.memory_tokens, model.anchor_reference
            )
            if (
                config.anchor_loss_weight
                and model.memory_tokens.requires_grad
            ):
                (
                    config.anchor_loss_weight * anchor_loss
                ).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for key, value in batch_components.items():
                component_totals[key] += value / active_rollout_steps
            component_totals["anchor"] += anchor_loss.item()
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
            "mean_burn_in_steps": burn_in_total / batches,
            "training_stage": training_stage,
            "trainable_parameter_count": trainable_parameter_count,
            **{
                f"training_{key}": value / batches
                for key, value in component_totals.items()
            },
            **metrics,
            "seconds": time.perf_counter() - started,
        }
        history_rows.append(row)
        print(
            f"epoch {epoch:02d} | burn {row['mean_burn_in_steps']:.0f} | "
            f"steps {active_rollout_steps:02d} | "
            f"train {row['training_loss']:.5f} | "
            f"slow {row['training_slow_velocity']:.5f} | "
            f"fast {row['training_fast_velocity']:.5f} | "
            f"cut {row['training_cut_gate']:.5f} | "
            f"gate {row['training_gate']:.3f} | "
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
        write_training_report(
            config.run_dir,
            title=report_title,
            status=f"Training epoch {epoch}/{config.epochs} complete",
            architecture=architecture_notes,
            config=serializable_config,
            command=config.reproduction_command,
            history=history_rows,
            checkpoint=(
                best_checkpoint if best_checkpoint.exists() else None
            ),
        )

    best_state = torch.load(
        best_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(best_state["state_dict"])
    calibration = _calibrate_polarity_head(
        model,
        polarities,
        device,
        steps=config.polarity_calibration_steps,
        learning_rate=config.polarity_calibration_learning_rate,
    )
    best_state["state_dict"] = model.state_dict()
    best_state["polarity_calibration"] = calibration
    torch.save(best_state, best_checkpoint)
    print(
        "Polarity calibration: "
        f"{calibration['accuracy']:.3f} accuracy, "
        f"loss {calibration['loss']:.5f}"
    )
    write_training_report(
        config.run_dir,
        title=report_title,
        status="Training complete",
        architecture=architecture_notes,
        config=serializable_config,
        command=config.reproduction_command,
        history=history_rows,
        checkpoint=best_checkpoint,
        notes=(
            f"Polarity calibration accuracy: "
            f"{calibration['accuracy']:.4f}.",
            "Use the matching output-folder report for binary-error and "
            "visual rollout metrics.",
        ),
    )
    return best_checkpoint
