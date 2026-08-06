from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .hybrid import HybridWindowDataset, _rollout_metrics, _set_seed
from .hybrid_v4 import BleedingSceneMemoryModel
from .training import resolve_device


@dataclass
class RecoveryFineTuneConfig:
    checkpoint: Path
    latent_cache: Path
    run_dir: Path
    epochs: int = 1
    history_length: int = 16
    rollout_steps: int = 16
    truncated_backprop_steps: int = 4
    minimum_burn_in_steps: int = 32
    burn_in_steps: int = 128
    clean_batch_interval: int = 5
    batch_size: int = 2
    learning_rate: float = 5e-5
    recovery_velocity_weight: float = 0.25
    scene_velocity_weight: float = 0.05
    dynamic_loss_weight: float = 0.5
    motion_mask_loss_weight: float = 0.05
    checkpoint_every_minutes: float = 5.0
    max_runtime_minutes: float = 100.0
    max_batches_per_epoch: int = 0
    seed: int = 7
    device: str = "auto"


def motion_velocity_target(
    current_state: torch.Tensor,
    target: torch.Tensor,
    memory_candidate: torch.Tensor,
    memory_gate: torch.Tensor,
) -> torch.Tensor:
    """Velocity needed before frozen memory fusion to reach the target."""
    retained_motion = (1.0 - memory_gate).clamp_min(0.1)
    motion_target = (
        target - memory_gate * memory_candidate
    ) / retained_motion
    return motion_target - current_state


def configure_recovery_modules(
    model: BleedingSceneMemoryModel,
) -> int:
    """Train late motion interpretation; freeze timeline and scene memory."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [
        model.decoder_middle,
        model.decoder_high,
        model.velocity_head,
        model.motion_mask_head,
    ]
    if model.fast_velocity_head is not None:
        modules.append(model.fast_velocity_head)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return (
        cache["normalized_latents"].float(),
        cache["polarities"].float(),
    )


def _renderable_checkpoint(
    source: dict,
    model: BleedingSceneMemoryModel,
    *,
    epoch: int,
    metrics: dict[str, float],
    recovery: dict,
) -> dict:
    checkpoint = dict(source)
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["metrics"] = metrics
    checkpoint["architecture_version"] = "v4.3-state-recovery"
    checkpoint["recovery_fine_tune"] = recovery
    return checkpoint


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_status(
    config: RecoveryFineTuneConfig,
    *,
    status: str,
    epoch: int,
    batch: int,
    batches: int,
    elapsed_seconds: float,
    history: list[dict],
    checkpoint: Path,
) -> None:
    payload = {
        "status": status,
        "epoch": epoch,
        "configured_epochs": config.epochs,
        "batch": batch,
        "batches_in_epoch": batches,
        "elapsed_seconds": elapsed_seconds,
        "latest_resumable_checkpoint": str(
            (config.run_dir / "resume.pt").resolve()
        ),
        "best_checkpoint": str(checkpoint.resolve()),
        "stop_file": str((config.run_dir / "STOP").resolve()),
        "history": history,
    }
    (config.run_dir / "status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _write_report(
    config: RecoveryFineTuneConfig,
    *,
    status: str,
    trainable_parameters: int,
    baseline_metrics: dict[str, float],
    history: list[dict],
) -> None:
    lines = [
        "# Hybrid V4.3 state-recovery fine-tune",
        "",
        f"**Status:** {status}",
        "",
        "## Purpose",
        "",
        "Correct the teacher-good/rollout-bad objective mismatch without "
        "retraining scene memory, polarity, or the full temporal backbone.",
        "",
        "- Primary velocity target is relative to the model's current "
        "predicted state.",
        "- Frozen memory fusion is algebraically included in that target.",
        "- Clean true-scene velocity remains a smaller auxiliary objective.",
        "- Decoder-middle, decoder-high, velocity heads, and motion-mask head "
        "are trainable.",
        f"- Trainable parameters: {trainable_parameters:,}.",
        "",
        "## Safe stopping",
        "",
        "- Atomic `resume.pt` saved every configured interval.",
        "- `model_last.pt` saved after every completed epoch or graceful stop.",
        "- Create `STOP` in this folder for graceful stop after current batch.",
        f"- Automatic runtime cutoff: {config.max_runtime_minutes:.1f} minutes.",
        "- Hard shutdown loses at most one checkpoint interval.",
        "",
        "```powershell",
        (
            f"New-Item \"{(config.run_dir / 'STOP').resolve()}\" "
            "-ItemType File"
        ),
        "```",
        "",
        "Remove `STOP`, then rerun the same command to resume.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(
            {
                **asdict(config),
                "checkpoint": str(config.checkpoint.resolve()),
                "latent_cache": str(config.latent_cache.resolve()),
                "run_dir": str(config.run_dir.resolve()),
            },
            indent=2,
        ),
        "```",
        "",
        "## Baseline",
        "",
        f"- Rollout latent MSE: {baseline_metrics['rollout_mse']:.6f}.",
        f"- Peak latent MSE: {baseline_metrics['peak_frame_mse']:.6f}.",
        "",
        "## Completed epochs",
        "",
        "| Epoch | Loss | Latent | Recovery velocity | Scene velocity | "
        "Rollout MSE | Seconds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(
            "| {epoch} | {training_loss:.6f} | {latent_loss:.6f} | "
            "{recovery_velocity_loss:.6f} | {scene_velocity_loss:.6f} | "
            "{rollout_mse:.6f} | {seconds:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```powershell",
            (
                "python prototype.py fine-tune-recovery "
                f"--run-dir \"{config.run_dir}\" "
                f"--epochs {config.epochs} "
                f"--max-runtime-minutes {config.max_runtime_minutes}"
            ),
            "```",
            "",
        ]
    )
    (config.run_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _save_resume(
    config: RecoveryFineTuneConfig,
    source_checkpoint: dict,
    model: BleedingSceneMemoryModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    next_batch: int,
    metrics: dict[str, float],
    history: list[dict],
    elapsed_seconds: float,
) -> None:
    payload = _renderable_checkpoint(
        source_checkpoint,
        model,
        epoch=epoch,
        metrics=metrics,
        recovery={
            "config": asdict(config),
            "epoch": epoch,
            "next_batch": next_batch,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["training_state"] = {
        "epoch": epoch,
        "next_batch": next_batch,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
    }
    _atomic_torch_save(payload, config.run_dir / "resume.pt")


def fine_tune_recovery(config: RecoveryFineTuneConfig) -> Path:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if config.truncated_backprop_steps < 1:
        raise ValueError("truncated_backprop_steps must be positive")
    if config.minimum_burn_in_steps < 0:
        raise ValueError("minimum burn-in must not be negative")
    if config.burn_in_steps < config.minimum_burn_in_steps:
        raise ValueError("maximum burn-in must cover minimum burn-in")
    if config.clean_batch_interval < 1:
        raise ValueError("clean batch interval must be positive")
    if config.checkpoint_every_minutes <= 0:
        raise ValueError("checkpoint interval must be positive")
    if config.max_runtime_minutes <= 0:
        raise ValueError("maximum runtime must be positive")

    _set_seed(config.seed)
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    stop_path = config.run_dir / "STOP"
    if stop_path.exists():
        raise RuntimeError(
            f"{stop_path} exists; remove it before starting or resuming"
        )

    source_checkpoint = torch.load(
        config.checkpoint, map_location="cpu", weights_only=False
    )
    model = BleedingSceneMemoryModel(
        **source_checkpoint["model_kwargs"]
    )
    model.load_state_dict(source_checkpoint["state_dict"])
    model.to(device)
    trainable_parameters = configure_recovery_modules(model)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )

    latents, polarities = _load_cache(config.latent_cache)
    dataset = HybridWindowDataset(
        latents,
        history_length=config.history_length,
        rollout_steps=config.burn_in_steps + config.rollout_steps,
        polarities=polarities,
    )
    baseline_metrics = dict(source_checkpoint.get("metrics", {}))
    if "rollout_mse" not in baseline_metrics:
        baseline_metrics = _rollout_metrics(
            model,
            latents,
            polarities,
            device,
            config.history_length,
        )
    best_rollout_mse = float(baseline_metrics["rollout_mse"])
    best_checkpoint = config.run_dir / "model_best.pt"
    if not best_checkpoint.exists():
        _atomic_torch_save(source_checkpoint, best_checkpoint)

    history: list[dict] = []
    start_epoch = 1
    resume_batch = 0
    carried_elapsed = 0.0
    resume_path = config.run_dir / "resume.pt"
    if resume_path.exists():
        resume = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        state = resume["training_state"]
        start_epoch = int(state["epoch"])
        resume_batch = int(state["next_batch"])
        history = list(state["history"])
        carried_elapsed = float(state.get("elapsed_seconds", 0.0))
        if history:
            best_rollout_mse = min(
                best_rollout_mse,
                min(float(row["rollout_mse"]) for row in history),
            )
        print(
            f"Resuming epoch {start_epoch}, batch {resume_batch}, "
            f"prior elapsed {carried_elapsed / 60.0:.1f}m"
        )

    started = time.perf_counter()
    last_checkpoint_time = started
    total_runtime_limit = config.max_runtime_minutes * 60.0
    current_metrics = baseline_metrics
    report_status = "Initialized"
    _write_report(
        config,
        status=report_status,
        trainable_parameters=trainable_parameters,
        baseline_metrics=baseline_metrics,
        history=history,
    )
    print(
        f"Recovery fine-tune on {device}: {len(dataset)} windows, "
        f"{trainable_parameters:,} trainable parameters"
    )

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        epoch_dataset = Subset(dataset, order)
        loader = DataLoader(
            epoch_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        batch_count = len(loader)
        totals = {
            "loss": 0.0,
            "latent": 0.0,
            "recovery_velocity": 0.0,
            "scene_velocity": 0.0,
            "dynamic": 0.0,
            "motion_mask": 0.0,
        }
        completed_batches = 0
        model.train()
        for batch_index, (sequences, times, _) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            if (
                config.max_batches_per_epoch
                and completed_batches >= config.max_batches_per_epoch
            ):
                break

            sequences = sequences.to(device, non_blocking=True)
            times = times.to(device, non_blocking=True)
            if batch_index % config.clean_batch_interval == 0:
                active_burn_in = 0
            else:
                batch_random = random.Random(
                    config.seed + epoch * 1_000_000 + batch_index
                )
                active_burn_in = batch_random.randint(
                    config.minimum_burn_in_steps,
                    config.burn_in_steps,
                )
            latent_history = sequences[:, : config.history_length]
            with torch.no_grad():
                for burn_index in range(active_burn_in):
                    predicted, _ = model(
                        latent_history, times[:, burn_index]
                    )
                    latent_history = torch.cat(
                        (
                            latent_history[:, 1:],
                            predicted.unsqueeze(1),
                        ),
                        dim=1,
                    )
            latent_history = latent_history.detach()

            optimizer.zero_grad(set_to_none=True)
            block_loss: torch.Tensor | None = None
            batch_totals = {name: 0.0 for name in totals}
            for rollout_index in range(config.rollout_steps):
                time_index = active_burn_in + rollout_index
                target_index = config.history_length + time_index
                target = sequences[:, target_index]
                true_previous = sequences[:, target_index - 1]
                current_state = latent_history[:, -1]
                predicted, extras = model(
                    latent_history, times[:, time_index]
                )

                recovery_velocity = motion_velocity_target(
                    current_state=current_state,
                    target=target,
                    memory_candidate=extras["memory_candidate"].detach(),
                    memory_gate=extras["spatial_memory_gate"].detach(),
                )
                true_scene_velocity = target - true_previous
                latent_loss = F.mse_loss(predicted, target)
                recovery_velocity_loss = F.smooth_l1_loss(
                    extras["predicted_velocity"],
                    recovery_velocity,
                    beta=0.25,
                )
                scene_velocity_loss = F.smooth_l1_loss(
                    extras["predicted_velocity"],
                    true_scene_velocity,
                    beta=0.25,
                )
                recovery_strength = recovery_velocity.abs().mean(
                    dim=1, keepdim=True
                )
                relative_recovery = recovery_strength / (
                    recovery_strength.mean(dim=(2, 3), keepdim=True)
                    + 1e-6
                )
                dynamic_loss = (
                    (1.0 + relative_recovery.clamp(max=4.0))
                    * (predicted - target).square()
                ).mean()
                target_motion_mask = (
                    relative_recovery / 2.0
                ).clamp(0.0, 1.0)
                motion_mask_loss = F.binary_cross_entropy(
                    extras["motion_mask"], target_motion_mask
                )
                step_loss = (
                    latent_loss
                    + config.recovery_velocity_weight
                    * recovery_velocity_loss
                    + config.scene_velocity_weight * scene_velocity_loss
                    + config.dynamic_loss_weight * dynamic_loss
                    + config.motion_mask_loss_weight * motion_mask_loss
                )
                scaled_loss = step_loss / config.rollout_steps
                block_loss = (
                    scaled_loss
                    if block_loss is None
                    else block_loss + scaled_loss
                )
                batch_totals["loss"] += step_loss.item()
                batch_totals["latent"] += latent_loss.item()
                batch_totals["recovery_velocity"] += (
                    recovery_velocity_loss.item()
                )
                batch_totals["scene_velocity"] += (
                    scene_velocity_loss.item()
                )
                batch_totals["dynamic"] += dynamic_loss.item()
                batch_totals["motion_mask"] += motion_mask_loss.item()

                latent_history = torch.cat(
                    (
                        latent_history[:, 1:],
                        predicted.unsqueeze(1),
                    ),
                    dim=1,
                )
                block_finished = (
                    (rollout_index + 1)
                    % config.truncated_backprop_steps
                    == 0
                    or rollout_index + 1 == config.rollout_steps
                )
                if block_finished:
                    if block_loss is None:
                        raise RuntimeError("empty recovery backprop block")
                    block_loss.backward()
                    block_loss = None
                    latent_history = latent_history.detach()

            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=1.0,
            )
            optimizer.step()
            for name, value in batch_totals.items():
                totals[name] += value / config.rollout_steps
            completed_batches += 1

            now = time.perf_counter()
            elapsed = carried_elapsed + now - started
            checkpoint_due = (
                now - last_checkpoint_time
                >= config.checkpoint_every_minutes * 60.0
            )
            stop_requested = stop_path.exists()
            runtime_reached = elapsed >= total_runtime_limit
            if checkpoint_due or stop_requested or runtime_reached:
                _save_resume(
                    config,
                    source_checkpoint,
                    model,
                    optimizer,
                    epoch=epoch,
                    next_batch=batch_index + 1,
                    metrics=current_metrics,
                    history=history,
                    elapsed_seconds=elapsed,
                )
                report_status = (
                    "Stopped by STOP file"
                    if stop_requested
                    else (
                        "Stopped at runtime limit"
                        if runtime_reached
                        else "Training in progress"
                    )
                )
                _write_status(
                    config,
                    status=report_status,
                    epoch=epoch,
                    batch=batch_index + 1,
                    batches=batch_count,
                    elapsed_seconds=elapsed,
                    history=history,
                    checkpoint=best_checkpoint,
                )
                _write_report(
                    config,
                    status=report_status,
                    trainable_parameters=trainable_parameters,
                    baseline_metrics=baseline_metrics,
                    history=history,
                )
                last_checkpoint_time = now
                print(
                    f"checkpoint | epoch {epoch} | "
                    f"batch {batch_index + 1}/{batch_count} | "
                    f"elapsed {elapsed / 60.0:.1f}m"
                )
                if stop_requested or runtime_reached:
                    stopped_checkpoint = _renderable_checkpoint(
                        source_checkpoint,
                        model,
                        epoch=epoch,
                        metrics=current_metrics,
                        recovery={
                            "status": report_status,
                            "elapsed_seconds": elapsed,
                        },
                    )
                    _atomic_torch_save(
                        stopped_checkpoint,
                        config.run_dir / "model_last.pt",
                    )
                    return best_checkpoint

        if completed_batches == 0:
            raise RuntimeError("no recovery fine-tune batches completed")
        current_metrics = _rollout_metrics(
            model,
            latents,
            polarities,
            device,
            config.history_length,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "training_loss": totals["loss"] / completed_batches,
            "latent_loss": totals["latent"] / completed_batches,
            "recovery_velocity_loss": (
                totals["recovery_velocity"] / completed_batches
            ),
            "scene_velocity_loss": (
                totals["scene_velocity"] / completed_batches
            ),
            "dynamic_loss": totals["dynamic"] / completed_batches,
            "motion_mask_loss": (
                totals["motion_mask"] / completed_batches
            ),
            "completed_batches": completed_batches,
            **current_metrics,
            "seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | loss {row['training_loss']:.5f} | "
            f"recovery {row['recovery_velocity_loss']:.5f} | "
            f"rollout {row['rollout_mse']:.5f} | "
            f"{epoch_seconds / 60.0:.1f}m"
        )
        completed_checkpoint = _renderable_checkpoint(
            source_checkpoint,
            model,
            epoch=epoch,
            metrics=current_metrics,
            recovery={"status": "epoch complete", "history": history},
        )
        _atomic_torch_save(
            completed_checkpoint, config.run_dir / "model_last.pt"
        )
        if current_metrics["rollout_mse"] < best_rollout_mse:
            best_rollout_mse = current_metrics["rollout_mse"]
            _atomic_torch_save(completed_checkpoint, best_checkpoint)

        elapsed = carried_elapsed + time.perf_counter() - started
        _save_resume(
            config,
            source_checkpoint,
            model,
            optimizer,
            epoch=epoch + 1,
            next_batch=0,
            metrics=current_metrics,
            history=history,
            elapsed_seconds=elapsed,
        )
        _write_status(
            config,
            status=f"Epoch {epoch} complete",
            epoch=epoch,
            batch=completed_batches,
            batches=batch_count,
            elapsed_seconds=elapsed,
            history=history,
            checkpoint=best_checkpoint,
        )
        _write_report(
            config,
            status=f"Epoch {epoch}/{config.epochs} complete",
            trainable_parameters=trainable_parameters,
            baseline_metrics=baseline_metrics,
            history=history,
        )
        resume_batch = 0

    elapsed = carried_elapsed + time.perf_counter() - started
    _write_status(
        config,
        status="Training complete",
        epoch=config.epochs,
        batch=0,
        batches=0,
        elapsed_seconds=elapsed,
        history=history,
        checkpoint=best_checkpoint,
    )
    _write_report(
        config,
        status="Training complete",
        trainable_parameters=trainable_parameters,
        baseline_metrics=baseline_metrics,
        history=history,
    )
    return best_checkpoint
