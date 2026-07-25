from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("basic", "attention"), default="basic")
    parser.add_argument("--data-dir", type=Path, default=Path("prototype_data/source_frames"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--input-threshold", type=float, default=0.5)
    parser.add_argument("--activation-threshold", type=float, default=0.5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")


def _training_config(args):
    from neural_bad_apple.training import TrainingConfig

    run_dir = args.run_dir or Path("prototype_runs") / args.model
    return TrainingConfig(
        frame_dir=args.data_dir,
        run_dir=run_dir,
        model_name=args.model,
        height=args.height,
        width=args.width,
        input_threshold=args.input_threshold,
        activation_threshold=args.activation_threshold,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_every=args.validation_every,
        seed=args.seed,
        device=args.device,
    )


def _extract(args) -> dict:
    from neural_bad_apple.data import extract_segment, find_source_video

    video_path = args.input or find_source_video()
    return extract_segment(
        video_path=video_path,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        start_seconds=args.start,
        end_seconds=args.end,
        fps=args.fps,
        force=args.force,
    )


def _autoregressive_training_config(args):
    from neural_bad_apple.autoregressive import AutoregressiveTrainingConfig

    return AutoregressiveTrainingConfig(
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        frame_dir=args.data_dir,
        run_dir=args.run_dir,
        hidden_channels=args.hidden_channels,
        max_residual_step=args.max_residual_step,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rollout_loss_weight=args.rollout_loss_weight,
        rollout_warmup_frames=args.rollout_warmup_frames,
        seed=args.seed,
        device=args.device,
    )


def _hybrid_training_config(args):
    from neural_bad_apple.hybrid import HybridTrainingConfig

    return HybridTrainingConfig(
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        frame_dir=args.data_dir,
        run_dir=args.run_dir,
        history_length=args.history_length,
        minimum_rollout_steps=args.minimum_rollout_steps,
        rollout_steps=args.rollout_steps,
        base_channels=args.base_channels,
        memory_token_count=args.memory_tokens,
        fourier_frequencies=args.fourier_frequencies,
        max_residual_step=args.max_residual_step,
        memory_temperature=args.memory_temperature,
        memory_entropy_weight=args.memory_entropy_weight,
        polarity_loss_weight=args.polarity_loss_weight,
        canonicalize_polarity=not args.disable_polarity_canonicalization,
        polarity_tracking_method=args.polarity_tracking_method,
        polarity_switch_penalty=args.polarity_switch_penalty,
        scene_cut_minimum_distance=args.scene_cut_minimum_distance,
        latent_noise_standard_deviation=args.latent_noise,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )


def _hybrid_v4_training_config(args):
    from neural_bad_apple.hybrid_v4 import HybridV4TrainingConfig

    return HybridV4TrainingConfig(
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        frame_dir=args.data_dir,
        run_dir=args.run_dir,
        history_length=args.history_length,
        minimum_rollout_steps=args.minimum_rollout_steps,
        rollout_steps=args.rollout_steps,
        truncated_backprop_steps=args.truncated_backprop_steps,
        base_channels=args.base_channels,
        anchor_count=args.anchors,
        anchor_temperature=args.anchor_temperature,
        maximum_anchor_gate=args.maximum_anchor_gate,
        anchor_minimum_distance=args.anchor_minimum_distance,
        fourier_frequencies=args.fourier_frequencies,
        max_velocity_step=args.max_velocity_step,
        use_dual_velocity=args.dual_velocity,
        max_fast_velocity_step=args.max_fast_velocity_step,
        velocity_loss_weight=args.velocity_loss_weight,
        slow_velocity_loss_weight=args.slow_velocity_loss_weight,
        fast_velocity_loss_weight=args.fast_velocity_loss_weight,
        fast_velocity_dynamic_weight=args.fast_velocity_dynamic_weight,
        dynamic_loss_weight=args.dynamic_loss_weight,
        motion_mask_loss_weight=args.motion_mask_loss_weight,
        anchor_loss_weight=args.anchor_loss_weight,
        polarity_loss_weight=args.polarity_loss_weight,
        polarity_calibration_steps=args.polarity_calibration_steps,
        polarity_calibration_learning_rate=(
            args.polarity_calibration_learning_rate
        ),
        polarity_tracking_method=args.polarity_tracking_method,
        polarity_switch_penalty=args.polarity_switch_penalty,
        latent_noise_standard_deviation=args.latent_noise,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warm_start_checkpoint=args.warm_start_checkpoint,
        fast_head_only_epochs=args.fast_head_only_epochs,
        motion_only_epochs=args.motion_only_epochs,
        seed=args.seed,
        device=args.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural network dreams Bad Apple — 45–60 second prototype"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="extract source frames")
    extract_parser.add_argument("--input", type=Path)
    extract_parser.add_argument(
        "--output-dir", type=Path, default=Path("prototype_data/source_frames")
    )
    extract_parser.add_argument(
        "--manifest", type=Path, default=Path("prototype_data/manifest.json")
    )
    extract_parser.add_argument("--start", type=float, default=45.0)
    extract_parser.add_argument("--end", type=float, default=60.0)
    extract_parser.add_argument("--fps", type=float)
    extract_parser.add_argument("--force", action="store_true")

    train_parser = subparsers.add_parser("train", help="train one model block")
    _add_training_arguments(train_parser)

    reconstruct_parser = subparsers.add_parser(
        "reconstruct", help="render neuron activations from a checkpoint"
    )
    reconstruct_parser.add_argument("--checkpoint", type=Path, required=True)
    reconstruct_parser.add_argument(
        "--data-dir", type=Path, default=Path("prototype_data/source_frames")
    )
    reconstruct_parser.add_argument("--output-dir", type=Path)
    reconstruct_parser.add_argument("--batch-size", type=int, default=8)
    reconstruct_parser.add_argument("--device", default="auto")
    reconstruct_parser.add_argument("--fps", type=float)
    reconstruct_parser.add_argument("--no-video", action="store_true")

    train_ar_parser = subparsers.add_parser(
        "train-ar",
        help="train next-latent prediction on the frozen autoencoder",
    )
    train_ar_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("prototype_runs/basic_full/model_best.pt"),
    )
    train_ar_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prototype_data/source_frames"),
    )
    train_ar_parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("prototype_runs/autoregressive"),
    )
    train_ar_parser.add_argument("--hidden-channels", type=int, default=64)
    train_ar_parser.add_argument(
        "--max-residual-step", type=float, default=0.5
    )
    train_ar_parser.add_argument("--sequence-length", type=int, default=16)
    train_ar_parser.add_argument("--epochs", type=int, default=30)
    train_ar_parser.add_argument("--batch-size", type=int, default=4)
    train_ar_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_ar_parser.add_argument(
        "--rollout-loss-weight", type=float, default=0.1
    )
    train_ar_parser.add_argument(
        "--rollout-warmup-frames", type=int, default=16
    )
    train_ar_parser.add_argument("--seed", type=int, default=7)
    train_ar_parser.add_argument("--device", default="auto")

    rollout_ar_parser = subparsers.add_parser(
        "rollout-ar",
        help="render free-running latent drift from either temporal model",
    )
    rollout_ar_parser.add_argument("--checkpoint", type=Path, required=True)
    rollout_ar_parser.add_argument(
        "--autoencoder-checkpoint", type=Path
    )
    rollout_ar_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prototype_data/source_frames"),
    )
    rollout_ar_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prototype_outputs/autoregressive"),
    )
    rollout_ar_parser.add_argument("--batch-size", type=int, default=8)
    rollout_ar_parser.add_argument("--device", default="auto")
    rollout_ar_parser.add_argument("--fps", type=float)
    rollout_ar_parser.add_argument("--warmup-frames", type=int)
    rollout_ar_parser.add_argument("--no-video", action="store_true")

    train_hybrid_parser = subparsers.add_parser(
        "train-hybrid",
        help="train temporal U-Net with time-addressed scene memory",
    )
    train_hybrid_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("prototype_runs/basic_full/model_best.pt"),
    )
    train_hybrid_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prototype_data/source_frames"),
    )
    train_hybrid_parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("prototype_runs/hybrid_memory"),
    )
    train_hybrid_parser.add_argument("--history-length", type=int, default=16)
    train_hybrid_parser.add_argument(
        "--minimum-rollout-steps", type=int, default=4
    )
    train_hybrid_parser.add_argument("--rollout-steps", type=int, default=16)
    train_hybrid_parser.add_argument("--base-channels", type=int, default=16)
    train_hybrid_parser.add_argument("--memory-tokens", type=int, default=8)
    train_hybrid_parser.add_argument(
        "--fourier-frequencies", type=int, default=6
    )
    train_hybrid_parser.add_argument(
        "--max-residual-step", type=float, default=0.5
    )
    train_hybrid_parser.add_argument(
        "--memory-temperature", type=float, default=0.5
    )
    train_hybrid_parser.add_argument(
        "--memory-entropy-weight", type=float, default=1e-3
    )
    train_hybrid_parser.add_argument(
        "--polarity-loss-weight", type=float, default=0.2
    )
    train_hybrid_parser.add_argument(
        "--scene-cut-minimum-distance", type=int, default=15
    )
    train_hybrid_parser.add_argument(
        "--disable-polarity-canonicalization", action="store_true"
    )
    train_hybrid_parser.add_argument(
        "--polarity-tracking-method",
        choices=("temporal", "border"),
        default="temporal",
    )
    train_hybrid_parser.add_argument(
        "--polarity-switch-penalty", type=float, default=0.05
    )
    train_hybrid_parser.add_argument("--latent-noise", type=float, default=0.03)
    train_hybrid_parser.add_argument("--epochs", type=int, default=20)
    train_hybrid_parser.add_argument("--batch-size", type=int, default=2)
    train_hybrid_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_hybrid_parser.add_argument("--seed", type=int, default=7)
    train_hybrid_parser.add_argument("--device", default="auto")

    train_hybrid_v4_parser = subparsers.add_parser(
        "train-hybrid-v4",
        help="train velocity model with softly bleeding scene anchors",
    )
    train_hybrid_v4_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("prototype_runs/basic_full/model_best.pt"),
    )
    train_hybrid_v4_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prototype_data/source_frames"),
    )
    train_hybrid_v4_parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("prototype_runs/hybrid_v4_bleed"),
    )
    train_hybrid_v4_parser.add_argument(
        "--history-length", type=int, default=16
    )
    train_hybrid_v4_parser.add_argument(
        "--minimum-rollout-steps", type=int, default=4
    )
    train_hybrid_v4_parser.add_argument(
        "--rollout-steps", type=int, default=32
    )
    train_hybrid_v4_parser.add_argument(
        "--truncated-backprop-steps", type=int, default=4
    )
    train_hybrid_v4_parser.add_argument(
        "--base-channels", type=int, default=8
    )
    train_hybrid_v4_parser.add_argument("--anchors", type=int, default=16)
    train_hybrid_v4_parser.add_argument(
        "--anchor-temperature", type=float, default=0.03
    )
    train_hybrid_v4_parser.add_argument(
        "--maximum-anchor-gate", type=float, default=0.35
    )
    train_hybrid_v4_parser.add_argument(
        "--anchor-minimum-distance", type=int, default=8
    )
    train_hybrid_v4_parser.add_argument(
        "--fourier-frequencies", type=int, default=6
    )
    train_hybrid_v4_parser.add_argument(
        "--max-velocity-step", type=float, default=0.5
    )
    train_hybrid_v4_parser.add_argument(
        "--dual-velocity", action="store_true"
    )
    train_hybrid_v4_parser.add_argument(
        "--max-fast-velocity-step", type=float, default=2.0
    )
    train_hybrid_v4_parser.add_argument(
        "--velocity-loss-weight", type=float, default=0.5
    )
    train_hybrid_v4_parser.add_argument(
        "--slow-velocity-loss-weight", type=float, default=0.5
    )
    train_hybrid_v4_parser.add_argument(
        "--fast-velocity-loss-weight", type=float, default=1.0
    )
    train_hybrid_v4_parser.add_argument(
        "--fast-velocity-dynamic-weight", type=float, default=4.0
    )
    train_hybrid_v4_parser.add_argument(
        "--dynamic-loss-weight", type=float, default=0.5
    )
    train_hybrid_v4_parser.add_argument(
        "--motion-mask-loss-weight", type=float, default=0.05
    )
    train_hybrid_v4_parser.add_argument(
        "--anchor-loss-weight", type=float, default=0.01
    )
    train_hybrid_v4_parser.add_argument(
        "--polarity-loss-weight", type=float, default=0.2
    )
    train_hybrid_v4_parser.add_argument(
        "--polarity-calibration-steps", type=int, default=500
    )
    train_hybrid_v4_parser.add_argument(
        "--polarity-calibration-learning-rate",
        type=float,
        default=0.03,
    )
    train_hybrid_v4_parser.add_argument(
        "--polarity-tracking-method",
        choices=("temporal", "border"),
        default="temporal",
    )
    train_hybrid_v4_parser.add_argument(
        "--polarity-switch-penalty", type=float, default=0.05
    )
    train_hybrid_v4_parser.add_argument(
        "--latent-noise", type=float, default=0.03
    )
    train_hybrid_v4_parser.add_argument("--epochs", type=int, default=12)
    train_hybrid_v4_parser.add_argument("--batch-size", type=int, default=2)
    train_hybrid_v4_parser.add_argument(
        "--learning-rate", type=float, default=5e-4
    )
    train_hybrid_v4_parser.add_argument(
        "--warm-start-checkpoint", type=Path
    )
    train_hybrid_v4_parser.add_argument(
        "--fast-head-only-epochs", type=int, default=0
    )
    train_hybrid_v4_parser.add_argument(
        "--motion-only-epochs", type=int, default=0
    )
    train_hybrid_v4_parser.add_argument("--seed", type=int, default=7)
    train_hybrid_v4_parser.add_argument("--device", default="auto")

    all_parser = subparsers.add_parser(
        "all", help="extract, train, and reconstruct one experiment"
    )
    all_parser.add_argument("--input", type=Path)
    all_parser.add_argument(
        "--output-dir", type=Path, default=Path("prototype_data/source_frames")
    )
    all_parser.add_argument(
        "--manifest", type=Path, default=Path("prototype_data/manifest.json")
    )
    all_parser.add_argument("--start", type=float, default=45.0)
    all_parser.add_argument("--end", type=float, default=60.0)
    all_parser.add_argument("--fps", type=float)
    all_parser.add_argument("--force", action="store_true")
    _add_training_arguments(all_parser)

    args = parser.parse_args()

    if args.command == "extract":
        print(json.dumps(_extract(args), indent=2))
        return

    if args.command == "train":
        from neural_bad_apple.training import train

        checkpoint = train(_training_config(args))
        print(f"Best checkpoint: {checkpoint.resolve()}")
        return

    if args.command == "reconstruct":
        from neural_bad_apple.rendering import reconstruct

        checkpoint_name = args.checkpoint.parent.name
        output_dir = args.output_dir or Path("prototype_outputs") / checkpoint_name
        summary = reconstruct(
            checkpoint_path=args.checkpoint,
            frame_dir=args.data_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device_name=args.device,
            make_videos=not args.no_video,
            fps=args.fps,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-ar":
        from neural_bad_apple.autoregressive import train_autoregressor

        checkpoint = train_autoregressor(
            _autoregressive_training_config(args)
        )
        print(f"Best autoregressive checkpoint: {checkpoint.resolve()}")
        return

    if args.command == "rollout-ar":
        from neural_bad_apple.drift_rendering import render_drift

        summary = render_drift(
            checkpoint_path=args.checkpoint,
            autoencoder_checkpoint_path=args.autoencoder_checkpoint,
            frame_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device_name=args.device,
            fps=args.fps,
            make_videos=not args.no_video,
            warmup_frames=args.warmup_frames,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-hybrid":
        from neural_bad_apple.hybrid import train_hybrid

        checkpoint = train_hybrid(_hybrid_training_config(args))
        print(f"Best hybrid checkpoint: {checkpoint.resolve()}")
        return

    if args.command == "train-hybrid-v4":
        from neural_bad_apple.hybrid_v4 import train_hybrid_v4

        checkpoint = train_hybrid_v4(
            _hybrid_v4_training_config(args)
        )
        print(f"Best hybrid v4 checkpoint: {checkpoint.resolve()}")
        return

    if args.command == "all":
        from neural_bad_apple.rendering import reconstruct
        from neural_bad_apple.training import train

        extraction = _extract(args)
        args.data_dir = args.output_dir
        config = _training_config(args)
        checkpoint = train(config)
        summary = reconstruct(
            checkpoint_path=checkpoint,
            frame_dir=args.data_dir,
            output_dir=Path("prototype_outputs") / args.model,
            batch_size=args.batch_size,
            device_name=args.device,
            make_videos=True,
            fps=float(extraction["frames"]["fps"]),
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
