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
