from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from neural_bad_apple.data import FrameDataset
from neural_bad_apple.autoregressive import (
    LatentAutoregressor,
    LatentWindowDataset,
    rollout_latents,
)
from neural_bad_apple.models import build_model
from neural_bad_apple.hybrid import (
    HybridTemporalMemoryModel,
    HybridWindowDataset,
    detect_frame_polarity,
    rollout_hybrid_latents,
    select_scene_memory_indices,
    track_frame_polarity,
)
from neural_bad_apple.hybrid_v4 import (
    BleedingSceneMemoryModel,
    select_covered_anchor_indices,
)
from neural_bad_apple.polarity import interpolate_polarity_logits
from neural_bad_apple.silhouette import VARIANTS, _variant_prediction
from neural_bad_apple.recovery import (
    configure_recovery_modules,
    motion_velocity_target,
)
from neural_bad_apple.reporting import (
    write_render_report,
    write_training_report,
)


class ModelTests(unittest.TestCase):
    def test_basic_model_preserves_pixel_grid(self) -> None:
        model, _ = build_model("basic", base_channels=4, latent_channels=8)
        inputs = torch.rand(2, 1, 96, 128)
        logits, extras = model(inputs)
        self.assertEqual(logits.shape, inputs.shape)
        self.assertEqual(extras, {})

    def test_attention_model_supports_non_stride_aligned_grids(self) -> None:
        model, _ = build_model("attention", base_channels=4, latent_channels=8)
        inputs = torch.rand(1, 1, 150, 200)
        logits, extras = model(inputs)
        self.assertEqual(logits.shape, inputs.shape)
        self.assertIn("attention", extras)
        self.assertEqual(extras["attention"].shape[1], 1)

    def test_autoencoder_exposes_replaceable_encoder_and_decoder(self) -> None:
        model, _ = build_model("basic", base_channels=4, latent_channels=8)
        inputs = torch.rand(2, 1, 95, 127)
        latents, _ = model.encode(inputs)
        logits = model.decode(latents, inputs.shape[-2:])
        self.assertEqual(logits.shape, inputs.shape)


class AutoregressiveTests(unittest.TestCase):
    def test_predictor_preserves_latent_grid(self) -> None:
        model = LatentAutoregressor(
            latent_channels=8, hidden_channels=12
        )
        current = torch.rand(2, 8, 6, 8)
        predicted, hidden = model.step(current)
        self.assertEqual(predicted.shape, current.shape)
        self.assertEqual(hidden.shape, (2, 12, 6, 8))

    def test_latent_windows_include_the_next_target(self) -> None:
        latents = torch.rand(10, 8, 2, 2)
        dataset = LatentWindowDataset(latents, sequence_length=4)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset[0].shape, (5, 8, 2, 2))

    def test_rollout_preserves_context_before_source_cutoff(self) -> None:
        model = LatentAutoregressor(
            latent_channels=8, hidden_channels=12
        )
        context = torch.rand(3, 8, 2, 2)
        rollout = rollout_latents(model, context, frame_count=6)
        self.assertEqual(rollout.shape, (6, 8, 2, 2))
        self.assertTrue(torch.equal(rollout[:3], context))


class HybridTests(unittest.TestCase):
    def test_temporal_unet_and_memory_preserve_latent_grid(self) -> None:
        model = HybridTemporalMemoryModel(
            latent_channels=8,
            latent_height=4,
            latent_width=6,
            base_channels=4,
            memory_token_count=3,
            fourier_frequencies=2,
        )
        history = torch.rand(2, 8, 8, 4, 6)
        times = torch.tensor([0.25, 0.75])
        predicted, extras = model(history, times)
        self.assertEqual(predicted.shape, (2, 8, 4, 6))
        self.assertEqual(extras["memory_weights"].shape, (2, 3))
        self.assertTrue(
            torch.allclose(
                extras["memory_weights"].sum(dim=1), torch.ones(2)
            )
        )
        _, weights, gates = model.address_memory(times)
        self.assertEqual(weights.shape, (2, 3))
        self.assertEqual(gates.shape, (2, 1))

    def test_hybrid_window_contains_rollout_targets(self) -> None:
        latents = torch.rand(20, 8, 2, 2)
        dataset = HybridWindowDataset(
            latents, history_length=8, rollout_steps=3
        )
        sequence, times, polarities = dataset[0]
        self.assertEqual(sequence.shape, (11, 8, 2, 2))
        self.assertEqual(times.shape, (3,))
        self.assertEqual(polarities.shape, (3,))

    def test_hybrid_rollout_preserves_context(self) -> None:
        model = HybridTemporalMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            memory_token_count=3,
            fourier_frequencies=2,
        )
        context = torch.rand(8, 8, 2, 2)
        rollout = rollout_hybrid_latents(model, context, frame_count=10)
        self.assertEqual(rollout.shape, (10, 8, 2, 2))
        self.assertTrue(torch.equal(rollout[:8], context))

    def test_border_polarity_detection(self) -> None:
        black_background = torch.zeros(1, 1, 8, 8)
        white_background = torch.ones(1, 1, 8, 8)
        frames = torch.cat((black_background, white_background), dim=0)
        self.assertTrue(
            torch.equal(
                detect_frame_polarity(frames), torch.tensor([0.0, 1.0])
            )
        )

    def test_temporal_polarity_ignores_border_occlusion(self) -> None:
        white_background = torch.ones(1, 1, 8, 8)
        white_background[:, :, 2:6, 2:6] = 0
        occluded_border = white_background.clone()
        occluded_border[:, :, 0, :] = 0
        occluded_border[:, :, -1, :] = 0
        occluded_border[:, :, :, 0] = 0
        occluded_border[:, :, :, -1] = 0
        frames = torch.cat(
            (white_background, occluded_border, white_background), dim=0
        )

        self.assertTrue(
            torch.equal(
                detect_frame_polarity(frames),
                torch.tensor([1.0, 0.0, 1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                track_frame_polarity(frames),
                torch.tensor([1.0, 1.0, 1.0]),
            )
        )

    def test_temporal_polarity_tracks_a_real_global_inversion(self) -> None:
        white_background = torch.ones(1, 1, 8, 8)
        frames = torch.cat(
            (white_background, 1.0 - white_background), dim=0
        )
        polarities = track_frame_polarity(frames)
        canonical = torch.where(
            polarities[:, None, None, None] > 0.5,
            1.0 - frames,
            frames,
        )

        self.assertTrue(
            torch.equal(polarities, torch.tensor([1.0, 0.0]))
        )
        self.assertTrue(torch.equal(canonical[0], canonical[1]))

    def test_scene_memories_are_spread_across_large_changes(self) -> None:
        latents = torch.zeros(30, 2, 2, 2)
        latents[10:] = 2.0
        latents[20:] = -2.0
        indices = select_scene_memory_indices(
            latents, memory_token_count=3, minimum_distance=5
        )
        self.assertEqual(len(indices), 3)
        self.assertIn(10, indices.tolist())
        self.assertIn(20, indices.tolist())

    def test_v4_uses_two_temporally_local_scene_anchors(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
        )
        _, weights, maximum_gate = model.address_memory(
            torch.tensor([0.1, 0.9])
        )
        self.assertEqual(weights.shape, (2, 4))
        self.assertTrue(torch.all((weights > 0).sum(dim=1) == 2))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.all(maximum_gate <= 0.35))

    def test_v4_velocity_and_bleed_fusion_preserve_latent_grid(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            maximum_anchor_gate=0.3,
        )
        history = torch.rand(2, 8, 8, 2, 2)
        predicted, extras = model(history, torch.tensor([0.2, 0.8]))
        self.assertEqual(predicted.shape, (2, 8, 2, 2))
        self.assertEqual(
            extras["predicted_velocity"].shape, predicted.shape
        )
        self.assertEqual(extras["motion_mask"].shape, (2, 1, 2, 2))
        self.assertEqual(
            extras["spatial_memory_gate"].shape, (2, 1, 2, 2)
        )
        self.assertTrue(
            torch.all(extras["spatial_memory_gate"] <= 0.3)
        )

    def test_v4_dual_velocity_adds_sparse_fast_motion(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            use_dual_velocity=True,
            max_fast_velocity_step=2.0,
        )
        history = torch.rand(2, 8, 8, 2, 2)
        _, extras = model(history, torch.tensor([0.2, 0.8]))

        self.assertTrue(
            torch.allclose(
                extras["predicted_velocity"],
                extras["slow_velocity"] + extras["fast_velocity"],
            )
        )
        self.assertTrue(
            torch.all(
                extras["fast_velocity"].abs()
                <= 2.0 * extras["motion_mask"] + 1e-6
            )
        )

    def test_v4_checkpoint_can_add_zero_initialized_fast_head(self) -> None:
        original = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
        )
        upgraded = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            use_dual_velocity=True,
        )
        missing, unexpected = upgraded.load_state_dict(
            original.state_dict(), strict=False
        )

        self.assertEqual(
            set(missing),
            {
                "fast_velocity_head.weight",
                "fast_velocity_head.bias",
            },
        )
        self.assertEqual(unexpected, [])
        self.assertTrue(
            torch.all(upgraded.fast_velocity_head.weight == 0)
        )

    def test_v4_anchor_selection_preserves_timeline_coverage(self) -> None:
        latents = torch.zeros(100, 2, 2, 2)
        latents[30:] = 2.0
        latents[70:] = -2.0
        indices = select_covered_anchor_indices(
            latents, anchor_count=8, minimum_distance=5
        )
        gaps = indices[1:] - indices[:-1]
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[-1].item(), 99)
        self.assertLessEqual(gaps.max().item(), 33)

    def test_v42_physical_fourier_features_keep_absolute_period(self) -> None:
        short = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=3,
            time_basis="seconds",
            timeline_seconds=15.0,
            time_fourier_base_frequency=0.0625,
        )
        long = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=3,
            time_basis="seconds",
            timeline_seconds=219.0,
            time_fourier_base_frequency=0.0625,
        )

        short_features = short._time_features(torch.tensor([1.0 / 15.0]))
        long_features = long._time_features(torch.tensor([1.0 / 219.0]))
        self.assertTrue(
            torch.allclose(
                short_features[:, 1:],
                long_features[:, 1:],
                atol=1e-6,
            )
        )

    def test_v42_scales_temperature_to_anchor_spacing(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            anchor_temperature_mode="spacing",
            anchor_temperature_ratio=0.5,
        )
        latents = torch.rand(100, 8, 2, 2)
        indices = torch.tensor([0, 20, 60, 99])
        model.initialize_memory(latents, indices)
        expected = 0.5 * torch.diff(
            indices.float() / 99.0
        ).median().item()
        self.assertAlmostEqual(model.anchor_temperature, expected)

    def test_v42_cut_gate_preserves_bleed_and_respects_cap(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            maximum_anchor_gate=0.3,
            maximum_transition_gate=0.65,
            use_cut_gate=True,
        )
        with torch.no_grad():
            model.cut_gate_head.bias.fill_(10.0)
        history = torch.rand(2, 8, 8, 2, 2)
        _, extras = model(history, torch.tensor([0.2, 0.8]))

        self.assertTrue(
            torch.all(
                extras["spatial_memory_gate"]
                >= extras["base_memory_gate"]
            )
        )
        self.assertTrue(
            torch.all(extras["spatial_memory_gate"] <= 0.65)
        )
        self.assertTrue(torch.all(extras["cut_gate"] > 0.99))

    def test_v42_separate_polarity_spline_interpolates_knots(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            polarity_knot_count=3,
        )
        with torch.no_grad():
            model.polarity_spline_logits.copy_(
                torch.tensor([-2.0, 2.0, -2.0])
            )

        predictions = model.predict_polarity(
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        )[:, 0]

        self.assertTrue(
            torch.allclose(
                predictions, torch.tensor([-2.0, 0.0, 2.0, 0.0, -2.0])
            )
        )
        self.assertTrue(
            torch.allclose(
                predictions,
                interpolate_polarity_logits(
                    model.polarity_spline_logits,
                    torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
                ),
            )
        )

    def test_silhouette_baseline_keeps_original_prediction(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            use_dual_velocity=True,
            use_cut_gate=True,
        )
        history = torch.rand(1, 8, 8, 2, 2)
        times = torch.tensor([0.5])
        expected, _ = model(history, times)
        actual, _ = _variant_prediction(
            model, history, times, VARIANTS["baseline"]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_moving_bleed_only_adds_bounded_memory_correction(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            use_dual_velocity=True,
            use_cut_gate=True,
            maximum_transition_gate=0.65,
        )
        history = torch.rand(1, 8, 8, 2, 2)
        _, extras = _variant_prediction(
            model, history, torch.tensor([0.5]), VARIANTS["moving-1.0"]
        )
        self.assertTrue(
            torch.all(
                extras["diagnostic_gate"]
                >= extras["spatial_memory_gate"]
            )
        )
        self.assertTrue(torch.all(extras["diagnostic_gate"] <= 0.65))

    def test_recovery_velocity_accounts_for_frozen_memory_fusion(self) -> None:
        current = torch.rand(2, 8, 2, 2)
        target = torch.rand(2, 8, 2, 2)
        memory = torch.rand(2, 8, 2, 2)
        gate = torch.rand(2, 1, 2, 2) * 0.65

        velocity = motion_velocity_target(
            current, target, memory, gate
        )
        motion = current + velocity
        reconstructed = motion + gate * (memory - motion)

        self.assertTrue(torch.allclose(reconstructed, target, atol=1e-6))

    def test_recovery_fine_tune_freezes_timeline_and_memory(self) -> None:
        model = BleedingSceneMemoryModel(
            latent_channels=8,
            latent_height=2,
            latent_width=2,
            base_channels=4,
            anchor_count=4,
            fourier_frequencies=2,
            use_dual_velocity=True,
            use_cut_gate=True,
            polarity_knot_count=8,
        )
        trainable = configure_recovery_modules(model)

        self.assertGreater(trainable, 0)
        self.assertFalse(model.memory_tokens.requires_grad)
        self.assertFalse(model.polarity_spline_logits.requires_grad)
        self.assertFalse(model.time_encoder[0].weight.requires_grad)
        self.assertTrue(model.decoder_high[0].weight.requires_grad)
        self.assertTrue(model.fast_velocity_head.weight.requires_grad)


class ReportingTests(unittest.TestCase):
    def test_training_report_keeps_configuration_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = write_training_report(
                Path(directory),
                title="V4.2 test",
                status="Prepared",
                architecture=("Physical time", "Long burn-in"),
                config={"burn_in_steps": 128},
                command="python prototype.py train-hybrid-v42",
                history=[
                    {
                        "epoch": 1,
                        "training_stage": "memory-frozen",
                        "mean_burn_in_steps": 80,
                        "active_rollout_steps": 8,
                        "training_loss": 1.0,
                        "rollout_mse": 0.5,
                        "peak_frame_mse": 1.2,
                        "seconds": 10.0,
                    }
                ],
            )
            contents = report.read_text(encoding="utf-8")
            self.assertIn("Physical time", contents)
            self.assertIn('"burn_in_steps": 128', contents)
            self.assertIn("memory-frozen", contents)

    def test_metrics_only_render_report_does_not_claim_videos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = write_render_report(
                root,
                title="V4.2 metrics",
                checkpoint=root / "model_best.pt",
                summary={"frame_count": 10, "fps": 30.0},
            )
            contents = report.read_text(encoding="utf-8")
            self.assertIn("metrics-only run", contents)
            self.assertNotIn("`comparison.mp4`", contents)


class DatasetTests(unittest.TestCase):
    def test_source_frame_becomes_binary_neuron_targets(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            frame_path = Path(directory) / "frame_00000.png"
            Image.new("L", (8, 8), color=128).save(frame_path)
            dataset = FrameDataset(Path(directory), height=4, width=6)
            pixels, name = dataset[0]
            self.assertEqual(pixels.shape, (1, 4, 6))
            self.assertEqual(name, frame_path.name)
            self.assertTrue(torch.all(pixels == 1))


if __name__ == "__main__":
    unittest.main()
