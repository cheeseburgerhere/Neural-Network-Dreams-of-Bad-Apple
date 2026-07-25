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
