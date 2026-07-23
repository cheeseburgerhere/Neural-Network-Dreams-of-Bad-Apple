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
