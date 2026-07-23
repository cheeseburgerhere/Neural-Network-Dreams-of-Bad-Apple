from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from neural_bad_apple.data import FrameDataset
from neural_bad_apple.models import build_model


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
