import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from neural_bad_apple.hybrid_v4 import BleedingSceneMemoryModel
from neural_bad_apple.memory_baseline import load_reference, memory_only_latents, summarize


class MemoryBaselineTests(unittest.TestCase):
    def test_uses_memory_not_temporal_forward(self):
        model = BleedingSceneMemoryModel(2, 4, 4, base_channels=4, anchor_count=2)
        with torch.no_grad():
            model.memory_tokens[0].fill_(2)
            model.memory_tokens[1].fill_(6)
            model.anchor_times.copy_(torch.tensor([0.0, 1.0]))
        with patch.object(model, "forward", side_effect=AssertionError("temporal model called")):
            result = memory_only_latents(model, torch.tensor([0.0, 0.5, 1.0]))
        self.assertTrue(torch.allclose(result[0], torch.full_like(result[0], 2)))
        self.assertTrue(torch.allclose(result[1], torch.full_like(result[1], 4)))
        self.assertTrue(torch.allclose(result[2], torch.full_like(result[2], 6)))

    def test_zero_anchors_rejected(self):
        model = BleedingSceneMemoryModel(2, 4, 4, base_channels=4, anchor_count=0)
        with self.assertRaises(ValueError):
            memory_only_latents(model, torch.tensor([0.5]))

    def test_reference_must_align(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "seconds", "rollout_binary_error"])
                writer.writerows([[0, 0, .1], [1, 1 / 30, .2]])
            self.assertEqual(len(load_reference(path, 2, 30)), 2)
            with self.assertRaises(ValueError):
                load_reference(path, 3, 30)
            with self.assertRaises(ValueError):
                load_reference(path, 2, 60)

    def test_scores_exclude_reference_warmup(self):
        rows = [dict(memory_binary_error=m, rollout_binary_error=r,
                     memory_mean_binary_iou=.5, memory_boundary_f1=.4)
                for m, r in [(1, 0), (.2, .1), (.4, .2)]]
        score = summarize(rows, 1)
        self.assertAlmostEqual(score["memory_binary_error"], .3)
        self.assertAlmostEqual(score["rollout_binary_error"], .15)
        self.assertAlmostEqual(score["temporal_error_reduction_fraction"], .5)
        self.assertEqual(score["scored_frames"], 2)


if __name__ == "__main__":
    unittest.main()
