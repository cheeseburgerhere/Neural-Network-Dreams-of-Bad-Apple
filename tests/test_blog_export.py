import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from neural_bad_apple.blog_export import checkpoint_path, panel, pixel_metrics, restore_polarity, sha256, summarize
from neural_bad_apple.blog_package import selected_models
from neural_bad_apple.hybrid_v4 import BleedingSceneMemoryModel
from neural_bad_apple.polarity import PolarityFixConfig, fix_checkpoint_polarity


class BlogExportTests(unittest.TestCase):
    def test_polarity_restoration_is_per_frame(self):
        probability = torch.tensor([.2, .3]).reshape(2, 1, 1, 1)
        restored = restore_polarity(probability, torch.tensor([False, True]))
        self.assertTrue(torch.allclose(restored.flatten(), torch.tensor([.2, .7])))

    def test_pixel_error_and_class_mean_iou(self):
        target = np.array([[False, True], [False, True]])
        self.assertEqual(pixel_metrics(target, target), (0., 1.))
        self.assertEqual(pixel_metrics(~target, target), (1., 0.))

    def test_score_warmup_exclusion(self):
        rows = [dict(rollout_error=r, teacher_error=.1, memory_error=m, polarity_correct=1)
                for r, m in [(1., .1), (.1, .3), (.2, .6)]]
        result = summarize(rows, warmup=1)
        self.assertAlmostEqual(result['rollout_error'], .15)
        self.assertAlmostEqual(result['memory_error'], .45)
        self.assertAlmostEqual(result['full_error_reduction_fraction'], 2/3)
        self.assertEqual(result['scored_frames'], 2)
        with self.assertRaises(ValueError):
            summarize(rows, warmup=3)

    def test_panel_preserves_content_area(self):
        image = panel(np.ones((384, 512), dtype=bool), 'Source', 'frame 0')
        self.assertEqual(image.size, (512, 442))
        self.assertTrue((np.asarray(image)[58:] == 255).all())

    def test_220_checkpoint_prefers_separate_calibrated_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('neural_bad_apple.blog_export.ROOT', root):
                raw = root / 'prototype_runs/anchors_220/model_best.pt'
                fixed = root / 'prototype_runs/anchors_220_polarity/model_best.pt'
                self.assertEqual(checkpoint_path(220), raw)
                fixed.parent.mkdir(parents=True)
                fixed.touch()
                self.assertEqual(checkpoint_path(220), fixed)

    def test_main_table_selects_only_one_220_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with self.assertRaises(FileNotFoundError):
                selected_models(work)
            for expected in ('anchors_220', 'anchors_220_shared_polarity', 'anchors_220_polarity'):
                folder = work / expected
                folder.mkdir()
                (folder / 'summary.json').touch()
                tags, primary = selected_models(work)
                self.assertEqual(primary, expected)
                self.assertEqual(tags, [f'anchors_{n:03d}' for n in (0, 16, 32, 55, 110)] + [expected])

    def test_calibration_preserves_original_checkpoint_and_latent_dynamics(self):
        torch.manual_seed(7)
        kwargs = dict(latent_channels=2, latent_height=4, latent_width=4,
                      base_channels=4, anchor_count=2)
        original = BleedingSceneMemoryModel(**kwargs).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'model.pt'
            target = root / 'targets.csv'
            torch.save(dict(model_type='hybrid_v4_bleeding_memory', model_kwargs=kwargs,
                            state_dict=original.state_dict()), raw)
            target.write_text('target_polarity\n' + '0\n'*8 + '1\n'*8, encoding='utf-8')
            original_hash = sha256(raw)
            fixed_path = fix_checkpoint_polarity(PolarityFixConfig(
                checkpoint=raw, target_csv=target, run_dir=root/'fixed',
                knot_counts=(4,), steps=8, device='cpu'))
            self.assertEqual(sha256(raw), original_hash)
            fixed = torch.load(fixed_path, map_location='cpu', weights_only=False)
            old_state = original.state_dict()
            new_state = fixed['state_dict']
            self.assertEqual(set(new_state) - set(old_state), {'polarity_spline_logits'})
            for key, value in old_state.items():
                self.assertTrue(torch.equal(value, new_state[key]), key)
            corrected = BleedingSceneMemoryModel(**fixed['model_kwargs']).eval()
            corrected.load_state_dict(new_state)
            history = torch.randn(2, 16, 2, 4, 4)
            times = torch.tensor([.2, .8])
            with torch.inference_mode():
                raw_latent, raw_extra = original(history, times)
                fixed_latent, fixed_extra = corrected(history, times)
            self.assertTrue(torch.equal(raw_latent, fixed_latent))
            self.assertTrue(torch.equal(raw_extra['spatial_memory_gate'], fixed_extra['spatial_memory_gate']))


if __name__ == '__main__':
    unittest.main()




