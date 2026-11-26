"""Timing accounting, deterministic selection, and instrumentation parity."""
import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.profile import Clock, Instrument, PHASES, selected_inputs, summarize, timed_call
from tools.public_utils import sha256


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[2] / ('profile_test_' + uuid.uuid4().hex)
        self.scratch.mkdir()

    def tearDown(self):
        shutil.rmtree(self.scratch)

    def test_nested_exclusive_partition(self):
        ticks = iter((0, 1, 2, 3, 4, 5, 6, 7))
        barriers = []
        clock = Clock(lambda: barriers.append(1), lambda: next(ticks))
        clock.reset()
        with clock.span('total'):
            with clock.span('cnn_feature_response'):
                with clock.span('matching_raw'):
                    pass
        result = clock.finish(8)
        self.assertEqual(result['total_ns'], sum(result['exclusive_wall_ns'].values()))
        self.assertEqual(result['exclusive_wall_ns']['matching_raw'], 1)
        self.assertEqual(result['exclusive_wall_ns']['cnn_feature_response'], 2)
        self.assertEqual(len(barriers), 4)
        self.assertEqual(result['synchronization_calls'], 6)

    def test_first_test_paths_no_mask_reads(self):
        for relative in ('bottle/test/good/003.png', 'bottle/test/good/001.png', 'bottle/test/broken/002.png'):
            path = self.scratch / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (2, 2)).save(path)
        # The malformed annotation must never be opened by a timing plan.
        mask = self.scratch / 'bottle/ground_truth/broken/002_mask.png'
        mask.parent.mkdir(parents=True)
        mask.write_text('not an image')
        plan = selected_inputs(self.scratch, 'mvtec', 'bottle', measurements=2, warmup=1)
        self.assertEqual([row['input'] for row in plan['measurements']],
                         ['bottle/test/broken/002.png', 'bottle/test/good/001.png'])
        self.assertEqual(plan['warmup'], plan['measurements'][:1])
        self.assertFalse(plan['GT_masks_read'])

    def test_verified_input_change(self):
        path = self.scratch / 'input.bin'
        path.write_bytes(b'before')
        digest = sha256(path)
        path.write_bytes(b'after')
        clock = Clock(lambda: None)
        with self.assertRaisesRegex(ValueError, 'changed after planning'):
            timed_call(SimpleNamespace(predict=lambda x: None), path, digest, clock, lambda: None)
        self.assertFalse(clock.active)

    def test_wrapping_restores_original_output_and_attributes(self):
        from models import public_inference as api
        import torch
        bank = SimpleNamespace(distance_map=lambda value: value + 1)
        extractor = SimpleNamespace(_tensor=lambda value: value)
        class FakeRuntime:
            def __init__(self):
                self.model = SimpleNamespace(prototype_bank=bank)
                self.extractor = extractor
            def cnn_raw(self, value):
                return bank.distance_map(value)
            def dino_features(self, value):
                return value * 2
            def score_raw(self, value):
                return value + 3
            def predict(self, value):
                return self.score_raw(self.cnn_raw(value) + self.dino_features(value))
        runtime = FakeRuntime()
        original_to, original_rgb = torch.Tensor.to, api._rgb512
        expected = runtime.predict(4)
        clock = Clock(lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'restore'):
            with Instrument(runtime, clock):
                clock.reset()
                with clock.span('total'):
                    self.assertEqual(runtime.predict(4), expected)
                self.assertNotIn('host_to_device', [event['label'] for event in clock.events])
                raise RuntimeError('restore')
        self.assertNotIn('cnn_raw', vars(runtime))
        self.assertIs(torch.Tensor.to, original_to)
        self.assertIs(api._rgb512, original_rgb)
        self.assertEqual(runtime.predict(4), expected)

    def test_equal_category_mean_and_pooled_quantile(self):
        items = []
        for mean, rows in ((1.0, [1.0]), (3.0, [2.0, 4.0])):
            items.append(dict(dataset='mvtec', total_mean_ms=mean,
                rows=[dict(total_ns=int(value * 1e6)) for value in rows],
                stage_mean_ms={phase: mean / len(PHASES) for phase in PHASES},
                peak_allocated_GiB=1., peak_reserved_GiB=2., loading_validation_calibration_ms=10.))
        result = summarize(items)['mvtec']
        self.assertEqual(result['latency_ms']['mean'], 2.0)
        self.assertEqual(result['latency_ms']['median'], 2.0)
        self.assertAlmostEqual(sum(result['stage_fraction'].values()), 1.)


if __name__ == '__main__':
    unittest.main()
