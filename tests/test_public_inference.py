"""Numerical and data-flow tests; no weights, dataset or GPU is required."""
import json
import math
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models import normal_tail_mvtec as mv, normal_tail_visa as vi, dinov2_features as dm
from models.ad_nvc_net import ScalePrototypeBank
from models.public_inference import PublicInference, _owner_excluded_distance
from models.public_metrics import disk_pixel_auc, threshold_aupro
from tools.calibrate import calibrate_resources
from tools.evaluate import verify_predictions, metrics
from tools.public_utils import sha256, write_json_new
from tests.reference_metrics import _aupro
from tests._temporary import test_directory


class NumericalTests(unittest.TestCase):
    def test_upper_tail_ties_interpolation_and_extrapolation(self):
        for op in (mv, vi):
            cal = op.fit([0., 0., 1., 2.])
            expected = -np.log(np.asarray([1., 3/5, 2/5]))
            np.testing.assert_array_equal(cal['knots'], [0., 1., 2.])
            np.testing.assert_allclose(cal['evidence'], expected, rtol=0, atol=0)
            values = [-1., 0., .5, 2., 2. + .5*float(cal['scale'])]
            np.testing.assert_allclose(op.transform(values, cal), [0., 0., expected[1]/2, expected[2], expected[2]+.5], rtol=0, atol=1e-15)

    def test_top_fraction_uses_ceil_and_visa_exact_sum(self):
        # ceil(201*.01)=3; using floor would incorrectly produce 199.5.
        self.assertEqual(mv.top_fraction(np.arange(201, dtype=np.float64)), 199.)
        self.assertEqual(vi.top_fraction(np.arange(201, dtype=np.float64)), 199.)
        values = np.zeros(501, dtype=np.float64)
        values[-6:] = [1.e16, 1., 1., 1., 1., 1.]
        expected = float(sum((Fraction.from_float(float(v)) for v in values[-6:]), Fraction()) / 6)
        rng = np.random.default_rng(19)
        for _ in range(12):
            rng.shuffle(values)
            self.assertEqual(vi.top_fraction(values), expected)
            k = math.ceil(len(values)*.01)
            self.assertEqual(mv.top_fraction(values), float(np.mean(np.partition(values, len(values)-k)[-k:])))

    def test_dino_calibration_precedes_interpolation(self):
        cal = {name: mv.fit([0., 0., 1., 2.]) for name in mv.COMPONENTS}
        cnn = np.zeros((512, 512), np.float32)
        patch = np.zeros((37, 37), np.float32)
        patch[18, 18] = 2.
        source = {'raw_global': patch, 'raw_spatial_r0': np.zeros_like(patch), 'raw_spatial_r1': np.zeros_like(patch)}
        actual = mv.pixel(cnn, source, cal)['pixel_score']
        expected = .5*mv.transform(cnn, cal['cnn']) + .5*cv2.resize(mv.transform(patch, cal['global']), (512, 512), interpolation=cv2.INTER_LINEAR)/3.
        np.testing.assert_array_equal(actual, expected)
        wrong_order = .5*mv.transform(cv2.resize(patch, (512,512), interpolation=cv2.INTER_LINEAR), cal['global'])/3.
        self.assertGreater(float(np.max(np.abs(actual-wrong_order))), .005)

    def test_last_four_layer_normalize_mean_normalize(self):
        pairs = []
        for index in range(4):
            local = torch.zeros(1, 1369, 4)
            local[..., index] = index+1
            cls = torch.tensor([[index+1., 2., 3., 4.]])
            pairs.append((local, cls))
        class Model:
            def get_intermediate_layers(self, x, **kwargs):
                assert kwargs == {'n':4, 'reshape':False, 'return_class_token':True, 'norm':True}
                return pairs
        extractor = SimpleNamespace(input_size=518, device=torch.device('cpu'), amp=False, model=Model(), _tensor=lambda rgb: torch.zeros(1))
        local, grid, cls, source = dm._extract_multilayer_patch_and_global(extractor, np.zeros((512,512,3)))
        expected_local = F.normalize(torch.stack([F.normalize(p[0][0].float(), dim=-1) for p in pairs]).mean(0), dim=-1)
        expected_cls = F.normalize(torch.stack([F.normalize(p[1][0].float(), dim=0) for p in pairs]).mean(0), dim=0)
        self.assertEqual(grid, (37,37))
        self.assertEqual(source, 'dinov2_last4_normalized_equal_mean_cls')
        torch.testing.assert_close(local, expected_local, rtol=0, atol=0)
        torch.testing.assert_close(cls, expected_cls, rtol=0, atol=0)

    def test_owner_mask_excludes_only_own_prototypes(self):
        bank = ScalePrototypeBank(['layer1'], {'layer1':3}, bank_size=4)
        bank.bank_layer1.copy_(torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [0.,1.,0.]]))
        bank.full_layer1.fill_(True)
        query = torch.tensor([[[[1.]],[[0.]],[[0.]]]])
        self.assertEqual(float(bank.distance_map('layer1', query)), 0.)
        bank._public_owner_masks = {'layer1':torch.tensor([True,False,False,False])}
        self.assertEqual(float(_owner_excluded_distance(bank, 'layer1', query)), 1.)

    def test_exact_pixel_auc_and_fast_aupro_match_independent_scan(self):
        rng = np.random.default_rng(41)
        scores = [rng.integers(0, 9, (12,15)).astype(np.float64)/8 for _ in range(4)]
        masks = [np.zeros((12,15), np.float32) for _ in range(4)]
        masks[1][1:4,2:6] = 1
        masks[2][6:9,10:13] = 1
        masks[3][1,1] = masks[3][2,2] = 1 # diagonal pixels form one 8-connected region
        with test_directory() as temp:
            area, proof = disk_pixel_auc(scores, masks, Path(temp)/'negative.npy')
        expected = roc_auc_score(np.concatenate([m.ravel() for m in masks]), np.concatenate([s.ravel() for s in scores]))
        self.assertAlmostEqual(area, expected, places=15)
        self.assertFalse(proof['pixel_sampling'])
        self.assertEqual(threshold_aupro(scores, masks, .3, 200), _aupro(scores, masks, .3, 200))


class DataFlowTests(unittest.TestCase):
    def test_calibration_uses_only_A_with_owner_exclusion(self):
        class Runtime:
            dataset = 'mvtec'
            operator = mv
            dino_source = staticmethod(PublicInference.dino_source)
            owners = []
            def raw_components(self, path, exclude_index=None):
                self.owners.append(exclude_index)
                value = float(exclude_index)/10
                return {'cnn':np.full((512,512), value, np.float32), 'global':np.full((37,37), value, np.float32),
                        'spatial_r0':np.full((37,37), value, np.float32), 'spatial_r1':np.full((37,37), value, np.float32), 'cls':np.asarray(value)}
        with test_directory() as temp:
            root = Path(temp)/'data';root.mkdir()
            for name in ('ref','a1','a2','b'):(root/name).write_bytes(name.encode())
            normal = [{'path':name,'owner':i} for i,name in enumerate(('ref','a1','a2','b'))]
            split = {'normal_inputs':normal, 'reference':[normal[0]], 'A':normal[1:3], 'B':[normal[3]]}
            runtime = Runtime()
            result = calibrate_resources(runtime, root, split, Path(temp)/'resources')
            self.assertEqual(runtime.owners, [1,2])
            self.assertFalse(result['normal_B_used_for_fitting'])
            self.assertEqual(int(runtime.calibration['cnn']['count']), 2*4096)
            self.assertEqual(int(runtime.calibration['global']['count']), 2*37*37)
            self.assertEqual(int(runtime.calibration['top']['count']), 2)
            self.assertEqual(int(runtime.calibration['cls']['count']), 2)

    def test_tampering_rejected_before_ground_truth_loading(self):
        with test_directory() as temp:
            root = Path(temp)/'data';root.mkdir()
            image = root/'image.png';image.write_bytes(b'input')
            pred = Path(temp)/'run';pred.mkdir()
            file = pred/'0.npz';file.write_bytes(b'prediction')
            manifest = {'dataset':'mvtec','category':'bottle','layout':'prepared','GT_masks_read':False,
                        'rows':[{'input':'image.png','input_sha256':sha256(image),'prediction':'0.npz','prediction_sha256':sha256(file)}]}
            write_json_new(pred/'predictions.json', manifest)
            write_json_new(pred/'prediction_seal.json', {'schema':'adnvcnet_public_prediction_seal_v1','GT_masks_read':False,
                           'manifest':'predictions.json','manifest_sha256':sha256(pred/'predictions.json'),'count':1})
            self.assertEqual(verify_predictions(pred, root)['category'], 'bottle')
            file.write_bytes(b'modified')
            with patch('tools.evaluate.ground_truth_rows') as gt:
                with self.assertRaisesRegex(ValueError, 'changed'):
                    metrics(SimpleNamespace(predictions=pred,data_root=root,output=None,split_file=None))
                gt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
