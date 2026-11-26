"""Spatial retrieval, normal fitting and prediction sealing without a model."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.dinov2_features import _raw_spatial
from models.scoring_controls import baseline, operator
from models.spatial_controls import (ARMS, STEPS, fit_controls, neighbors,
    ring_response, score_control, score_map)
from tools.public_utils import sha256, write_json_new
from tools.spatial_controls import fit_resources, load_fits, predict_controls
from tools.evaluate import verify_predictions
from tests._temporary import test_directory


class SpatialControlTests(unittest.TestCase):
    def test_eight_offsets_match_independent_border_search_and_precision(self):
        generator = torch.Generator().manual_seed(19)
        patches = F.normalize(torch.rand(12, 7, generator=generator), dim=-1)
        bank = F.normalize(torch.rand(2, 3, 4, 7, generator=generator), dim=-1)
        for dtype in (torch.float32, torch.float64):
            query, memory = patches.to(dtype), bank.to(dtype)
            actual = neighbors(query, memory)
            expected = np.empty((3, 4), np.float32)
            for y in range(3):
                for x in range(4):
                    candidates = []
                    for dy, dx in STEPS:
                        if 0 <= y+dy < 3 and 0 <= x+dx < 4:
                            # einsum preserves the accepted dot-product reduction;
                            # the candidate enumeration is independently scalar.
                            candidates.extend(torch.einsum('c,kc->k', query[y*4+x], memory[:, y+dy, x+dx]).tolist())
                    expected[y, x] = 1.-np.float32(max(candidates))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
            self.assertEqual(actual.dtype, np.float32)
            s0 = _raw_spatial(query, (3,4), memory, 0)
            s1 = _raw_spatial(query, (3,4), memory, 1)
            np.testing.assert_array_equal(np.minimum(s0, actual), s1)

    def test_center_is_excluded_even_when_it_is_the_best_match(self):
        patches = torch.eye(9)
        bank = patches.reshape(1,3,3,9).clone()
        np.testing.assert_array_equal(_raw_spatial(patches,(3,3),bank,0),np.zeros((3,3)))
        np.testing.assert_array_equal(neighbors(patches,bank),np.ones((3,3)))

    def test_fitting_reuses_A_and_keeps_CLS_and_pixel_operation_order(self):
        rng = np.random.default_rng(29)
        rows = [dict(cnn=rng.random((512,512),dtype=np.float32),
                     **{name:rng.random((37,37),dtype=np.float32) for name in ('global','spatial_r0','spatial_r1')},
                     cls=np.asarray(.1+i*.02)) for i in range(3)]
        rings = [rng.random((37,37),dtype=np.float32) for _ in rows]
        for dataset in ('mvtec','visa'):
            op=operator(dataset)
            cal={name:op.fit(np.concatenate([op.sample(row[name]) for row in rows])) for name in op.COMPONENTS}
            cal['cls']=op.fit([row['cls'] for row in rows])
            from models.scoring_controls import dino_source
            cal['top']=op.fit([op.pixel(row['cnn'],dino_source(row),cal)['pixel_top1'] for row in rows])
            fits,_=fit_controls(rows,rings,cal,dataset)
            for arm in ARMS:
                result=score_control(rows[0],rings[0],cal,fits,dataset,arm)
                self.assertEqual(float(result['image_cls_evidence']),float(op.transform(rows[0]['cls'],cal['cls'])))
            alternate={**dino_source(rows[0]),'raw_spatial_r0':rows[0]['spatial_r1']}
            alternate_cal={**cal,'spatial_r0':cal['spatial_r1']}
            np.testing.assert_array_equal(score_map(op,'S1_DOUBLE',rows[0]['cnn'],dino_source(rows[0]),cal),
                                          op.pixel(rows[0]['cnn'],alternate,alternate_cal)['pixel_score'])
            with self.assertRaises(ValueError):fit_controls(rows[:1],rings[:1],cal,dataset)

    def test_normal_fit_and_all_arms_produce_standard_seals_without_GT(self):
        generator=torch.Generator().manual_seed(71)
        bank=F.normalize(torch.rand(1,37,37,384,generator=generator),dim=-1)
        patch=F.normalize(torch.rand(1369,384,generator=generator),dim=-1)
        runtime=SimpleNamespace(dataset='mvtec',match_memory={'spatial_bank':bank},
            dino_features=lambda path:(patch,(37,37),torch.zeros(384),'dinov2_last4_normalized_equal_mean_cls'))
        s0=_raw_spatial(patch,(37,37),bank,0);s1=_raw_spatial(patch,(37,37),bank,1)
        rng=np.random.default_rng(99)
        raw=dict(cnn=rng.random((512,512),dtype=np.float32),global_=None)
        raw.pop('global_');raw.update({'global':rng.random((37,37),dtype=np.float32),
            'spatial_r0':s0,'spatial_r1':s1,'cls':np.asarray(.12)})
        op=operator('mvtec')
        cal={name:op.fit(op.sample(raw[name])) for name in op.COMPONENTS}
        from models.scoring_controls import dino_source
        cal['top']=op.fit([op.pixel(raw['cnn'],dino_source(raw),cal)['pixel_top1']])
        cal['cls']=op.fit([raw['cls']])
        with test_directory() as folder:
            data=folder/'data';data.mkdir();resources=folder/'resources';resources.mkdir()
            (resources/'calibration_raw').mkdir();source=folder/'source';(source/'predictions').mkdir(parents=True)
            for name in ('normal.png','input.png'):Image.new('RGB',(8,8),(50,100,150)).save(data/name)
            op.save_calibration(resources/'calibration.npz',cal)
            write_json_new(resources/'profile.json',dict(training_seed=3407,dataset=dict(name='mvtec',category='bottle')))
            for name in ('bank.npz','memories.npz'):np.savez(resources/name,fixture=np.asarray([1.]))
            raw_path=resources/'calibration_raw/000000.npz';np.savez_compressed(raw_path,**raw)
            row=dict(path='normal.png',owner=0,sha256=sha256(data/'normal.png'))
            write_json_new(resources/'normal_split.json',dict(A=[row],normal_inputs=[row],reference=[],B=[]))
            write_json_new(resources/'calibration.json',dict(dataset='mvtec',normal_only=True,
                normal_B_used_for_fitting=False,sha256=sha256(resources/'calibration.npz'),
                inputs=[dict(row,raw_file='calibration_raw/000000.npz',raw_sha256=sha256(raw_path))]))
            tail=baseline(raw,cal,'mvtec');target=source/'predictions/000000.npz'
            np.savez_compressed(target,**dict(tail,anomaly_map=tail['pixel_score']),**{'raw_'+k:v for k,v in raw.items()})
            identity=dict(checkpoint_sha256='synthetic_checkpoint',dino_weights_sha256='synthetic_weights')
            from tools.spatial_controls import resource_identity
            write_json_new(source/'predictions.json',dict(schema='adnvcnet_public_predictions_v1',
                dataset='mvtec',category='bottle',training_seed=3407,variant='A0',layout='prepared',
                save_components=True,GT_masks_read=False,count=1,checkpoint_sha256=identity['checkpoint_sha256'],
                resource_sha256=resource_identity(resources),rows=[dict(input='input.png',input_sha256=sha256(data/'input.png'),
                    prediction='predictions/000000.npz',prediction_sha256=sha256(target))]))
            write_json_new(source/'prediction_seal.json',dict(schema='adnvcnet_public_prediction_seal_v1',
                manifest='predictions.json',manifest_sha256=sha256(source/'predictions.json'),count=1,GT_masks_read=False))
            # Any mask-loading code path would fail this workflow test.
            with mock_patch('tools.evaluate.ground_truth_rows',side_effect=AssertionError('No test GT before seals')):
                fit=fit_resources(runtime,resources=resources,data_root=data,output=folder/'fits',model_identity=identity)
                self.assertTrue(fit['normal_A0_Top_fit_exact'])
                out=predict_controls(runtime,resources=resources,data_root=data,fits_dir=folder/'fits',
                    predictions=source,output=folder/'controls',model_identity=identity)
                for arm in ARMS:
                    manifest=verify_predictions(out/arm,data)
                    self.assertEqual(manifest['variant'],arm)
                    self.assertEqual(manifest['count'],1)
                    self.assertFalse(manifest['GT_masks_read'])
            (folder/'fits/normal_s8/000000.npz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):load_fits(folder/'fits',resources,identity)


if __name__=='__main__':unittest.main()
