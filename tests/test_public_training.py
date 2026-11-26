"""Small CPU checks; no dataset, pretrained download, or trained weights."""
import copy
from contextlib import contextmanager
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.public_data import normal_samples, load_rgb_tensor, order_by_sha, file_sha256
from models.public_training import PseudoAnomalyGenerator, load_training_profile, loss_terms


@contextmanager
def test_directory():
    # Use ordinary inherited permissions in managed Windows workspaces.
    base=Path(__file__).resolve().parents[1]
    folder=base/('.test_training_'+uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent!=base or not folder.name.startswith('.test_training_'):
            raise RuntimeError('Refusing cleanup outside the temporary test directory')
        shutil.rmtree(folder)


class PublicTrainingTests(unittest.TestCase):
    def test_train_and_evaluation_parameters_are_distinct(self):
        mvtec = load_training_profile('mvtec','cable',3409)
        self.assertEqual(mvtec['model']['score_proto_weight'],.48)
        self.assertEqual(mvtec['evaluation_model']['score_proto_weight'],.35)
        self.assertEqual(mvtec['model']['layer_weights'],{'layer1':.42,'layer2':.42,'layer3':.16})
        self.assertEqual(mvtec['evaluation_model']['layer_weights'],{'layer1':.28,'layer2':.42,'layer3':.3})
        self.assertEqual(mvtec['train']['seed'],3409)
        self.assertEqual(load_training_profile('visa','candle')['model']['score_proto_weight'],.45)

    def test_supervised_term_detaches_clean_discrepancy(self):
        cfg=load_training_profile('mvtec','bottle')['pseudo']
        cfg=copy.deepcopy(cfg)
        cfg['normal_align_weight']=cfg['normal_score_weight']=0.0
        clean={'disc_score_map':torch.full((1,1,2,2),.2,requires_grad=True),
               'raw_score_map':torch.ones((1,1,2,2),requires_grad=True),
               'feature_align_loss':torch.ones((),requires_grad=True)}
        pseudo={'disc_score_map':torch.full((1,1,2,2),.4,requires_grad=True),
                'raw_score_map':torch.ones((1,1,2,2),requires_grad=True),
                'feature_align_loss':torch.ones((),requires_grad=True)}
        mask=torch.tensor([[[[1.,0.],[0.,1.]]]])
        values=loss_terms(clean,pseudo,mask,cfg)
        values['train_loss'].backward()
        self.assertIsNone(clean['disc_score_map'].grad)
        self.assertTrue(torch.isfinite(pseudo['disc_score_map'].grad).all())
        self.assertGreater(pseudo['disc_score_map'].grad.abs().sum().item(),0)
        self.assertTrue(all(torch.isfinite(v) for v in values.values()))

    def test_pseudo_rng_and_hard_copy_paste(self):
        cfg=copy.deepcopy(load_training_profile('visa','candle')['pseudo'])
        cfg['modes']=['copy_paste'];cfg['category_modes']={};cfg['enable_prob']=1.
        image=torch.from_numpy(np.random.default_rng(7).random((2,3,64,64),dtype=np.float32))
        a=PseudoAnomalyGenerator(cfg,'candle',3407).generate(image,1,0)
        b=PseudoAnomalyGenerator(cfg,'candle',3407).generate(image,1,0)
        c=PseudoAnomalyGenerator(cfg,'candle',3408).generate(image,1,0)
        self.assertTrue(torch.equal(a.images,b.images))
        self.assertTrue(torch.equal(a.masks,b.masks))
        self.assertFalse(torch.equal(a.images,c.images))
        self.assertTrue(torch.equal(a.masks,a.masks.round()))
        outside=(a.masks==0).expand_as(image)
        self.assertTrue(torch.equal(a.images[outside],image[outside]))

    def test_normal_reader_does_not_open_test_data(self):
        with test_directory() as tmp:
            root=Path(tmp);normal=root/'bottle/train/good';normal.mkdir(parents=True)
            Image.fromarray(np.full((9,13,3),117,dtype=np.uint8)).save(normal/'000.png')
            (root/'bottle/test').mkdir()
            (root/'bottle/test/invalid_image.png').write_text('unreadable test data')
            records=normal_samples(root,'mvtec','bottle')
            self.assertEqual(len(records),1)
            self.assertEqual(records[0].relative_path,'bottle/train/good/000.png')
            self.assertEqual(tuple(load_rgb_tensor(records[0].path).shape),(3,512,512))
            self.assertEqual(order_by_sha(records,[file_sha256(records[0].path)]),records)


if __name__=='__main__':
    unittest.main()
