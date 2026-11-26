"""CPU checks for portable normal resource construction; no dataset or GT."""
import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid

import numpy as np

from models.resource_builders import (align_normal_paths, balanced_positions,
    balanced_quotas, build_balanced_bank, build_dino_memories, load_resource_profile,
    normal_split, save_npz)


@contextmanager
def test_directory():
    # Ordinary inherited directory permissions also work in managed Windows
    # workspaces where tempfile's restrictive mkdir mode is unsupported.
    base = Path(os.environ.get('ADNVCNET_TEST_TEMP_ROOT', tempfile.gettempdir())).resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = base / ('adnvcnet_resource_test_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.resolve().parent != base or not root.name.startswith('adnvcnet_resource_test_'):
            raise RuntimeError('Refusing cleanup outside the temporary test directory')
        shutil.rmtree(root)


class ResourceBuilderTests(unittest.TestCase):
    def test_quotas_cover_every_normal_owner(self):
        quotas = balanced_quotas(209)
        self.assertEqual(sum(quotas), 24576)
        self.assertEqual(quotas[:123], [118] * 123)
        self.assertEqual(quotas[123:], [117] * 86)
        self.assertEqual(balanced_quotas(2), [512, 512])
        self.assertEqual(balanced_quotas(60), [410] * 36 + [409] * 24)

    def test_content_alignment_preserves_duplicate_owners(self):
        with test_directory() as folder:
            root = Path(folder)
            paths = [root / name for name in ['a.bin', 'b.bin', 'c.bin']]
            for path, value in zip(paths, [b'alpha', b'beta', b'alpha']):
                path.write_bytes(value)
            h1 = hashlib.sha256(b'alpha').hexdigest();h2 = hashlib.sha256(b'beta').hexdigest()
            ordered, hashes = align_normal_paths(paths, {'normal_count': 3, 'normal_sha256': [h2, h1, h1]})
            self.assertEqual(ordered, [paths[1], paths[0], paths[2]])
            self.assertEqual(hashes, [h2, h1, h1])
            with self.assertRaises(ValueError):
                align_normal_paths(paths[:2], {'normal_count': 3, 'normal_sha256': [h2, h1, h1]})

    def test_all_fixed_profiles_reconstruct_the_declared_splits(self):
        categories = {
            'mvtec': ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper'],
            'visa': ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']}
        with test_directory() as folder:
            root = Path(folder)
            for ds, names in categories.items():
                for category in names:
                    profile = load_resource_profile(ds, category)
                    paths = [root / category / f'{index}.png' for index in range(profile['normal_count'])]
                    split = normal_split(paths, profile['normal_sha256'], profile, root)
                    self.assertEqual([x['owner'] for x in split['A']], profile['A_indices'])
                    self.assertEqual([x['owner'] for x in split['B']], profile['B_indices'])
                    self.assertFalse({x['owner'] for x in split['reference']} & {x['owner'] for x in split['calibration']})
                    self.assertEqual(len(split['reference']) + len(split['calibration']), len(paths))

    def test_partial_bank_owns_rows_and_preserves_selected_features(self):
        import torch
        import torch.nn.functional as F
        features = {}
        for layer, shape in [('layer1',(1,64,128,128)),('layer2',(1,128,64,64)),('layer3',(1,256,32,32))]:
            generator = torch.Generator().manual_seed(42 + shape[1])
            features[layer] = torch.rand(shape, generator=generator)
        paths = [Path('first'), Path('second')]
        hashes = ['0' * 64, '1' * 64]
        result = build_balanced_bank(lambda _path: features, paths, hashes)
        for layer, spatial in [('layer1',16384),('layer2',4096),('layer3',1024)]:
            self.assertEqual(result['bank_' + layer].shape[0], 24576)
            self.assertEqual(int(result['ptr_' + layer][0]), 1024)
            self.assertFalse(bool(result['full_' + layer][0]))
            np.testing.assert_array_equal(result['origin_' + layer][:1024], [0] * 512 + [1] * 512)
            self.assertTrue((result['origin_' + layer][1024:] == -1).all())
            indices = balanced_positions(hashes[0], layer, spatial, 512)
            expected = F.normalize(features[layer], dim=1)[0].permute(1, 2, 0).reshape(spatial, -1)[torch.from_numpy(indices)].numpy()
            np.testing.assert_array_equal(result['bank_' + layer][:512], expected)
        with test_directory() as folder:
            receipt = save_npz(Path(folder) / 'bank.npz', result)
            self.assertEqual(receipt['arrays']['origin_layer1']['dtype'], 'int64')
            with self.assertRaises(FileExistsError):
                save_npz(Path(folder) / 'bank.npz', result)

    def test_dino_builder_keeps_all_spatial_and_cls_references(self):
        import torch
        import torch.nn.functional as F
        references = [Path('0'), Path('1'), Path('2')]
        def extract(path):
            offset = int(str(path))
            patches = F.normalize(torch.arange(1, 25, dtype=torch.float32).reshape(6, 4) + offset, dim=-1)
            cls = F.normalize(torch.arange(1, 5, dtype=torch.float32) + offset, dim=0)
            return patches, (2, 3), cls, 'test_features'
        a = build_dino_memories(extract, references, 13027, patches_per_image=3, max_patches=4, device='cpu')
        b = build_dino_memories(extract, references, 13027, patches_per_image=3, max_patches=4, device='cpu')
        self.assertEqual(a['global_memory'].shape, (4, 4))
        self.assertEqual(a['spatial_bank'].shape, (3, 2, 3, 4))
        self.assertEqual(a['image_memory'].shape, (3, 4))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a[key].dtype, np.float32)
            np.testing.assert_allclose(np.linalg.norm(a[key], axis=-1), 1, atol=2e-7)
        with self.assertRaises(ValueError):
            build_dino_memories(extract, [], 13027, device='cpu')


if __name__ == '__main__':
    unittest.main()
